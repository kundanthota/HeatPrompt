#!/usr/bin/env python3
"""Simple regression analysis comparing simulated demand to Energieatlas ground truth."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

LOGGER = logging.getLogger(__name__)

ERA_TO_NUM = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
}


def _slugify(value: object) -> str:
    """Normalise column names into safe feature identifiers."""

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def _load_gpt_features(path: Path) -> pd.DataFrame:
    """Load GPT-derived features keyed by Energieatlas aggregation ID."""

    if not path.exists():
        raise FileNotFoundError(f"GPT feature file not found: {path}")

    gpt = pd.read_csv(path, index_col=0)
    if gpt.empty:
        raise ValueError(f"GPT feature file '{path}' is empty")

    gpt.index = gpt.index.map(int)
    gpt.index.name = "id_agg"
    gpt = gpt.apply(pd.to_numeric, errors="coerce")
    gpt.columns = [f"gpt_{_slugify(col)}" for col in gpt.columns]
    return gpt


def _build_tabula_type_shares(buildings: pd.DataFrame) -> pd.DataFrame:
    """Return share of buildings per TABULA type for each Energieatlas cell."""

    counts = (
        buildings.groupby(["id_agg", "tabula_type"]).size().unstack(fill_value=0)
        if not buildings.empty
        else pd.DataFrame()
    )
    if counts.empty:
        return counts

    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.columns = [f"type_share_{col.lower()}" for col in shares.columns]
    return shares


def _build_era_shares(buildings: pd.DataFrame) -> pd.DataFrame:
    """Return share of buildings per TABULA era for each Energieatlas cell."""

    counts = (
        buildings.groupby(["id_agg", "tabula_era"]).size().unstack(fill_value=0)
        if not buildings.empty
        else pd.DataFrame()
    )
    if counts.empty:
        return counts

    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.columns = [f"era_share_{col}" for col in shares.columns]
    return shares


def _summarise_building_attributes(buildings: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated building statistics per Energieatlas cell."""

    if buildings.empty:
        return pd.DataFrame()

    numeric_buildings = buildings.copy()
    numeric_buildings["tabula_era_num"] = numeric_buildings["tabula_era"].map(ERA_TO_NUM)

    aggregations = {
        "measuredHeight": ["mean", "std"],
        "n_floors": ["mean", "max"],
        "floor_area_m2": ["mean"],
        "footprint_area_m2": ["mean"],
        "tabula_era_num": ["mean", "std"],
        "q_h_kwh_m2a": ["mean", "std"],
    }

    grouped = numeric_buildings.groupby("id_agg").agg(aggregations)
    grouped.columns = ["_".join(filter(None, col)).strip("_") for col in grouped.columns]
    return grouped


def prepare_dataset(
    buildings_path: Path,
    summary_path: Path,
    gpt_features_path: Path | None = None,
) -> pd.DataFrame:
    """Load source CSVs and prepare a modelling dataset keyed by Energieatlas id."""

    buildings = pd.read_csv(buildings_path)
    summary = pd.read_csv(summary_path)

    buildings = buildings.dropna(subset=["id_agg", "wb_gs"])
    summary = summary.dropna(subset=["id_agg"])

    buildings["id_agg"] = buildings["id_agg"].astype(int)
    summary["id_agg"] = summary["id_agg"].astype(int)

    summary = summary.set_index("id_agg")

    target = (
        buildings.groupby("id_agg")["wb_gs"].mean().rename("wb_gs")
        if not buildings.empty
        else pd.Series(dtype=float)
    )

    features = [
        summary,
        _summarise_building_attributes(buildings),
        _build_tabula_type_shares(buildings),
        _build_era_shares(buildings),
    ]
    feature_frame = pd.concat(features, axis=1)
    dataset = feature_frame.join(target, how="inner")

    if gpt_features_path is not None:
        try:
            gpt_features = _load_gpt_features(gpt_features_path)
        except FileNotFoundError:
            LOGGER.warning(
                "GPT features file '%s' not found; continuing without additional features",
                gpt_features_path,
            )
        except ValueError as exc:
            LOGGER.warning("%s; continuing without GPT features", exc)
        else:
            dataset = dataset.join(gpt_features, how="left")
            LOGGER.info(
                "Joined %d GPT feature columns from '%s'",
                gpt_features.shape[1],
                gpt_features_path,
            )

    if dataset.empty:
        raise ValueError("No overlapping Energieatlas IDs between inputs; dataset empty")

    numeric_cols = dataset.columns.difference(["wb_gs"])
    dataset[numeric_cols] = dataset[numeric_cols].replace([np.inf, -np.inf], np.nan)

    for col in numeric_cols:
        series = dataset[col]
        if series.isna().all():
            dataset[col] = series.fillna(0.0)
        else:
            dataset[col] = series.fillna(series.median())

    return dataset.reset_index()


def _select_regressor(name: str, ridge_alpha: float = 1.0):
    lowered = name.lower()
    if lowered == "ridge":
        return Ridge(alpha=ridge_alpha)
    if lowered in {"gradient_boosting", "gbr"}:
        return GradientBoostingRegressor(random_state=42)
    if lowered in {"random_forest", "rf"}:
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    if lowered == "svr":
        return SVR(kernel="rbf", C=10.0, gamma="scale")
    raise ValueError(
        "Unknown regressor '%s'. Choose from ridge, gradient_boosting, random_forest, svr"
        % name
    )


def run_regression(
    dataset: pd.DataFrame,
    cv_splits: int = 3,
    pca_components: int | None = None,
    pca_variance: float | None = None,
    regressor: str = "ridge",
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Perform k-fold cross-validation using Ridge regression."""

    feature_cols = dataset.columns.difference(["id_agg", "wb_gs"])
    X = dataset[feature_cols].values
    y = dataset["wb_gs"].values

    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if pca_components is not None or pca_variance is not None:
        n_components: int | float
        pca_label: str
        if pca_components is not None:
            n_components = pca_components
            pca_label = f"n_components={pca_components}"
        else:
            n_components = pca_variance if pca_variance is not None else 0.95
            pca_label = f"variance={n_components}"
        LOGGER.info(
            "Applying PCA (%s) to %d raw features",
            pca_label,
            X.shape[1],
        )
        steps.append(("pca", PCA(n_components=n_components)))
    model = _select_regressor(regressor, ridge_alpha=ridge_alpha)
    needs_scaling = isinstance(model, SVR)
    if needs_scaling and steps[0][0] != "scaler":
        steps.insert(0, ("scaler", StandardScaler()))
    steps.append(("model", model))

    pipeline = Pipeline(steps)

    n_samples = len(dataset)
    splits = min(cv_splits, n_samples)
    if splits < 2:
        raise ValueError("Need at least two samples for cross-validation")

    cv = KFold(n_splits=splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    return scores, float(scores.mean()), float(scores.std())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare simulated demand with Energieatlas ground truth using regression",
    )
    parser.add_argument(
        "--buildings",
        type=Path,
        default=Path("buildings_output.csv"),
        help="Path to building-level CSV export",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("energy_summary.csv"),
        help="Path to aggregated energy CSV export",
    )
    parser.add_argument(
        "--gpt-features",
        type=Path,
        default=Path("gpt_features.csv"),
        help="Optional path to GPT feature CSV (default: gpt_features.csv)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Apply PCA with this many components before regression",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=None,
        help="Apply PCA retaining this fraction of explained variance (0 < v ≤ 1)",
    )
    parser.add_argument(
        "--regressor",
        type=str,
        default="ridge",
        help="Regression model to use: ridge, gradient_boosting, random_forest, svr",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Regularisation strength for Ridge regression (ignored for other models)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output",
    )
    args = parser.parse_args(argv)
    if args.pca_components is not None and args.pca_components <= 0:
        parser.error("--pca-components must be a positive integer")
    if args.pca_variance is not None and not (0.0 < args.pca_variance <= 1.0):
        parser.error("--pca-variance must be between 0 and 1")
    if args.pca_components is not None and args.pca_variance is not None:
        parser.error("Specify either --pca-components or --pca-variance, not both")
    if args.regressor.lower() not in {"ridge", "gradient_boosting", "gbr", "random_forest", "rf", "svr"}:
        parser.error("--regressor must be one of: ridge, gradient_boosting, random_forest, svr")
    if args.ridge_alpha <= 0:
        parser.error("--ridge-alpha must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    LOGGER.info("Loading data from \"%s\" and \"%s\"", args.buildings, args.summary)
    dataset = prepare_dataset(args.buildings, args.summary, args.gpt_features)
    LOGGER.info(
        "Prepared dataset with %d Energieatlas cells and %d features",
        len(dataset),
        dataset.shape[1] - 2,
    )

    scores, mean_score, std_score = run_regression(
        dataset,
        cv_splits=args.cv,
        pca_components=args.pca_components,
        pca_variance=args.pca_variance,
        regressor=args.regressor,
        ridge_alpha=args.ridge_alpha,
    )

    for idx, score in enumerate(scores, start=1):
        LOGGER.info("Fold %d R^2: %.3f", idx, score)

    LOGGER.info(
        "Mean R^2 over %d folds: %.3f ± %.3f",
        len(scores),
        mean_score,
        std_score,
    )


if __name__ == "__main__":
    main()
