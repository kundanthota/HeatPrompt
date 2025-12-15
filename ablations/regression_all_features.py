#!/usr/bin/env python3
"""Regression analysis using building composition features to predict Energieatlas demand."""

from __future__ import annotations

import argparse
import json
import logging
import re
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


def _fill_numeric(dataset: pd.DataFrame) -> pd.DataFrame:
    filled = dataset.copy()
    numeric_cols = filled.columns.difference(["wb_gs"])
    filled[numeric_cols] = filled[numeric_cols].replace([np.inf, -np.inf], np.nan)

    for col in numeric_cols:
        series = filled[col]
        filled[col] = series.fillna(0.0 if series.isna().all() else series.median())

    return filled


def _slugify(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def _normalise_construction_year(values: pd.Series) -> pd.Series:
    """Parse text or numeric construction years into a single numeric estimate."""

    parsed: list[float | None] = []
    for raw in values:
        if pd.isna(raw):
            parsed.append(np.nan)
            continue
        if isinstance(raw, (int, float)) and not pd.isna(raw):
            parsed.append(float(raw))
            continue
        text = str(raw).strip()
        digits = re.findall(r"\d{4}", text)
        if not digits:
            parsed.append(np.nan)
            continue
        years = sorted({int(token) for token in digits})
        if len(years) == 1:
            parsed.append(float(years[0]))
        else:
            parsed.append(sum(years) / len(years))
    return pd.Series(parsed, index=values.index)


def _build_category_features(buildings: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    data = buildings.dropna(subset=[column])
    if data.empty:
        return pd.DataFrame()

    counts = data.groupby(["id_agg", column]).size().unstack(fill_value=0)
    counts = counts.rename(columns=lambda val: f"{prefix}_count_{_slugify(val)}")

    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    shares = shares.rename(columns=lambda col: col.replace("_count_", "_share_"))
    return pd.concat([counts, shares], axis=1)


def _summarise_buildings(buildings: pd.DataFrame) -> pd.DataFrame:
    group = buildings.groupby("id_agg")
    features = pd.DataFrame(index=group.size().index)
    features["n_buildings"] = group.size()

    def add_stat(col: str, stat: str, name: str) -> None:
        if col in buildings.columns:
            features[name] = group[col].agg(stat)

    numeric_cols = [
        "measuredHeight",
        "n_floors",
        "floor_area_m2",
        "footprint_area_m2",
    "construction_year_numeric",
    "construction_year",
    "year_built",
    "built_year",
        "estimated_storey_height",
        "floor_to_footprint_ratio",
    ]
    for col in numeric_cols:
        if col in buildings.columns:
            buildings[col] = pd.to_numeric(buildings[col], errors="coerce")

    if "height" in buildings.columns and "measuredHeight" not in buildings.columns:
        buildings["measuredHeight"] = pd.to_numeric(buildings["height"], errors="coerce")

    if "floors" in buildings.columns and "n_floors" not in buildings.columns:
        buildings["n_floors"] = pd.to_numeric(buildings["floors"], errors="coerce")

    if {"measuredHeight", "n_floors"}.issubset(buildings.columns):
        floors = buildings["n_floors"].replace({0: np.nan})
        buildings["estimated_storey_height"] = buildings["measuredHeight"] / floors
    else:
        buildings["estimated_storey_height"] = np.nan

    if {"floor_area_m2", "footprint_area_m2"}.issubset(buildings.columns):
        denom = buildings["footprint_area_m2"].replace({0: np.nan})
        buildings["floor_to_footprint_ratio"] = buildings["floor_area_m2"] / denom
    else:
        buildings["floor_to_footprint_ratio"] = np.nan

    add_stat("footprint_area_m2", "sum", "footprint_area_m2_sum")
    add_stat("footprint_area_m2", "mean", "footprint_area_m2_mean")
    add_stat("footprint_area_m2", "median", "footprint_area_m2_median")
    add_stat("floor_area_m2", "sum", "floor_area_m2_sum")
    add_stat("floor_area_m2", "mean", "floor_area_m2_mean")
    add_stat("floor_area_m2", "median", "floor_area_m2_median")
    add_stat("n_floors", "mean", "n_floors_mean")
    add_stat("n_floors", "median", "n_floors_median")
    add_stat("n_floors", "max", "n_floors_max")
    add_stat("measuredHeight", "mean", "height_mean")
    add_stat("measuredHeight", "median", "height_median")
    add_stat("measuredHeight", "std", "height_std")
    add_stat("measuredHeight", "max", "height_max")
    add_stat("estimated_storey_height", "mean", "storey_height_mean")
    add_stat("estimated_storey_height", "median", "storey_height_median")
    add_stat("floor_to_footprint_ratio", "mean", "floor_to_footprint_ratio_mean")
    add_stat("floor_to_footprint_ratio", "median", "floor_to_footprint_ratio_median")

    year_columns = [
        col
        for col in ("construction_year", "year_built", "built_year")
        if col in buildings.columns
    ]
    for col in year_columns:
        add_stat(col, "mean", f"{col}_mean")
        add_stat(col, "median", f"{col}_median")
        add_stat(col, "min", f"{col}_min")
        add_stat(col, "max", f"{col}_max")

    add_stat("construction_year_numeric", "mean", "construction_year_mean")
    add_stat("construction_year_numeric", "median", "construction_year_median")
    add_stat("construction_year_numeric", "min", "construction_year_min")
    add_stat("construction_year_numeric", "max", "construction_year_max")

    if {"floor_area_m2_sum", "n_buildings"}.issubset(features.columns):
        features["gross_floor_area_per_building"] = (
            features["floor_area_m2_sum"] / features["n_buildings"]
        )

    if {"footprint_area_m2_sum", "n_buildings"}.issubset(features.columns):
        features["avg_footprint_per_building"] = (
            features["footprint_area_m2_sum"] / features["n_buildings"]
        )

    if {"floor_area_m2_sum", "footprint_area_m2_sum"}.issubset(features.columns):
        denom = features["footprint_area_m2_sum"].replace({0: np.nan})
        features["floor_area_density"] = features["floor_area_m2_sum"] / denom

    return features


def prepare_dataset(buildings_path: Path) -> pd.DataFrame:
    buildings = pd.read_csv(buildings_path)

    required = {"id_agg", "wb_gs"}
    missing = required - set(buildings.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    buildings = buildings.dropna(subset=["id_agg", "wb_gs"]).copy()
    buildings["id_agg"] = buildings["id_agg"].astype(int)
    buildings["wb_gs"] = pd.to_numeric(buildings["wb_gs"], errors="coerce")
    buildings = buildings.dropna(subset=["wb_gs"])

    if buildings.empty:
        raise ValueError("No usable building rows remain after cleaning")

    if "tabula_era" in buildings.columns:
        buildings["tabula_era_num"] = buildings["tabula_era"].map(ERA_TO_NUM)

    if "tabula_type" in buildings.columns:
        buildings["tabula_type"] = buildings["tabula_type"].astype(str).str.upper().str.strip()

    if any(col in buildings.columns for col in ("construction_year", "year_built", "built_year")):
        source_col = next(
            col for col in ("construction_year", "year_built", "built_year") if col in buildings.columns
        )
        buildings["construction_year_numeric"] = _normalise_construction_year(buildings[source_col])
        buildings["construction_decade"] = buildings["construction_year_numeric"].apply(
            lambda val: int(val // 10 * 10) if pd.notna(val) else np.nan
        )
    else:
        buildings["construction_year_numeric"] = np.nan
        buildings["construction_decade"] = np.nan

    features = _summarise_buildings(buildings)

    if "tabula_type" in buildings.columns:
        features = features.join(
            _build_category_features(buildings, "tabula_type", "tabula_type"),
            how="left",
        )
    elif "osm_building" in buildings.columns:
        features = features.join(
            _build_category_features(buildings, "osm_building", "osm_building"),
            how="left",
        )

    if "tabula_era" in buildings.columns:
        features = features.join(
            _build_category_features(buildings, "tabula_era", "tabula_era"),
            how="left",
        )

    for age_col in ("building_age_group", "construction_period"):
        if age_col in buildings.columns:
            features = features.join(
                _build_category_features(buildings, age_col, age_col),
                how="left",
            )
            break

        if "construction_decade" in buildings.columns:
            features = features.join(
                _build_category_features(buildings, "construction_decade", "construction_decade"),
                how="left",
            )

    target = buildings.groupby("id_agg")["wb_gs"].mean().rename("wb_gs")
    dataset = features.join(target, how="inner")

    if dataset.empty:
        raise ValueError("No overlapping Energieatlas IDs; dataset is empty")

    dataset = _fill_numeric(dataset)
    return dataset.reset_index()


def _load_embeddings(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Embeddings file '{path}' has unexpected structure")

    per_model: dict[str, dict[int, Sequence[float]]] = {}
    for id_key, model_map in raw.items():
        try:
            id_agg = int(id_key)
        except (TypeError, ValueError):
            LOGGER.debug("Skipping non-integer id_agg key '%s'", id_key)
            continue

        if not isinstance(model_map, dict):
            LOGGER.debug("Skipping id_agg %s with non-dict embeddings", id_key)
            continue

        for model_name, embedding in model_map.items():
            if embedding is None:
                continue
            per_model.setdefault(model_name, {})[id_agg] = embedding

    frames: dict[str, pd.DataFrame] = {}
    for model_name, mapping in per_model.items():
        if not mapping:
            continue
        frame = pd.DataFrame.from_dict(mapping, orient="index")
        frame.index.name = "id_agg"
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame.sort_index(inplace=True)
        slug = _slugify(model_name)
        frame.columns = [f"{slug}_embedding_{idx}" for idx in range(frame.shape[1])]
        frames[model_name] = frame

    if not frames:
        raise ValueError(f"No usable embeddings found in '{path}'")

    LOGGER.info("Loaded embeddings for models: %s", ", ".join(sorted(frames)))
    return frames


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
    cv_splits: int = 5,
    pca_components: int | None = None,
    pca_variance: float | None = None,
    regressor: str = "ridge",
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, float, float]:
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
        description="Predict Energieatlas heat demand from building composition features",
    )
    parser.add_argument(
        "--buildings",
        type=Path,
        default=Path("buildings_output.csv"),
        help="Path to building-level CSV export",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=6,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("embeddings_model.json"),
        help="Path to embeddings JSON generated for each Energieatlas cell",
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

    LOGGER.info("Loading building data from \"%s\"", args.buildings)
    dataset = prepare_dataset(args.buildings)
    LOGGER.info(
        "Prepared dataset with %d Energieatlas cells and %d feature columns",
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
        LOGGER.info("Baseline fold %d R^2: %.3f", idx, score)
    LOGGER.info(
        "Baseline mean R^2 over %d folds: %.3f ± %.3f",
        len(scores),
        mean_score,
        std_score,
    )

    embeddings_by_model: dict[str, pd.DataFrame] = {}
    try:
        embeddings_by_model = _load_embeddings(args.embeddings)
    except FileNotFoundError as exc:
        LOGGER.warning("%s; skipping embedding-based regressions", exc)
    except ValueError as exc:
        LOGGER.warning("%s; skipping embedding-based regressions", exc)

    if not embeddings_by_model:
        return

    base_indexed = dataset.set_index("id_agg")
    summary_rows: list[tuple[str, int, int, float, float]] = []

    for model_name, embed_frame in sorted(embeddings_by_model.items()):
        LOGGER.info("Running regression with '%s' embeddings", model_name)
        combined = base_indexed.join(embed_frame, how="inner")
        if combined.empty:
            LOGGER.warning(
                "No overlapping records between dataset and '%s' embeddings; skipping",
                model_name,
            )
            continue

        combined = _fill_numeric(combined.reset_index())
        scores, mean_score, std_score = run_regression(
            combined,
            cv_splits=args.cv,
            pca_components=args.pca_components,
            pca_variance=args.pca_variance,
            regressor=args.regressor,
            ridge_alpha=args.ridge_alpha,
        )
        for idx, score in enumerate(scores, start=1):
            LOGGER.info("%s fold %d R^2: %.3f", model_name, idx, score)

        LOGGER.info(
            "%s mean R^2 over %d folds: %.3f ± %.3f (%d cells, %d features)",
            model_name,
            len(scores),
            mean_score,
            std_score,
            len(combined),
            combined.shape[1] - 2,
        )
        summary_rows.append(
            (
                model_name,
                len(combined),
                combined.shape[1] - 2,
                mean_score,
                std_score,
            )
        )

    if summary_rows:
        LOGGER.info("Embedding model performance summary:")
        for model_name, n_cells, n_features, mean_score, std_score in summary_rows:
            LOGGER.info(
                "  %s: mean R^2 %.3f ± %.3f (%d cells, %d features)",
                model_name,
                mean_score,
                std_score,
                n_cells,
                n_features,
            )


if __name__ == "__main__":
    main()