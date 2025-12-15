import os
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_rel

# Load embeddings and features
with open("data/embeddings/embeddings_model.json", "r") as f:
    emb_data = json.load(f)
with open("data/atlas_data/features_by_id.json", "r") as f:
    feat_data = json.load(f)

# Only use keys present in both
keys = [k for k in emb_data if k in feat_data]

# Prepare feature matrices
input_X = np.array([
    [feat_data[k].get("Shape_Area", 0), feat_data[k].get("Shape_Length", 0), feat_data[k].get("geb_n", 0)]
    for k in keys
])
clip_X = np.array([emb_data[k]["clip"] for k in keys])
gpt4o_X = np.array([emb_data[k]["gpt4o"] for k in keys])
qwen_X = np.array([emb_data[k]["qwen2.5"] for k in keys])

# Target variable
y = np.array([feat_data[k].get("wb_gs", 0) for k in keys])

def make_stratified_splits(y, n_splits=5, random_state=42):
    y = np.asarray(y)
    y_bins = pd.qcut(y, q=n_splits, labels=False, duplicates="drop")
    uniq = np.unique(y_bins)
    if len(uniq) < n_splits:
        n_splits = len(uniq)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros_like(y), y_bins)), n_splits

def evaluate_feature_set(name, model, X, y, splits, baseline_abs_err=None, log_target=False):
    X = np.asarray(X); y = np.asarray(y)
    r2_list, rmse_list, mae_list = [], [], []
    full_abs_err = np.empty_like(y, dtype=float)
    for fold, (tr_idx, val_idx) in enumerate(splits, 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        y_tr_fit = np.log1p(y_tr) if log_target else y_tr
        mdl = clone(model).fit(X_tr, y_tr_fit)
        y_pred = mdl.predict(X_val)
        if log_target:
            y_pred = np.expm1(y_pred)
        r2_list.append(r2_score(y_val, y_pred))
        rmse_list.append(root_mean_squared_error(y_val, y_pred))
        mae_list.append(mean_absolute_error(y_val, y_pred))
        full_abs_err[val_idx] = np.abs(y_val - y_pred)
        print(f"[{name}] Fold {fold}: R²={r2_list[-1]:.3f}  RMSE={rmse_list[-1]:.1f}  MAE={mae_list[-1]:.1f}")
    n_splits = len(splits)
    print(f"\n[{name}] === Stratified {n_splits}-Fold (mean ± std) ===")
    print(f"R²   {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")
    print(f"RMSE {np.mean(rmse_list):.1f} ± {np.std(rmse_list):.1f}")
    print(f"MAE  {np.mean(mae_list):.1f} ± {np.std(mae_list):.1f}")
    t_stat = p_val = None
    if baseline_abs_err is not None:
        try:
            t_stat, p_val = ttest_rel(baseline_abs_err, full_abs_err, alternative='greater')
        except TypeError:
            t_stat_two, p_two = ttest_rel(baseline_abs_err, full_abs_err)
            t_stat = t_stat_two
            p_val = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        sig = "significant" if p_val < 0.05 else "not significant"
        print(f"[{name}] Paired t-test vs baseline abs-error: t = {t_stat:.2f}, p = {p_val:.4f} → {sig}")
    return {
        "name": name,
        "R2_mean": np.mean(r2_list), "R2_std": np.std(r2_list),
        "RMSE_mean": np.mean(rmse_list), "RMSE_std": np.std(rmse_list),
        "MAE_mean": np.mean(mae_list), "MAE_std": np.std(mae_list),
        "abs_err": full_abs_err,
        "t": t_stat, "p": p_val,
    }

reg_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("gbm",   HistGradientBoostingRegressor())
])

splits, n_splits = make_stratified_splits(y, n_splits=5, random_state=42)

results = []
baseline_abs_err = None

for name, X in [("Baseline", input_X), ("CLIP", clip_X), ("GPT-4o", gpt4o_X), ("Qwen2.5", qwen_X)]:
    print(f"\n### {name} embeddings ###")
    summary = evaluate_feature_set(
        name=name,
        model=reg_pipe,
        X=X,
        y=y,
        splits=splits,
        baseline_abs_err=baseline_abs_err
    )
    results.append(summary)
    if name == "Baseline":
        baseline_abs_err = summary["abs_err"]

summary_df = pd.DataFrame([
    {
        "Model": r["name"],
        "R2 (mean±std)": f"{r['R2_mean']:.3f} ± {r['R2_std']:.3f}",
        "RMSE (mean±std)": f"{r['RMSE_mean']:.1f} ± {r['RMSE_std']:.1f}",
        "MAE (mean±std)": f"{r['MAE_mean']:.1f} ± {r['MAE_std']:.1f}",
        "t (vs baseline)": (f"{r['t']:.2f}" if r["t"] is not None else "-"),
        "p (vs baseline)": (f"{r['p']:.4f}" if r["p"] is not None else "-"),
    }
    for r in results
])
print("\n=== Summary ===")
print(summary_df.to_string(index=False))