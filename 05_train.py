import os
import json
import numpy as np
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd

os.makedirs("models", exist_ok=True)
# Load embeddings and features
with open("data/embeddings/embeddings_model.json", "r") as f:
    emb_data = json.load(f)
with open("data/atlas_data/features_by_id.json", "r") as f:
    feat_data = json.load(f)

# Only use keys present in both
keys = [k for k in emb_data if k in feat_data]

# Prepare feature matrix and target
X = np.array([emb_data[k]["gpt4o"] + [feat_data[k].get("Shape_Area", 0), feat_data[k].get("Shape_Length", 0), feat_data[k].get("geb_n", 0)] for k in keys])
y = np.array([feat_data[k].get("wb_gs", 0) for k in keys])

def make_stratified_splits(y, n_splits=5, random_state=42):
    y = np.asarray(y)
    y_bins = pd.qcut(y, q=n_splits, labels=False, duplicates="drop")
    uniq = np.unique(y_bins)
    if len(uniq) < n_splits:
        n_splits = len(uniq)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros_like(y), y_bins)), n_splits

splits, n_splits = make_stratified_splits(y, n_splits=5, random_state=42)

# Convert splits to train/test indices for GridSearchCV
# GridSearchCV expects a cross-validation generator, so we create a list of (train, test) indices
from sklearn.model_selection import PredefinedSplit
test_fold = np.zeros(len(y), dtype=int) - 1
cv_indices = []
for i, (train_idx, test_idx) in enumerate(splits):
    fold = np.zeros(len(y), dtype=int) - 1
    fold[test_idx] = i
    cv_indices.append(fold)
# Use the first fold for GridSearchCV (or use KFold if you want all splits)
test_fold = cv_indices[0]
ps = PredefinedSplit(test_fold)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("gbm", HistGradientBoostingRegressor(random_state=42))
])
param_grid = {
    "gbm__max_iter": [100, 200, 300],
    "gbm__max_depth": [3, 5, 7],
    "gbm__learning_rate": [0.05, 0.1, 0.2],
    "gbm__l2_regularization": [0.0, 0.1, 1.0]
}

search = GridSearchCV(
    pipe,
    param_grid,
    cv=ps,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)
search.fit(X, y)

best_model = search.best_estimator_
best_params = search.best_params_
best_r2 = search.best_score_

import joblib
joblib.dump(best_model, "models/best_gpt4o_regressor.joblib")

params_out = {
    "best_params": best_params,
}
with open("params.yml", "w") as f:
    yaml.dump(params_out, f)

print(f"Best R2: {best_r2:.3f}")
print("Best parameters saved to params.yml")
print("Model saved to models/best_gpt4o_regressor.joblib")


