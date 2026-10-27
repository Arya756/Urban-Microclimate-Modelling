"""
Feature Selection and PCA Utilities for S06b.
Stage: S06b

Provides leakage-free utilities for:
  - Dataset construction and validation (excluding frozen test groups)
  - Correlation filtering (deterministic, training-data only)
  - Mutual Information filtering (SelectKBest / mutual_info_regression)
  - F-test filtering (SelectKBest / f_regression)
  - Wrapper feature selection (RFECV with group-aware GroupKFold inner CV)
  - Principal Component Analysis (PCA with StandardScaler, training-fitted)
  - Evaluation harness with LightGBM (400 trees, lr=0.05 baseline)
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, mutual_info_regression, f_regression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature definitions (exact match to S06)
# ---------------------------------------------------------------------------
BASIC_FEATURES = [
    "lambda_p", "mean_h", "std_h", "max_h",
    "forcing_dir_deg", "lambda_f_dir", "wind_cos", "wind_sin",
    "subset_idealized",
]

MORPH_FEATURES = BASIC_FEATURES + [
    "lambda_w", "skew_h", "cv_h", "max_to_mean_h",
    "volume_per_area_m", "components_per_hectare",
    "mean_footprint_area_m2", "gamma_h",
]

PHYSICS_FEATURES = MORPH_FEATURES + [
    "lambda_f_x", "lambda_f_y",
    "d_over_h_macdonald", "z0_over_h_macdonald",
    "d_over_h_kanda", "z0_over_h_kanda",
    "upwind_fetch_mean_m", "upwind_fetch_median_m",
    "downwind_fetch_mean_m", "lateral_fetch_mean_m",
    "unobstructed_fraction_gt50m",
    "dist_mean_m", "dist_median_m", "dist_p90_m",
    "facet_orientation_entropy", "windward_wall_share",
    "relative_wind_facet_angle_deg",
]

TARGETS = ["U_mean", "U_p95", "U_std", "TKE_mean"]


def load_dataset(repo_path: pathlib.Path | str | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads and merges S05 features with the S02 case targets (U/u_tau over open
    ground, see urbanwind.data), enforces split assignments, and excludes frozen
    test groups.

    Returns:
        full_df: DataFrame with 482 dev cases, split_group, fold, targets, and features.
        physics_features: Validated list of 34 physics feature column names.
    """
    if repo_path is None:
        repo_path = pathlib.Path(__file__).resolve().parents[2]
    else:
        repo_path = pathlib.Path(repo_path)

    features_path = repo_path / "pipeline" / "S05_feature_engineering" / "outputs" / "case_features.csv"
    target_table = repo_path / "pipeline" / "S02_preprocessing" / "outputs" / "case_targets.csv"
    splits_path = repo_path / "data" / "splits.json"

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    fold_map = splits["folds"]
    test_groups = set(splits["test_groups"])

    feat_df = pd.read_csv(features_path)
    tgt_df = pd.read_csv(target_table)
    tgt_df = tgt_df[tgt_df["status"] != "exclude"].copy()

    feat_df["fold"] = feat_df["split_group"].map({k: int(v) for k, v in fold_map.items()})
    feat_df["is_test"] = feat_df["split_group"].isin(test_groups)

    # Strictly exclude frozen test groups
    dev_df = feat_df[~feat_df["is_test"]].copy()

    full_df = dev_df.merge(
        tgt_df[["case"] + TARGETS],
        on="case",
        how="inner",
    )
    full_df["subset_idealized"] = (full_df["subset"] == "idealized").astype(float)

    # Validate feature list
    seen = set()
    physics_features = []
    for f_col in PHYSICS_FEATURES:
        if f_col not in seen and f_col in full_df.columns:
            seen.add(f_col)
            physics_features.append(f_col)

    assert len(physics_features) == 34, f"Expected 34 physics features, got {len(physics_features)}"
    assert len(full_df) == 482, f"Expected 482 development cases, got {len(full_df)}"
    assert (full_df["subset"] == "idealized").sum() == 190, "Expected 190 idealized cases"
    assert (full_df["subset"] == "realistic").sum() == 292, "Expected 292 realistic cases"
    assert not full_df["split_group"].isin(test_groups).any(), "Found frozen test groups in dev_df!"

    return full_df, physics_features


# ---------------------------------------------------------------------------
# Filter methods (Strictly fit on training fold)
# ---------------------------------------------------------------------------
def filter_correlation(
    X_tr_df: pd.DataFrame,
    y_tr_arr: np.ndarray,
    threshold: float = 0.95,
) -> List[str]:
    """
    Deterministic correlation filtering on training data only.

    Rule:
      1. Rank features by their absolute Pearson correlation with y_train (descending).
      2. Iterate in ranked order. Retain the feature, and drop all subsequent unselected
         features that correlate with it above the threshold (|r| > threshold).
      3. Never touches validation or test data.

    Returns:
        List of retained feature names.
    """
    corr_with_target = {}
    for col in X_tr_df.columns:
        x_col = X_tr_df[col].values
        # If constant in train fold, correlation is 0
        if np.std(x_col) == 0:
            corr_with_target[col] = 0.0
        else:
            r = np.corrcoef(x_col, y_tr_arr)[0, 1]
            corr_with_target[col] = abs(r) if not np.isnan(r) else 0.0

    sorted_features = sorted(X_tr_df.columns, key=lambda c: corr_with_target[c], reverse=True)
    corr_mat = X_tr_df.corr().abs()

    retained = []
    dropped = set()

    for feat in sorted_features:
        if feat in dropped:
            continue
        retained.append(feat)
        for other in sorted_features:
            if other != feat and other not in dropped:
                pair_corr = corr_mat.loc[feat, other]
                if not np.isnan(pair_corr) and pair_corr > threshold:
                    dropped.add(other)

    return sorted(retained)


def filter_mutual_info(
    X_tr_df: pd.DataFrame,
    y_tr_arr: np.ndarray,
    k: int = 15,
    random_state: int = 42,
) -> List[str]:
    """
    Mutual Information filter on training fold only.
    Selects top-k features by mutual_info_regression with fixed random_state.
    """
    features = list(X_tr_df.columns)
    mi_scores = mutual_info_regression(X_tr_df.values, y_tr_arr, random_state=random_state)
    top_indices = np.argsort(mi_scores)[-k:]
    selected = [features[i] for i in top_indices]
    return sorted(selected)


def filter_f_test(
    X_tr_df: pd.DataFrame,
    y_tr_arr: np.ndarray,
    k: int = 15,
) -> List[str]:
    """
    F-test (f_regression) filter on training fold only.
    Selects top-k features by F-statistic.
    """
    features = list(X_tr_df.columns)
    # Check for constant columns (variance == 0) to avoid division by zero in f_regression
    var = np.var(X_tr_df.values, axis=0)
    scores = np.zeros(len(features))
    valid_cols = var > 1e-12
    if np.any(valid_cols):
        f_vals, _ = f_regression(X_tr_df.values[:, valid_cols], y_tr_arr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        scores[valid_cols] = f_vals

    top_indices = np.argsort(scores)[-k:]
    selected = [features[i] for i in top_indices]
    return sorted(selected)


# ---------------------------------------------------------------------------
# Wrapper method (RFECV with GroupKFold on outer training set)
# ---------------------------------------------------------------------------
def run_rfecv(
    X_tr_df: pd.DataFrame,
    y_tr_arr: np.ndarray,
    groups_tr: np.ndarray,
    n_inner_splits: int = 3,
    min_features: int = 5,
    step: int = 1,
    random_state: int = 42,
) -> Tuple[List[str], float, RFECV]:
    """
    Group-aware RFECV wrapper feature selection.
    
    Two-stage architecture:
      - Feature selection search: Uses LightGBM (100 trees, lr=0.05) and GroupKFold on outer training groups only.
      - Final evaluation: Evaluated separately using the standardized 400-tree baseline LightGBM model on X_val.

    Returns:
        selected_features: List of feature names selected by RFECV.
        elapsed_sec: Time in seconds taken by RFECV.
        selector: Fitted RFECV instance.
    """
    features = list(X_tr_df.columns)
    est = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    cv = GroupKFold(n_splits=n_inner_splits)
    rfecv = RFECV(
        estimator=est,
        step=step,
        cv=cv,
        scoring="r2",
        min_features_to_select=min_features,
        n_jobs=1,
    )

    t0 = time.time()
    rfecv.fit(X_tr_df.values, y_tr_arr, groups=groups_tr)
    elapsed = time.time() - t0

    selected = [features[i] for i, supp in enumerate(rfecv.support_) if supp]
    return sorted(selected), elapsed, rfecv


# ---------------------------------------------------------------------------
# PCA method (StandardScaler + PCA fit on training fold only)
# ---------------------------------------------------------------------------
def fit_transform_pca(
    X_tr_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    variance_threshold: float,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, PCA, StandardScaler]:
    """
    Fits StandardScaler and PCA on training data only, transforms both train and val.

    Returns:
        X_tr_pca: Scaled and PCA-projected training array.
        X_val_pca: Scaled and PCA-projected validation array.
        pca: Fitted PCA object.
        scaler: Fitted StandardScaler object.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_df.values)
    X_val_scaled = scaler.transform(X_val_df.values)

    pca = PCA(n_components=variance_threshold, random_state=random_state)
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    return X_tr_pca, X_val_pca, pca, scaler


# ---------------------------------------------------------------------------
# Baseline LightGBM model trainer & evaluator
# ---------------------------------------------------------------------------
def train_and_eval_lgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Trains standard S06 ablation baseline LightGBM model (400 trees, lr=0.05, default num_leaves=31)
    and evaluates on validation fold.

    Returns:
        preds: Array of predictions on X_val.
        metrics: Dictionary with R2, MAE, RMSE.
        train_time: Elapsed seconds for training and prediction.
    """
    m = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    t0 = time.time()
    m.fit(X_tr, y_tr)
    preds = m.predict(X_val)
    elapsed = time.time() - t0

    r2 = float(r2_score(y_val, preds))
    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(root_mean_squared_error(y_val, preds))

    return preds, {"r2": r2, "mae": mae, "rmse": rmse}, elapsed


def compute_stability(
    selected_features_per_fold: Dict[int, List[str]],
    all_features: List[str],
) -> pd.DataFrame:
    """
    Calculates feature selection stability across folds.
    Returns DataFrame sorted by selection count and frequency.
    """
    n_folds = len(selected_features_per_fold)
    counts = {feat: 0 for feat in all_features}
    for feats in selected_features_per_fold.values():
        for feat in feats:
            counts[feat] = counts.get(feat, 0) + 1

    rows = []
    for feat, count in counts.items():
        rows.append({
            "feature": feat,
            "selection_count": count,
            "selection_frequency": count / n_folds,
            "stable_across_all_folds": count == n_folds,
        })
    df = pd.DataFrame(rows).sort_values(by=["selection_count", "feature"], ascending=[False, True])
    return df
