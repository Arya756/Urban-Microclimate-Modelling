"""
Unit tests for S06b: Feature Selection and PCA Experiments.
Stage: S06b

Tests verify:
  1. No data leakage: frozen test groups are excluded.
  2. Targets and metadata identifiers are not in feature matrix.
  3. Group awareness: no group overlap between train and validation folds.
  4. Feature selection methods fit strictly on training fold only.
  5. PCA StandardScaler and PCA transform are fit strictly on training fold only.
  6. S06 baseline reproduction: exact match with reported ablation table.
  7. Subset logic: subset_idealized is dropped in subset-specific runs.
"""
from __future__ import annotations

import json
import pathlib
import sys
import numpy as np
import pandas as pd

TEST_DIR = pathlib.Path(__file__).resolve().parent
STAGE_DIR = TEST_DIR.parent
REPO_ROOT = STAGE_DIR.parents[1]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STAGE_DIR))

from feature_selection_utils import (
    load_dataset,
    filter_correlation,
    fit_transform_pca,
    train_and_eval_lgbm,
)


def test_no_test_groups_leakage():
    """Verify that none of the 24 frozen test groups leak into the development set."""
    splits_path = REPO_ROOT / "data" / "splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_groups = set(splits["test_groups"])
    assert len(test_groups) == 24, "Expected exactly 24 frozen test groups"

    full_df, physics_features = load_dataset(REPO_ROOT)
    dev_groups = set(full_df["split_group"].unique())

    overlap = dev_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Critical leakage! Frozen test groups found in dev set: {overlap}"
    assert len(full_df) == 482, f"Expected 482 dev cases, got {len(full_df)}"


def test_target_and_metadata_exclusion():
    """Verify that targets, case IDs, and metadata are excluded from the feature matrix."""
    full_df, physics_features = load_dataset(REPO_ROOT)

    forbidden_cols = [
        "case", "split_group", "fold", "is_test", "subset",
        "u_ped_mean", "u_p95", "u_ped_std", "tke_mean",
        "U_mean", "U_p95", "U_std", "TKE_mean",
    ]

    for col in forbidden_cols:
        assert col not in physics_features, f"Forbidden column '{col}' leaked into feature matrix!"

    assert len(physics_features) == 34, f"Expected exactly 34 physics features, got {len(physics_features)}"


def test_group_awareness_across_folds():
    """Verify that outer folds have zero group overlap between train and val sets."""
    full_df, _ = load_dataset(REPO_ROOT)
    folds = full_df["fold"].values.astype(int)
    groups = full_df["split_group"].values

    for k in range(5):
        train_groups = set(groups[folds != k])
        val_groups = set(groups[folds == k])
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Fold {k} has group overlap between train and validation: {overlap}"


def test_correlation_filter_train_only():
    """Verify that correlation filtering is computed solely on training fold."""
    full_df, physics_features = load_dataset(REPO_ROOT)
    folds = full_df["fold"].values.astype(int)
    y = full_df["U_mean"].values
    X_df = full_df[physics_features].fillna(0)

    # Train on fold 0 train data
    tr_mask = folds != 0
    retained_fold0 = filter_correlation(X_df.iloc[tr_mask], y[tr_mask], threshold=0.95)

    assert isinstance(retained_fold0, list)
    assert len(retained_fold0) < len(physics_features), "Correlation filter should remove collinear features"
    assert len(retained_fold0) >= 20, "Should retain at least 20 informative features"

    # Verify no feature pair in retained has correlation > 0.95 on training data
    corr_vals = X_df.iloc[tr_mask][retained_fold0].corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 0.0)
    max_corr = float(np.nanmax(corr_vals))
    assert max_corr <= 0.9500000001, f"Found pair with correlation {max_corr} > 0.95 in retained features!"


def test_pca_train_fit_only():
    """Verify that PCA and StandardScaler fit on training data without leaking validation info."""
    full_df, physics_features = load_dataset(REPO_ROOT)
    folds = full_df["fold"].values.astype(int)
    X_df = full_df[physics_features].fillna(0)

    tr_mask = folds != 0
    val_mask = folds == 0

    X_tr_pca, X_val_pca, pca, scaler = fit_transform_pca(
        X_df.iloc[tr_mask], X_df.iloc[val_mask], variance_threshold=0.95, random_state=42
    )

    # Number of features should equal n_components
    assert X_tr_pca.shape[1] == pca.n_components_
    assert X_val_pca.shape[1] == pca.n_components_
    assert X_tr_pca.shape[0] == tr_mask.sum()
    assert X_val_pca.shape[0] == val_mask.sum()

    # Cumulative variance should be >= 95%
    cum_var = np.sum(pca.explained_variance_ratio_)
    assert cum_var >= 0.95, f"Expected cumulative variance >= 0.95, got {cum_var}"


def test_baseline_reproduction_exact_match():
    """Verify that our evaluation harness reproduces the S06 ablation baseline score.

    Reference values updated after the S05 corrections (topography orientation fix, valid
    target-cell mask, and Kanda aerodynamic parameter corrections) regenerated
    case_features.csv. Directional features (upwind fetch, lambda_f_dir) changed due to
    the corrected np.flipud(H) orientation, shifting the cross-validated baseline by ~0.0015 R².
    The updated reference is stable and reproducible from the corrected feature table.

    Pre-fix reference  (case_features.csv 2026-10-25): mean_r2 = 0.8217, std_r2 = 0.0552
    Corrected reference (2026-10-27): mean_r2 = 0.8170, std_r2 = 0.0627
    """
    full_df, physics_features = load_dataset(REPO_ROOT)
    folds = full_df["fold"].values.astype(int)
    y = full_df["U_mean"].values
    X = full_df[physics_features].fillna(0).values

    fold_r2s = []
    for k in range(5):
        tm = folds != k
        vm = folds == k
        _, metrics, _ = train_and_eval_lgbm(
            X[tm], y[tm], X[vm], y[vm],
            n_estimators=400, learning_rate=0.05, random_state=42
        )
        fold_r2s.append(metrics["r2"])

    mean_r2 = float(np.mean(fold_r2s))
    std_r2 = float(np.std(fold_r2s, ddof=0))

    # Reference: corrected targets (U/u_tau, raster mask) and features (orientation, mask, Kanda), 2026-10-27.
    # LightGBM results shift slightly between library versions, hence the 1e-3 tolerance.
    assert abs(mean_r2 - 0.816954) < 1e-3, f"Baseline mean R2 mismatch: {mean_r2} vs 0.8170"
    assert abs(std_r2 - 0.062683) < 1e-3, f"Baseline std R2 mismatch: {std_r2} vs 0.0627"


def test_subset_drops_constant_feature():
    """Verify that subset-specific evaluations drop subset_idealized."""
    full_df, physics_features = load_dataset(REPO_ROOT)
    subset_features = [f for f in physics_features if f != "subset_idealized"]

    assert len(subset_features) == 33
    assert "subset_idealized" not in subset_features

    # Check that in idealized subset, subset_idealized was indeed constant
    ideal_df = full_df[full_df["subset"] == "idealized"]
    assert ideal_df["subset_idealized"].nunique() == 1
    assert ideal_df["subset_idealized"].iloc[0] == 1.0

    real_df = full_df[full_df["subset"] == "realistic"]
    assert real_df["subset_idealized"].nunique() == 1
    assert real_df["subset_idealized"].iloc[0] == 0.0


def test_consolidation_idempotent_no_duplicates():
    """Verify that result consolidation is strictly idempotent and creates zero duplicate rows."""
    from plot_and_consolidate import consolidate_results
    out_dir = STAGE_DIR / "outputs"

    # Run consolidation first time
    consolidate_results(out_dir)

    fm1 = pd.read_csv(out_dir / "fold_metrics.csv")
    oof1 = pd.read_csv(out_dir / "oof_predictions.csv")
    sub1 = pd.read_csv(out_dir / "subset_metrics.csv")
    comp1 = pd.read_csv(out_dir / "comparison.csv")

    assert fm1.duplicated(subset=["method", "fold"]).sum() == 0, "Duplicate (method, fold) found in fold_metrics!"
    assert oof1.duplicated(subset=["case", "target", "method", "fold"]).sum() == 0, "Duplicate OOF predictions found!"
    assert sub1.duplicated(subset=["subset", "method"]).sum() == 0, "Duplicate subset entries found!"

    # Run consolidation second time
    consolidate_results(out_dir)

    fm2 = pd.read_csv(out_dir / "fold_metrics.csv")
    oof2 = pd.read_csv(out_dir / "oof_predictions.csv")
    sub2 = pd.read_csv(out_dir / "subset_metrics.csv")
    comp2 = pd.read_csv(out_dir / "comparison.csv")

    # Verify second run produced identical shape and content
    assert fm1.shape == fm2.shape, f"fold_metrics duplicated rows on repeated run: {fm1.shape} vs {fm2.shape}"
    assert oof1.shape == oof2.shape, f"oof_predictions duplicated rows on repeated run: {oof1.shape} vs {oof2.shape}"
    assert sub1.shape == sub2.shape, f"subset_metrics duplicated rows on repeated run: {sub1.shape} vs {sub2.shape}"
    assert comp1.shape == comp2.shape, f"comparison duplicated rows on repeated run: {comp1.shape} vs {comp2.shape}"
    pd.testing.assert_frame_equal(fm1, fm2)
    pd.testing.assert_frame_equal(comp1, comp2)
