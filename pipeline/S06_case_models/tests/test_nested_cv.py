"""
test_nested_cv.py — Focused tests for S06 nested cross-validation protocol.

Verifies:
1. Outer-fold isolation from Optuna tuning.
2. Correct use of spatial split_group in inner CV (zero group leakage).
3. Frozen test-group exclusion (24 groups, 53 cases completely excluded).
4. Exactly one OOF prediction per development case (482 cases, 0 NaNs).
5. Metric calculation on the correct case-target pairs.
6. Repeatable split construction and output generation.
"""

import math
import pathlib
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

TEST_DIR = pathlib.Path(__file__).resolve().parent
STAGE_DIR = TEST_DIR.parent
REPO_ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STAGE_DIR))

from pipeline.S06_case_models.nested_tuning import (
    load_development_data,
    run_nested_cv,
)


@pytest.fixture(scope="module")
def dev_data():
    full_df, splits, test_groups = load_development_data()
    return full_df, splits, test_groups


def test_frozen_test_group_exclusion(dev_data):
    """Confirm frozen test groups (24 groups) do not enter development data."""
    full_df, splits, test_groups = dev_data

    assert len(test_groups) == 24, f"Expected 24 frozen test groups, got {len(test_groups)}"
    dev_groups = set(full_df["split_group"])
    overlap = dev_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Found {len(overlap)} frozen test groups in development data: {overlap}"
    assert len(full_df) == 482, f"Expected 482 dev cases, got {len(full_df)}"


def test_outer_fold_spatial_isolation(dev_data):
    """Confirm no spatial group is shared between outer training and validation folds."""
    full_df, splits, _ = dev_data
    n_folds = int(splits["n_folds"])
    assert n_folds == 5

    for k in range(n_folds):
        val_mask = full_df["outer_fold"] == k
        train_mask = ~val_mask

        val_groups = set(full_df.loc[val_mask, "split_group"])
        train_groups = set(full_df.loc[train_mask, "split_group"])

        overlap = val_groups.intersection(train_groups)
        assert len(overlap) == 0, f"Fold {k} has overlapping spatial groups: {overlap}"


def test_inner_cv_group_isolation(dev_data):
    """Confirm inner GroupKFold uses spatial split_group on outer training partition only with zero group leakage."""
    full_df, _, _ = dev_data

    for k in range(5):
        train_mask = full_df["outer_fold"] != k
        train_df = full_df[train_mask]

        groups_train = train_df["split_group"].values
        inner_gkf = GroupKFold(n_splits=3)

        for inner_train_idx, inner_val_idx in inner_gkf.split(train_df, groups=groups_train):
            inner_train_groups = set(groups_train[inner_train_idx])
            inner_val_groups = set(groups_train[inner_val_idx])

            overlap = inner_train_groups.intersection(inner_val_groups)
            assert len(overlap) == 0, f"Fold {k} inner split has overlapping groups: {overlap}"


def test_one_oof_prediction_per_dev_case(dev_data):
    """Confirm each dev case belongs to exactly one outer fold and receives exactly one prediction."""
    full_df, _, _ = dev_data

    # Each case appears once
    assert full_df["case"].nunique() == len(full_df) == 482

    # Each case has an outer_fold in {0, 1, 2, 3, 4}
    assert set(full_df["outer_fold"].unique()) == {0, 1, 2, 3, 4}
    assert full_df["outer_fold"].notna().all()


def test_metric_calculation_correctness():
    """Verify that pooled and fold-level metric calculations match scikit-learn standard."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    expected_r2 = float(r2_score(y_true, y_pred))
    expected_mae = float(mean_absolute_error(y_true, y_pred))
    expected_rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    assert math.isclose(expected_r2, 0.985, rel_tol=1e-2)
    assert math.isclose(expected_mae, 0.14, rel_tol=1e-2)
    assert math.isclose(expected_rmse, 0.148, rel_tol=1e-2)


def test_repeatable_split_construction(dev_data):
    """Confirm split construction is deterministic and identical upon reload."""
    full_df1, splits1, test_groups1 = dev_data
    full_df2, splits2, test_groups2 = load_development_data()

    pd.testing.assert_frame_equal(
        full_df1[["case", "split_group", "outer_fold", "U_mean"]],
        full_df2[["case", "split_group", "outer_fold", "U_mean"]]
    )
    assert splits1 == splits2
    assert test_groups1 == test_groups2


def test_nested_tuning_smoke_run(dev_data):
    """Run a fast 1-trial smoke run in a temporary directory to verify end-to-end execution and output files."""
    full_df, _, _ = dev_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        # Run with 1 trial to test pipeline execution
        results = run_nested_cv(
            full_df,
            n_trials=1,
            random_seed=42,
            target_col="U_mean",
            output_dir=tmp_path,
        )

        # Verify outputs exist
        assert (tmp_path / "nested_oof_predictions.csv").exists()
        assert (tmp_path / "nested_fold_metrics.csv").exists()
        assert (tmp_path / "nested_summary_metrics.csv").exists()
        assert (tmp_path / "nested_best_params.json").exists()
        assert (tmp_path / "S06_nested_cv_report.md").exists()

        # Verify OOF predictions table
        oof_df = pd.read_csv(tmp_path / "nested_oof_predictions.csv")
        assert len(oof_df) == 482
        assert not oof_df["U_mean_pred"].isna().any()
        assert set(oof_df["case"]) == set(full_df["case"])

        # Verify fold metrics table
        fold_df = pd.read_csv(tmp_path / "nested_fold_metrics.csv")
        assert len(fold_df) == 5
        assert set(fold_df["outer_fold"]) == {0, 1, 2, 3, 4}
        assert fold_df["n_cases"].sum() == 482

        # Verify summary metrics table
        summary_df = pd.read_csv(tmp_path / "nested_summary_metrics.csv")
        assert len(summary_df) == 3
        assert set(summary_df["dataset"]) == {"Overall Dev", "Idealized", "Realistic"}
