"""
test_evaluation.py: Tests for the scalar and field evaluation logic.

Covers:
1. Multi-case scalar evaluation against scikit-learn standard metrics.
2. Target isolation: U_mean metrics do not mix with TKE_mean.
3. Correct actual/predicted pairing by case identifier.
4. Duplicate or missing case-target prediction detection.
5. Non-finite predictions and actual values (NaN / Inf).
6. Empty input and constant-target R² handling.
7. Idealized/realistic subset calculations matching pooled rows.
8. Spatial evaluation with unequal valid-cell counts (case-weighted vs cell-pooled).
9. Empty valid-cell masks without division-by-zero crashes.
10. Frozen-test group exclusion verification using data/splits.json.
11. Schema validation distinguishing scalar predictions from spatial-field data.
12. Regression preservation of existing valid spatial metric behavior.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from urbanwind.evaluate import (
    evaluate_scalar_preds,
    is_scalar_case_table,
)
from urbanwind.metrics import (
    case_metrics,
    pooled_metrics,
    regression_metrics,
)


def test_multi_case_scalar_evaluation_against_sklearn():
    """Test 1: Verify regression_metrics matches scikit-learn standard exactly."""
    y_true = np.array([0.25, 0.45, 0.60, 0.35, 0.80, 0.15])
    y_pred = np.array([0.27, 0.42, 0.58, 0.39, 0.75, 0.18])

    res = regression_metrics(y_true, y_pred, target="U_mean")

    exp_r2 = float(r2_score(y_true, y_pred))
    exp_mae = float(mean_absolute_error(y_true, y_pred))
    exp_rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

    assert math.isclose(res["r2"], exp_r2, rel_tol=1e-6)
    assert math.isclose(res["mae"], exp_mae, rel_tol=1e-6)
    assert math.isclose(res["rmse"], exp_rmse, rel_tol=1e-6)
    assert res["n"] == 6
    assert res["target"] == "U_mean"


def test_target_isolation_multitarget():
    """Test 2: Multi-target scalar tables evaluate each target strictly in isolation."""
    df = pd.DataFrame({
        "case": ["c1", "c2", "c3", "c1", "c2", "c3"],
        "target": ["U_mean", "U_mean", "U_mean", "TKE_mean", "TKE_mean", "TKE_mean"],
        "y_true": [0.30, 0.50, 0.70, 0.02, 0.05, 0.08],
        "y_pred": [0.32, 0.48, 0.68, 0.03, 0.04, 0.09],
    })

    res = evaluate_scalar_preds(df)
    assert "U_mean" in res["targets"]
    assert "TKE_mean" in res["targets"]

    u_res = res["targets"]["U_mean"]["overall"]
    t_res = res["targets"]["TKE_mean"]["overall"]

    # Verify U_mean matches separate evaluation of U_mean rows only
    u_exp = regression_metrics([0.30, 0.50, 0.70], [0.32, 0.48, 0.68], target="U_mean")
    assert math.isclose(u_res["r2"], u_exp["r2"], rel_tol=1e-6)
    assert math.isclose(u_res["mae"], u_exp["mae"], rel_tol=1e-6)
    assert math.isclose(u_res["rmse"], u_exp["rmse"], rel_tol=1e-6)

    # Verify TKE_mean matches separate evaluation of TKE_mean rows only
    t_exp = regression_metrics([0.02, 0.05, 0.08], [0.03, 0.04, 0.09], target="TKE_mean")
    assert math.isclose(t_res["r2"], t_exp["r2"], rel_tol=1e-6)
    assert math.isclose(t_res["mae"], t_exp["mae"], rel_tol=1e-6)


def test_case_identifier_pairing():
    """Test 3: Case-level pairing is robust to row permutation."""
    df_ordered = pd.DataFrame({
        "case": ["case_A", "case_B", "case_C"],
        "target": ["U_mean", "U_mean", "U_mean"],
        "y_true": [1.0, 2.0, 3.0],
        "y_pred": [1.1, 2.1, 2.9],
    })
    df_shuffled = df_ordered.sample(frac=1.0, random_state=42).reset_index(drop=True)

    res_ordered = evaluate_scalar_preds(df_ordered)
    res_shuffled = evaluate_scalar_preds(df_shuffled)

    assert math.isclose(res_ordered["targets"]["U_mean"]["overall"]["r2"],
                        res_shuffled["targets"]["U_mean"]["overall"]["r2"], rel_tol=1e-6)
    assert math.isclose(res_ordered["targets"]["U_mean"]["overall"]["mae"],
                        res_shuffled["targets"]["U_mean"]["overall"]["mae"], rel_tol=1e-6)


def test_duplicate_case_detection():
    """Test 4: Duplicate case predictions for a single target raise ValueError."""
    df_dup = pd.DataFrame({
        "case": ["case_A", "case_A", "case_B"],
        "target": ["U_mean", "U_mean", "U_mean"],
        "y_true": [1.0, 1.2, 2.0],
        "y_pred": [1.1, 1.1, 2.1],
    })
    # For scalar evaluation, duplicate case IDs within the same target must raise ValueError
    with pytest.raises(ValueError, match="Duplicate cases found"):
        evaluate_scalar_preds(df_dup)


def test_non_finite_values_raise_error():
    """Test 5: NaN or Inf predictions or targets raise ValueError."""
    with pytest.raises(ValueError, match="non-finite"):
        regression_metrics([1.0, np.nan, 3.0], [1.1, 2.0, 3.1])

    with pytest.raises(ValueError, match="non-finite"):
        regression_metrics([1.0, 2.0, 3.0], [1.1, np.inf, 3.1])


def test_empty_input_and_constant_target_r2_handling():
    """Test 6: Empty inputs return n=0 and NaN; constant target returns documented NaN."""
    # Empty inputs
    empty_res = regression_metrics([], [], target="empty")
    assert empty_res["n"] == 0
    assert math.isnan(empty_res["r2"])
    assert math.isnan(empty_res["mae"])

    # Constant ground truth: ss_tot == 0
    const_res = regression_metrics([5.0, 5.0, 5.0], [5.1, 4.9, 5.2], target="const")
    assert const_res["n"] == 3
    assert math.isnan(const_res["r2"])
    assert math.isclose(const_res["mae"], 0.133333333, rel_tol=1e-4)

    # Perfect prediction of constant: ss_tot == 0 and ss_res == 0
    perfect_const = regression_metrics([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], target="perfect_const")
    assert perfect_const["r2"] == 1.0


def test_idealized_realistic_subset_breakdown():
    """Test 7: Stratified metrics on subsets match independent subset calculations."""
    df = pd.DataFrame({
        "case": ["c1", "c2", "c3", "c4"],
        "subset": ["idealized", "idealized", "realistic", "realistic"],
        "target": ["U_mean", "U_mean", "U_mean", "U_mean"],
        "y_true": [0.2, 0.4, 0.6, 0.8],
        "y_pred": [0.22, 0.38, 0.65, 0.75],
    })

    res = evaluate_scalar_preds(df)
    subsets = res["targets"]["U_mean"]["subsets"]

    exp_ideal = regression_metrics([0.2, 0.4], [0.22, 0.38])
    exp_real = regression_metrics([0.6, 0.8], [0.65, 0.75])

    assert math.isclose(subsets["idealized"]["r2"], exp_ideal["r2"], rel_tol=1e-6)
    assert math.isclose(subsets["idealized"]["mae"], exp_ideal["mae"], rel_tol=1e-6)
    assert math.isclose(subsets["realistic"]["r2"], exp_real["r2"], rel_tol=1e-6)
    assert math.isclose(subsets["realistic"]["mae"], exp_real["mae"], rel_tol=1e-6)


def test_spatial_evaluation_unequal_valid_cells_case_weighted():
    """Test 8: Spatial evaluation aggregates cases with equal weight (case-weighted)."""
    # Case A: 10 valid cells, error = 1.0 everywhere -> MAE = 1.0
    case_a = case_metrics(y_true=np.zeros(10), y_pred=np.ones(10), case="A")
    # Case B: 100 valid cells, error = 2.0 everywhere -> MAE = 2.0
    case_b = case_metrics(y_true=np.zeros(100), y_pred=np.full(100, 2.0), case="B")

    pooled = pooled_metrics([case_a, case_b])
    # Case-weighted mean of MAE = (1.0 + 2.0) / 2 = 1.5
    # Cell-weighted mean would be (10*1.0 + 100*2.0) / 110 = 1.909
    assert math.isclose(pooled["pooled_mae"], 1.5, rel_tol=1e-6)
    assert pooled["n_cases"] == 2


def test_empty_valid_cell_mask_no_crash():
    """Test 9: Case with mask=True everywhere returns n=0 without zero-division error."""
    all_building_mask = np.ones((5, 5), dtype=bool)
    y_true = np.ones((5, 5))
    y_pred = np.ones((5, 5))

    m = case_metrics(y_true, y_pred, mask=all_building_mask, case="empty_case")
    assert m["n"] == 0
    assert math.isnan(m["r2"])
    assert math.isnan(m["mae"])
    assert math.isnan(m["rmse"])


def test_frozen_test_group_exclusion_check(tmp_path):
    """Test 10: evaluate_scalar_preds asserts that no frozen test groups are present."""
    splits_file = tmp_path / "splits.json"
    splits_data = {
        "test_groups": ["group_test_1", "group_test_2"],
        "folds": {"group_dev_1": 0, "group_dev_2": 1},
        "n_folds": 2,
    }
    with open(splits_file, "w") as f:
        json.dump(splits_data, f)

    # Clean dev data
    clean_df = pd.DataFrame({
        "case": ["c1", "c2"],
        "split_group": ["group_dev_1", "group_dev_2"],
        "target": ["U_mean", "U_mean"],
        "y_true": [1.0, 2.0],
        "y_pred": [1.1, 1.9],
    })
    res = evaluate_scalar_preds(clean_df, splits_path=splits_file)
    assert res["test_exclusion_verified"] is True

    # Leaked test data
    leaked_df = pd.DataFrame({
        "case": ["c1", "c_test"],
        "split_group": ["group_dev_1", "group_test_1"],
        "target": ["U_mean", "U_mean"],
        "y_true": [1.0, 2.0],
        "y_pred": [1.1, 1.9],
    })
    with pytest.raises(AssertionError, match="Frozen test groups detected"):
        evaluate_scalar_preds(leaked_df, splits_path=splits_file)


def test_schema_validation_distinguishes_scalar_from_spatial():
    """Test 11: is_scalar_case_table correctly identifies scalar vs spatial tables."""
    # Scalar case table (1 row per case per target)
    scalar_df = pd.DataFrame({
        "case": ["c1", "c2", "c3"],
        "target": ["U_mean", "U_mean", "U_mean"],
        "y_true": [1.0, 2.0, 3.0],
        "y_pred": [1.1, 2.1, 3.1],
    })
    assert is_scalar_case_table(scalar_df) is True

    # Wide scalar table
    wide_df = pd.DataFrame({
        "case": ["c1", "c2"],
        "U_mean_true": [1.0, 2.0],
        "U_mean_pred": [1.1, 1.9],
    })
    assert is_scalar_case_table(wide_df) is True

    # Spatial field table (multiple rows per case for the same target)
    spatial_df = pd.DataFrame({
        "case": ["c1", "c1", "c2", "c2"],
        "target": ["U_mean", "U_mean", "U_mean", "U_mean"],
        "y_true": [1.0, 1.1, 2.0, 2.1],
        "y_pred": [1.05, 1.15, 2.05, 2.15],
    })
    assert is_scalar_case_table(spatial_df) is False


def test_regression_preserves_spatial_field_behavior():
    """Test 12: Existing spatial field metrics behavior is preserved exactly."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [3.2, 3.8]])
    mask = np.array([[False, False], [False, True]])  # 4th cell excluded

    m = case_metrics(y_true, y_pred, mask=mask, case="case_grid")
    assert m["n"] == 3
    assert math.isclose(m["mae"], 0.133333333, rel_tol=1e-4)
    # Check SSIM calculation on spatial field
    assert math.isfinite(m["corr"])
    assert math.isfinite(m["within_r2"])
