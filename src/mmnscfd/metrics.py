"""
metrics.py – evaluation metrics for the Urban Microclimate Surrogate.

All metrics operate on open-ground cells only (where the building mask is False).
Case-weighted aggregates weight each case equally regardless of domain size.

Public API
----------
case_metrics(y_true, y_pred, mask)   : dict of metrics for a single case
pooled_metrics(per_case_list)         : aggregate across a list of case dicts
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Single-case metrics
# ---------------------------------------------------------------------------

def case_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
    case: str = "",
) -> dict[str, Any]:
    """Compute metrics for one case on open-ground pixels.

    Parameters
    ----------
    y_true, y_pred : 2-D float arrays (ny × nx) or 1-D flat arrays of U/u*.
    mask           : bool array, True = building (excluded). None = use all.
    case           : case identifier for logging.

    Returns
    -------
    dict with keys: case, n, mae, rmse, r2, nrmse, within_r2, ssim,
                    delta_p95, delta_max, level_err, pattern_err.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if mask is not None:
        mask_flat = np.asarray(mask, dtype=bool).ravel()
        y_true = y_true[~mask_flat]
        y_pred = y_pred[~mask_flat]

    n = len(y_true)
    if n == 0:
        return _empty_metrics(case)

    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mean_u = float(y_true.mean())
    nrmse = rmse / mean_u if mean_u > 0 else float("nan")

    # Within-case R² is the same as case R² for flat arrays
    within_r2 = r2

    # SSIM (simplified, scalar version: pearson correlation of standardised arrays)
    std_true = y_true.std()
    std_pred = y_pred.std()
    if std_true > 0 and std_pred > 0:
        ssim = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        ssim = float("nan")

    # Peak-speed errors
    p95_true = float(np.percentile(y_true, 95))
    p95_pred = float(np.percentile(y_pred, 95))
    delta_p95 = float(p95_pred - p95_true)

    max_true = float(y_true.max())
    max_pred = float(y_pred.max())
    delta_max = float(max_pred - max_true)

    # Level / pattern decomposition  MSE = (mean error)² + pattern_error
    mean_err = float(residual.mean())
    level_err = float(mean_err ** 2)
    pattern_err = float(np.mean((residual - mean_err) ** 2))

    return {
        "case": case,
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nrmse": nrmse,
        "within_r2": within_r2,
        "ssim": ssim,
        "delta_p95": delta_p95,
        "delta_max": delta_max,
        "level_err": level_err,
        "pattern_err": pattern_err,
    }


def _empty_metrics(case: str) -> dict[str, Any]:
    nan = float("nan")
    return {
        "case": case, "n": 0, "mae": nan, "rmse": nan, "r2": nan,
        "nrmse": nan, "within_r2": nan, "ssim": nan,
        "delta_p95": nan, "delta_max": nan, "level_err": nan, "pattern_err": nan,
    }


# ---------------------------------------------------------------------------
# Pooled metrics
# ---------------------------------------------------------------------------

def pooled_metrics(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    """Case-weighted aggregate from a list of case_metrics dicts.

    Returns a dict with prefixed keys: pooled_mae, pooled_rmse, pooled_r2,
    median_nrmse, median_ssim, mean_delta_p95, mean_delta_max,
    mean_level_err, mean_pattern_err, n_cases.
    """
    valid = [d for d in per_case if d["n"] > 0]
    n_cases = len(valid)
    if n_cases == 0:
        return {"n_cases": 0}

    def _nanmean(key: str) -> float:
        vals = [d[key] for d in valid if not _is_nan(d[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _nanmedian(key: str) -> float:
        vals = [d[key] for d in valid if not _is_nan(d[key])]
        return float(np.median(vals)) if vals else float("nan")

    return {
        "n_cases": n_cases,
        "pooled_mae": _nanmean("mae"),
        "pooled_rmse": _nanmean("rmse"),
        "pooled_r2": _nanmean("r2"),
        "median_nrmse": _nanmedian("nrmse"),
        "median_ssim": _nanmedian("ssim"),
        "mean_delta_p95": _nanmean("delta_p95"),
        "mean_delta_max": _nanmean("delta_max"),
        "mean_level_err": _nanmean("level_err"),
        "mean_pattern_err": _nanmean("pattern_err"),
    }


def _is_nan(x: Any) -> bool:
    try:
        return bool(x != x)  # NaN != NaN
    except Exception:
        return False
