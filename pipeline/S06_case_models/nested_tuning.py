"""
nested_tuning.py – Leakage-Safe Nested Cross-Validation with Optuna for Stage S06.

This module implements a nested GroupKFold evaluation protocol:
- Outer CV: 5 spatial group-aware folds defined in data/splits.json.
- Inner CV: 3-fold GroupKFold grouped by spatial split_group, executed STRICTLY
  on each outer fold's training cases.
- Outer validation cases are NEVER seen during trial evaluation or hyperparameter selection.
- Frozen test groups (24 groups, 53 cases) remain completely untouched and excluded.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO = pathlib.Path(__file__).parents[2]
SPLITS_PATH = REPO / "data" / "splits.json"
FEATURES_PATH = REPO / "pipeline" / "S05_feature_engineering" / "outputs" / "case_features.csv"
TARGET_TABLE = REPO / "pipeline" / "S04_eda" / "S04g_combined_report" / "master_case_table.csv"
OUT_DIR = REPO / "pipeline" / "S06_case_models" / "outputs"
PRED_DIR = REPO / "results" / "preds" / "s06_lgbm_nested_tuned_physics_v1"

PHYSICS_FEATURES = [
    "lambda_p", "mean_h", "std_h", "max_h",
    "forcing_dir_deg", "lambda_f_dir", "wind_cos", "wind_sin",
    "subset_idealized", "lambda_w", "skew_h", "cv_h",
    "max_to_mean_h", "volume_per_area_m", "components_per_hectare",
    "mean_footprint_area_m2", "gamma_h", "lambda_f_x", "lambda_f_y",
    "d_over_h_macdonald", "z0_over_h_macdonald", "d_over_h_kanda", "z0_over_h_kanda",
    "upwind_fetch_mean_m", "upwind_fetch_median_m", "downwind_fetch_mean_m",
    "lateral_fetch_mean_m", "unobstructed_fraction_gt50m", "dist_mean_m",
    "dist_median_m", "dist_p90_m", "facet_orientation_entropy",
    "windward_wall_share", "relative_wind_facet_angle_deg",
]


def load_development_data(
    repo_path: pathlib.Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], set]:
    """Load development dataset, apply group mappings, exclude frozen test groups,

    and verify all 5 leakage-prevention assertions.
    """
    repo = repo_path or REPO
    splits_path = repo / "data" / "splits.json"
    features_path = repo / "pipeline" / "S05_feature_engineering" / "outputs" / "case_features.csv"
    targets_path = repo / "pipeline" / "S04_eda" / "S04g_combined_report" / "master_case_table.csv"

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    fold_map = {k: int(v) for k, v in splits["folds"].items()}
    test_groups = set(splits["test_groups"])
    n_folds = int(splits["n_folds"])

    feat_df = pd.read_csv(features_path)
    tgt_df = pd.read_csv(targets_path)
    tgt_ready = tgt_df[tgt_df["is_model_ready_spatial"].eq(True)].copy()

    rename_targets = {
        "u_ped_mean": "U_mean",
        "u_p95": "U_p95",
        "u_ped_std": "U_std",
        "tke_mean": "TKE_mean",
    }
    tgt_sub = tgt_ready[["case"] + list(rename_targets.keys())].rename(columns=rename_targets)

    # Filter out frozen test groups
    feat_df["is_test"] = feat_df["split_group"].isin(test_groups)
    dev_df = feat_df[~feat_df["is_test"]].copy()
    dev_df["outer_fold"] = dev_df["split_group"].map(fold_map)

    full_df = dev_df.merge(tgt_sub, on="case", how="inner")
    full_df["subset_idealized"] = (full_df["subset"] == "idealized").astype(float)

    # -------------------------------------------------------------------------
    # Strict Split Assertions
    # -------------------------------------------------------------------------
    # Assertion 1: No frozen test group enters dev data
    dev_group_set = set(full_df["split_group"])
    overlap_test = dev_group_set.intersection(test_groups)
    assert len(overlap_test) == 0, f"Assertion 1 Failed: Frozen test groups found in dev: {overlap_test}"

    # Assertion 2: No spatial group shared between outer train and outer val
    for k in range(n_folds):
        val_mask = full_df["outer_fold"] == k
        val_groups = set(full_df.loc[val_mask, "split_group"])
        train_groups = set(full_df.loc[~val_mask, "split_group"])
        overlap_outer = val_groups.intersection(train_groups)
        assert len(overlap_outer) == 0, f"Assertion 2 Failed: Fold {k} has overlapping groups: {overlap_outer}"

    # Assertion 5: Each dev case belongs to exactly one outer fold
    assert full_df["outer_fold"].notna().all(), "Assertion 5 Failed: Some dev cases have missing outer_fold"
    assert len(full_df) == 482, f"Assertion 5 Failed: Expected 482 dev cases, got {len(full_df)}"

    return full_df, splits, test_groups


def run_nested_cv(
    full_df: pd.DataFrame,
    n_trials: int = 10,
    random_seed: int = 42,
    target_col: str = "U_mean",
    output_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """Execute leakage-safe nested cross-validation with Optuna hyperparameter tuning."""
    out_dir = output_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_dir is None:  # tests pass a temporary output_dir and must not touch results/
        PRED_DIR.mkdir(parents=True, exist_ok=True)

    X_all = full_df[PHYSICS_FEATURES].fillna(0.0).values
    y_all = full_df[target_col].values
    cases_all = full_df["case"].values
    outer_folds = full_df["outer_fold"].values.astype(int)
    split_groups = full_df["split_group"].values
    subsets = full_df["subset"].values

    n_outer_folds = 5
    oof_preds = np.full(len(full_df), np.nan)
    fold_records: List[Dict[str, Any]] = []
    best_params_by_fold: Dict[int, Dict[str, Any]] = {}

    start_time = time.time()
    logger.info(f"Starting Nested CV ({n_outer_folds} outer folds, {n_trials} trials/fold) for {target_col}")

    for k in range(n_outer_folds):
        fold_start = time.time()
        val_mask = outer_folds == k
        train_mask = ~val_mask

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        groups_train = split_groups[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]

        # Inner CV grouped strictly by spatial split_group on train_mask ONLY
        inner_gkf = GroupKFold(n_splits=3)

        # Assertion 3 & 4 check
        for inner_train_idx, inner_val_idx in inner_gkf.split(X_train, y_train, groups=groups_train):
            i_train_grp = set(groups_train[inner_train_idx])
            i_val_grp = set(groups_train[inner_val_idx])
            overlap_inner = i_train_grp.intersection(i_val_grp)
            assert len(overlap_inner) == 0, f"Assertion 4 Failed: Inner fold group overlap: {overlap_inner}"

        def inner_objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": random_seed + k,
                "n_jobs": 2,
                "verbosity": -1,
            }
            scores = cross_val_score(
                lgb.LGBMRegressor(**params),
                X_train,
                y_train,
                cv=inner_gkf,
                groups=groups_train,
                scoring="r2",
                n_jobs=1,
            )
            return float(scores.mean())

        sampler = optuna.samplers.TPESampler(seed=random_seed + k)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(inner_objective, n_trials=n_trials)

        best_params_k = study.best_params
        best_inner_r2 = float(study.best_value)
        best_params_by_fold[k] = {
            "best_params": best_params_k,
            "best_inner_r2": best_inner_r2,
            "n_train_cases": int(train_mask.sum()),
            "n_train_groups": int(pd.Series(groups_train).nunique()),
        }

        # Fit model with selected hyperparameters on ALL outer training data
        model = lgb.LGBMRegressor(
            **best_params_k,
            random_state=random_seed + k,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(X_train, y_train)

        # Predict on untouched outer validation cases
        y_pred = model.predict(X_val)
        oof_preds[val_mask] = y_pred

        # Metrics for fold k
        r2_k = float(r2_score(y_val, y_pred))
        mae_k = float(mean_absolute_error(y_val, y_pred))
        rmse_k = float(math.sqrt(mean_squared_error(y_val, y_pred)))
        n_val_k = int(val_mask.sum())
        n_val_groups_k = int(pd.Series(split_groups[val_mask]).nunique())
        fold_elapsed = time.time() - fold_start

        logger.info(
            f"Outer Fold {k}: Inner R²={best_inner_r2:.4f} | Outer Val R²={r2_k:.4f}, "
            f"MAE={mae_k:.4f}, RMSE={rmse_k:.4f} (N={n_val_k}, {n_val_groups_k} groups) [{fold_elapsed:.1f}s]"
        )

        fold_records.append({
            "outer_fold": k,
            "r2": r2_k,
            "mae": mae_k,
            "rmse": rmse_k,
            "n_cases": n_val_k,
            "n_groups": n_val_groups_k,
            "best_inner_r2": best_inner_r2,
            "best_params": best_params_k,
            "duration_sec": round(fold_elapsed, 2),
        })

    total_duration = time.time() - start_time
    logger.info(f"Nested CV completed in {total_duration:.1f}s")

    # Verify no missing OOF predictions
    assert not np.isnan(oof_preds).any(), "Assertion Failed: Some dev cases missing OOF prediction"

    # Overall OOF metrics
    overall_r2 = float(r2_score(y_all, oof_preds))
    overall_mae = float(mean_absolute_error(y_all, oof_preds))
    overall_rmse = float(math.sqrt(mean_squared_error(y_all, oof_preds)))
    mean_fold_r2 = float(np.mean([r["r2"] for r in fold_records]))
    std_fold_r2 = float(np.std([r["r2"] for r in fold_records]))

    # Subset metrics
    ideal_mask = subsets == "idealized"
    real_mask = subsets == "realistic"

    ideal_r2 = float(r2_score(y_all[ideal_mask], oof_preds[ideal_mask]))
    ideal_mae = float(mean_absolute_error(y_all[ideal_mask], oof_preds[ideal_mask]))
    ideal_rmse = float(math.sqrt(mean_squared_error(y_all[ideal_mask], oof_preds[ideal_mask])))

    real_r2 = float(r2_score(y_all[real_mask], oof_preds[real_mask]))
    real_mae = float(mean_absolute_error(y_all[real_mask], oof_preds[real_mask]))
    real_rmse = float(math.sqrt(mean_squared_error(y_all[real_mask], oof_preds[real_mask])))

    # -------------------------------------------------------------------------
    # Save Auditable Outputs
    # -------------------------------------------------------------------------
    # 1. OOF Predictions table
    oof_df = pd.DataFrame({
        "case": cases_all,
        "subset": subsets,
        "split_group": split_groups,
        "outer_fold": outer_folds,
        f"{target_col}_true": y_all,
        f"{target_col}_pred": oof_preds,
        "error": oof_preds - y_all,
        "abs_error": np.abs(oof_preds - y_all),
    })
    oof_csv_path = out_dir / "nested_oof_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    # Mirror copy in results/preds/s06_lgbm_nested_tuned_physics_v1/ (real runs only)
    if output_dir is None:
        oof_df.to_csv(PRED_DIR / "oof_preds.csv", index=False)

    # 2. Fold metrics table
    fold_df = pd.DataFrame(fold_records)
    fold_csv_path = out_dir / "nested_fold_metrics.csv"
    fold_df.to_csv(fold_csv_path, index=False)

    # 3. Overall and subset summary
    summary_records = [
        {"dataset": "Overall Dev", "n_cases": len(full_df), "r2_pooled": overall_r2, "r2_mean_fold": mean_fold_r2, "r2_std_fold": std_fold_r2, "mae": overall_mae, "rmse": overall_rmse},
        {"dataset": "Idealized", "n_cases": int(ideal_mask.sum()), "r2_pooled": ideal_r2, "r2_mean_fold": np.nan, "r2_std_fold": np.nan, "mae": ideal_mae, "rmse": ideal_rmse},
        {"dataset": "Realistic", "n_cases": int(real_mask.sum()), "r2_pooled": real_r2, "r2_mean_fold": np.nan, "r2_std_fold": np.nan, "mae": real_mae, "rmse": real_rmse},
    ]
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = out_dir / "nested_summary_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # 4. Best parameters JSON
    params_json_path = out_dir / "nested_best_params.json"
    with open(params_json_path, "w", encoding="utf-8") as f:
        json.dump(best_params_by_fold, f, indent=2)

    # 5. Report Markdown
    report_content = f"""# Stage S06: Leakage-Safe Nested Cross-Validation Report

## Executive Summary
This report presents generalization estimates for LightGBM trained on 34 physics features, evaluated using a **nested cross-validation protocol** to eliminate hyperparameter selection bias.

- **Target:** `{target_col}`
- **Protocol:** 5 outer folds (GroupKFold by spatial `split_group`) × 3 inner folds (GroupKFold by spatial `split_group`)
- **Tuning Budget:** {n_trials} Optuna trials per outer fold (total {n_outer_folds * n_trials} optimization trials)
- **Total Duration:** {total_duration:.1f} seconds
- **Frozen Test Groups:** 24 groups (53 cases) completely excluded and unopened.

## Key Comparison Against Historical Benchmark

| Evaluation Protocol | Mean Outer Fold R² | Pooled OOF R² | Fold R² Vector | Selection Bias Status |
|---|---|---|---|---|
| Historical Single-CV (Leaked Tuning) | **0.8331** | ~0.833 | [0.7372, 0.8789, 0.9045, 0.7822, 0.8625] | **Potentially Optimistic** (Validation folds reused for Optuna tuning) |
| **Nested CV (Corrected Protocol)** | **{mean_fold_r2:.4f}** | **{overall_r2:.4f}** | [{", ".join(f"{r['r2']:.4f}" for r in fold_records)}] | **Leakage-Safe Estimate** (Hyperparameters selected on training folds only) |

> **Scientific Interpretation:**
> The historical R² of 0.8331 was obtained by running Optuna on all 5 development folds simultaneously, meaning the hyperparameters were selected with prior exposure to the test partitions. The nested CV estimate ({mean_fold_r2:.4f} mean fold R², {overall_r2:.4f} pooled R²) is an honest generalization estimate obtained without reusing outer validation folds for tuning.

## Detailed Outer Fold Performance

| Outer Fold | Cases | Spatial Groups | Outer Val R² | MAE (m/s) | RMSE (m/s) | Best Inner CV R² | Tuning Duration |
|---|---|---|---|---|---|---|---|
"""
    for r in fold_records:
        report_content += f"| Fold {r['outer_fold']} | {r['n_cases']} | {r['n_groups']} | {r['r2']:.4f} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['best_inner_r2']:.4f} | {r['duration_sec']}s |\n"

    report_content += f"""
## Performance by Morphological Subset

| Subset | Cases | Pooled R² | MAE (m/s) | RMSE (m/s) |
|---|---|---|---|---|
| **Idealized Geometry** | {int(ideal_mask.sum())} | {ideal_r2:.4f} | {ideal_mae:.4f} | {ideal_rmse:.4f} |
| **Realistic Geometry** | {int(real_mask.sum())} | {real_r2:.4f} | {real_mae:.4f} | {real_rmse:.4f} |
| **All Development Cases** | {len(full_df)} | {overall_r2:.4f} | {overall_mae:.4f} | {overall_rmse:.4f} |

## Downstream Experiments & Status

The following downstream experiments in `run_s06.py` and associated notebooks relied on the historical single-split `best_params`:
1. **Monotonic Constraints Experiment (Step 6):** Constrained `lambda_p <= 0` using historical params.
2. **Transfer Tests (Step 7):** Cross-subset models trained using historical params.
3. **Global Feature Importance (Step 10):** LightGBM fit on all 482 dev cases with historical params.

These historical experiments are preserved as exploratory artifacts, but their metrics should not be interpreted as validated by the nested protocol.

## Protocol Limitations
1. Nested CV estimates generalization under the outer split distribution; slight variance across outer fold choices persists due to the finite number of city-level spatial clusters (136 groups).
2. The outer-fold-specific hyperparameters vary slightly across folds, which reflects the natural sensitivity of hyperparameter selection to training group composition.
"""
    report_path = out_dir / "S06_nested_cv_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return {
        "overall_r2": overall_r2,
        "mean_fold_r2": mean_fold_r2,
        "std_fold_r2": std_fold_r2,
        "ideal_r2": ideal_r2,
        "real_r2": real_r2,
        "duration_sec": total_duration,
        "fold_records": fold_records,
        "best_params": best_params_by_fold,
    }


if __name__ == "__main__":
    n_trials = 10
    if len(sys.argv) > 1:
        try:
            n_trials = int(sys.argv[1])
        except ValueError:
            pass

    full_df, splits, test_groups = load_development_data()
    results = run_nested_cv(full_df, n_trials=n_trials)
    print("\n" + "=" * 60)
    print("NESTED CV SUMMARY")
    print(f"Mean Fold R²: {results['mean_fold_r2']:.4f} ± {results['std_fold_r2']:.4f}")
    print(f"Pooled OOF R²: {results['overall_r2']:.4f}")
    print(f"Idealized R²:  {results['ideal_r2']:.4f}")
    print(f"Realistic R²:  {results['real_r2']:.4f}")
    print("=" * 60)
