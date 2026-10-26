"""
S06b: Feature Selection Experiments (Filter & Wrapper)
Stage: S06b

Implements and evaluates:
  1. Baseline (34 Physics features, LightGBM 400 trees, lr=0.05)
  2. Correlation Filtering (threshold 0.95, deterministic rule)
  3. Mutual Information (top 10, 15, 20)
  4. F-test (top 10, 15, 20)
  5. Wrapper: RFECV with inner GroupKFold (100-tree search estimator, 400-tree evaluation model)
  6. Idealized vs. Realistic subset-specific runs (dropping subset_idealized)
  7. Paired fold comparisons and out-of-fold predictions.

Usage:
  python3 pipeline/S06b_feature_selection_AS/run_feature_selection.py
  python3 pipeline/S06b_feature_selection_AS/run_feature_selection.py --smoke
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Setup imports
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from feature_selection_utils import (
    load_dataset,
    filter_correlation,
    filter_mutual_info,
    filter_f_test,
    run_rfecv,
    train_and_eval_lgbm,
    compute_stability,
)


def run_experiments(
    smoke: bool = False,
    target: str = "U_mean",
    output_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = CURRENT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"S06b: Feature Selection Experiments (Target: {target})")
    print(f"Mode: {'SMOKE TEST' if smoke else 'FULL EXPERIMENT'}")
    print("=" * 70)

    full_df, physics_features = load_dataset(REPO_ROOT)
    y_all = full_df[target].values
    folds = full_df["fold"].values.astype(int)
    groups = full_df["split_group"].values
    n_folds = 2 if smoke else 5

    # Storage for outputs
    fold_metrics_records = []
    selected_features_records = []
    selected_per_method_and_fold: Dict[str, Dict[int, List[str]]] = {}
    oof_predictions_records = []

    # Prepare methods to run
    methods_config = [
        ("Baseline (34 features)", "baseline", {}),
        ("Corr_filter_0.95", "corr_filter", {"threshold": 0.95}),
        ("Mutual_Info_top10", "mi", {"k": 10}),
        ("Mutual_Info_top15", "mi", {"k": 15}),
        ("Mutual_Info_top20", "mi", {"k": 20}),
        ("F_test_top10", "f_test", {"k": 10}),
        ("F_test_top15", "f_test", {"k": 15}),
        ("F_test_top20", "f_test", {"k": 20}),
        ("Wrapper_RFECV", "rfecv", {"min_features": 5, "step": 1 if not smoke else 4}),
    ]

    for m_name, _, _ in methods_config:
        selected_per_method_and_fold[m_name] = {}

    print(f"\nRunning {len(methods_config)} methods across {n_folds} outer folds...")

    for fold_idx in range(n_folds):
        print(f"\n--- Outer Fold {fold_idx + 1}/{n_folds} ---")
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_tr_df = full_df.loc[train_mask, physics_features].fillna(0)
        X_val_df = full_df.loc[val_mask, physics_features].fillna(0)
        y_tr = y_all[train_mask]
        y_val = y_all[val_mask]
        groups_tr = groups[train_mask]
        val_cases = full_df.loc[val_mask, "case"].values

        for method_label, method_type, params in methods_config:
            t0_sel = time.time()
            if method_type == "baseline":
                selected_cols = list(physics_features)
                sel_time = 0.0
            elif method_type == "corr_filter":
                selected_cols = filter_correlation(X_tr_df, y_tr, threshold=params["threshold"])
                sel_time = time.time() - t0_sel
            elif method_type == "mi":
                selected_cols = filter_mutual_info(X_tr_df, y_tr, k=params["k"], random_state=42)
                sel_time = time.time() - t0_sel
            elif method_type == "f_test":
                selected_cols = filter_f_test(X_tr_df, y_tr, k=params["k"])
                sel_time = time.time() - t0_sel
            elif method_type == "rfecv":
                inner_splits = 2 if smoke else 3
                selected_cols, sel_time, _ = run_rfecv(
                    X_tr_df,
                    y_tr,
                    groups_tr,
                    n_inner_splits=inner_splits,
                    min_features=params["min_features"],
                    step=params["step"],
                    random_state=42,
                )
            else:
                raise ValueError(f"Unknown method type {method_type}")

            selected_per_method_and_fold[method_label][fold_idx] = selected_cols

            # Record selected features
            selected_features_records.append({
                "method": method_label,
                "fold": fold_idx,
                "n_features": len(selected_cols),
                "selection_time_sec": round(sel_time, 4),
                "selected_features": ";".join(selected_cols),
            })

            # Train and evaluate model on validation fold
            X_tr_sub = X_tr_df[selected_cols].values
            X_val_sub = X_val_df[selected_cols].values

            preds, metrics, train_time = train_and_eval_lgbm(
                X_tr_sub, y_tr, X_val_sub, y_val,
                n_estimators=400, learning_rate=0.05, random_state=42
            )

            fold_metrics_records.append({
                "method": method_label,
                "fold": fold_idx,
                "n_features": len(selected_cols),
                "selection_time_sec": round(sel_time, 4),
                "train_time_sec": round(train_time, 4),
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            })

            for case_id, actual, pred in zip(val_cases, y_val, preds):
                oof_predictions_records.append({
                    "case": case_id,
                    "target": target,
                    "method": method_label,
                    "fold": fold_idx,
                    "actual": actual,
                    "predicted": pred,
                    "error": pred - actual,
                    "abs_error": abs(pred - actual),
                })

            print(f"  {method_label:22s} | k={len(selected_cols):2d} | R2={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | Sel={sel_time:5.2f}s | Train={train_time:5.2f}s")

    # -----------------------------------------------------------------------
    # Compile fold metrics & comparison tables
    # -----------------------------------------------------------------------
    fold_metrics_df = pd.DataFrame(fold_metrics_records)
    selected_features_df = pd.DataFrame(selected_features_records)
    oof_predictions_df = pd.DataFrame(oof_predictions_records)

    # Calculate comparison table across folds (np.std with ddof=0, as in S06)
    comparison_rows = []

    for method_label, _, _ in methods_config:
        m_df = fold_metrics_df[fold_metrics_df["method"] == method_label]
        mean_r2 = float(np.mean(m_df["r2"]))
        std_r2 = float(np.std(m_df["r2"], ddof=0))
        mean_mae = float(np.mean(m_df["mae"]))
        std_mae = float(np.std(m_df["mae"], ddof=0))
        mean_rmse = float(np.mean(m_df["rmse"]))
        std_rmse = float(np.std(m_df["rmse"], ddof=0))
        mean_features = float(np.mean(m_df["n_features"]))
        total_sel_time = float(np.sum(m_df["selection_time_sec"]))
        mean_train_time = float(np.mean(m_df["train_time_sec"]))

        # Pooled OOF R2 across all out-of-fold predictions
        m_oof = oof_predictions_df[oof_predictions_df["method"] == method_label]
        pooled_r2 = float(r2_score(m_oof["actual"], m_oof["predicted"]))
        pooled_mae = float(mean_absolute_error(m_oof["actual"], m_oof["predicted"]))
        pooled_rmse = float(root_mean_squared_error(m_oof["actual"], m_oof["predicted"]))

        comparison_rows.append({
            "method": method_label,
            "mean_n_features": mean_features,
            "mean_r2": round(mean_r2, 6),
            "std_r2": round(std_r2, 6),
            "mean_mae": round(mean_mae, 6),
            "std_mae": round(std_mae, 6),
            "mean_rmse": round(mean_rmse, 6),
            "std_rmse": round(std_rmse, 6),
            "pooled_oof_r2": round(pooled_r2, 6),
            "pooled_oof_mae": round(pooled_mae, 6),
            "pooled_oof_rmse": round(pooled_rmse, 6),
            "total_selection_time_sec": round(total_sel_time, 2),
            "mean_train_time_sec": round(mean_train_time, 4),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # -----------------------------------------------------------------------
    # Paired fold-by-fold comparisons (method R2 - baseline R2)
    # -----------------------------------------------------------------------
    paired_records = []
    for method_label, _, _ in methods_config:
        m_df = fold_metrics_df[fold_metrics_df["method"] == method_label].sort_values("fold")
        b_df = fold_metrics_df[fold_metrics_df["method"] == "Baseline (34 features)"].sort_values("fold")

        for fold_idx in range(n_folds):
            b_r2 = b_df[b_df["fold"] == fold_idx]["r2"].values[0]
            m_r2 = m_df[m_df["fold"] == fold_idx]["r2"].values[0]
            b_mae = b_df[b_df["fold"] == fold_idx]["mae"].values[0]
            m_mae = m_df[m_df["fold"] == fold_idx]["mae"].values[0]
            b_rmse = b_df[b_df["fold"] == fold_idx]["rmse"].values[0]
            m_rmse = m_df[m_df["fold"] == fold_idx]["rmse"].values[0]
            paired_diff = m_r2 - b_r2

            paired_records.append({
                "method": method_label,
                "fold": fold_idx,
                "baseline_r2": round(b_r2, 6),
                "method_r2": round(m_r2, 6),
                "paired_r2_diff": round(paired_diff, 6),
                "baseline_mae": round(b_mae, 6),
                "method_mae": round(m_mae, 6),
                "baseline_rmse": round(b_rmse, 6),
                "method_rmse": round(m_rmse, 6),
            })

    paired_df = pd.DataFrame(paired_records)

    # -----------------------------------------------------------------------
    # Feature selection stability across folds
    # -----------------------------------------------------------------------
    stability_records = []
    for method_label in ["Corr_filter_0.95", "Mutual_Info_top15", "F_test_top20", "Wrapper_RFECV"]:
        if method_label in selected_per_method_and_fold:
            stab = compute_stability(selected_per_method_and_fold[method_label], physics_features)
            stab.insert(0, "method", method_label)
            stability_records.append(stab)

    stability_df = pd.concat(stability_records, ignore_index=True) if stability_records else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Section 8: Idealized vs Realistic Subsets (Dropping subset_idealized)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Section 8: Idealized vs Realistic Subset-Specific Evaluation")
    print("Protocol: Folds restricted to subset (folds[mask]), dropping subset_idealized")
    print("=" * 70)

    subset_features = [f for f in physics_features if f != "subset_idealized"]
    assert len(subset_features) == 33, f"Expected 33 features after dropping subset_idealized, got {len(subset_features)}"

    subset_metrics_records = []
    subset_methods = [
        ("Baseline (33 features)", "baseline", {}),
        ("Corr_filter_0.95", "corr_filter", {"threshold": 0.95}),
        ("Mutual_Info_top15", "mi", {"k": 15}),
        ("F_test_top20", "f_test", {"k": 20}),
        ("Wrapper_RFECV", "rfecv", {"min_features": 5, "step": 1 if not smoke else 4}),
    ]

    for subset_name in ["idealized", "realistic"]:
        sub_mask = (full_df["subset"] == subset_name).values
        sub_df = full_df[sub_mask].reset_index(drop=True)
        sub_folds = sub_df["fold"].values.astype(int)
        sub_groups = sub_df["split_group"].values
        sub_y = sub_df[target].values
        n_cases = len(sub_df)
        print(f"\nSubset: {subset_name} ({n_cases} cases)")

        for method_label, method_type, params in subset_methods:
            sub_r2s = []
            sub_maes = []
            sub_rmses = []
            for fold_idx in range(n_folds):
                tr_m = sub_folds != fold_idx
                val_m = sub_folds == fold_idx
                if val_m.sum() == 0:
                    continue

                X_tr_sub_df = sub_df.loc[tr_m, subset_features].fillna(0)
                X_val_sub_df = sub_df.loc[val_m, subset_features].fillna(0)
                y_tr_sub = sub_y[tr_m]
                y_val_sub = sub_y[val_m]
                grp_tr_sub = sub_groups[tr_m]

                if method_type == "baseline":
                    sel = list(subset_features)
                elif method_type == "corr_filter":
                    sel = filter_correlation(X_tr_sub_df, y_tr_sub, threshold=params["threshold"])
                elif method_type == "mi":
                    sel = filter_mutual_info(X_tr_sub_df, y_tr_sub, k=min(params["k"], len(subset_features)), random_state=42)
                elif method_type == "f_test":
                    sel = filter_f_test(X_tr_sub_df, y_tr_sub, k=min(params["k"], len(subset_features)))
                elif method_type == "rfecv":
                    sel, _, _ = run_rfecv(
                        X_tr_sub_df, y_tr_sub, grp_tr_sub,
                        n_inner_splits=2 if smoke else 3,
                        min_features=params["min_features"],
                        step=params["step"],
                        random_state=42
                    )

                preds_sub, metrics_sub, _ = train_and_eval_lgbm(
                    X_tr_sub_df[sel].values, y_tr_sub,
                    X_val_sub_df[sel].values, y_val_sub,
                    n_estimators=400, learning_rate=0.05, random_state=42
                )
                sub_r2s.append(metrics_sub["r2"])
                sub_maes.append(metrics_sub["mae"])
                sub_rmses.append(metrics_sub["rmse"])

            mean_sub_r2 = float(np.mean(sub_r2s))
            std_sub_r2 = float(np.std(sub_r2s, ddof=0))
            mean_sub_mae = float(np.mean(sub_maes))
            mean_sub_rmse = float(np.mean(sub_rmses))

            subset_metrics_records.append({
                "subset": subset_name,
                "n_cases": n_cases,
                "method": method_label,
                "mean_r2": round(mean_sub_r2, 6),
                "std_r2": round(std_sub_r2, 6),
                "mean_mae": round(mean_sub_mae, 6),
                "mean_rmse": round(mean_sub_rmse, 6),
                "fold_r2s": ";".join([f"{r:.4f}" for r in sub_r2s]),
            })
            print(f"  {method_label:24s} | Mean R2={mean_sub_r2:.4f} +/- {std_sub_r2:.4f} | MAE={mean_sub_mae:.4f}")

    subset_metrics_df = pd.DataFrame(subset_metrics_records)

    # -----------------------------------------------------------------------
    # Save outputs (both raw source files and standard output files)
    # -----------------------------------------------------------------------
    fold_metrics_df.to_csv(output_dir / "fs_fold_metrics.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    paired_df.to_csv(output_dir / "paired_fold_comparisons.csv", index=False)
    selected_features_df.to_csv(output_dir / "selected_features.csv", index=False)
    stability_df.to_csv(output_dir / "feature_selection_stability.csv", index=False)
    oof_predictions_df.to_csv(output_dir / "fs_oof_predictions.csv", index=False)
    oof_predictions_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    subset_metrics_df.to_csv(output_dir / "fs_subset_metrics.csv", index=False)
    subset_metrics_df.to_csv(output_dir / "subset_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("Files Saved Successfully:")
    print(f"  {output_dir / 'fold_metrics.csv'} (and fs_fold_metrics.csv)")
    print(f"  {output_dir / 'comparison.csv'}")
    print(f"  {output_dir / 'paired_fold_comparisons.csv'}")
    print(f"  {output_dir / 'selected_features.csv'}")
    print(f"  {output_dir / 'feature_selection_stability.csv'}")
    print(f"  {output_dir / 'oof_predictions.csv'} (and fs_oof_predictions.csv)")
    print(f"  {output_dir / 'subset_metrics.csv'} (and fs_subset_metrics.csv)")
    print(f"  {output_dir / 'subset_metrics.csv'}")
    print("=" * 70)

    return {
        "fold_metrics": fold_metrics_df,
        "comparison": comparison_df,
        "paired": paired_df,
        "stability": stability_df,
        "selected_features": selected_features_df,
        "oof_predictions": oof_predictions_df,
        "subset_metrics": subset_metrics_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S06b feature selection experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test (2 folds, coarse step)")
    parser.add_argument("--target", type=str, default="U_mean", help="Target column name")
    args = parser.parse_args()

    run_experiments(smoke=args.smoke, target=args.target)
