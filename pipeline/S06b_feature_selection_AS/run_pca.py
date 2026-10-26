"""
S06b: Principal Component Analysis (PCA) Experiments
Stage: S06b
Owner: AS

Implements and evaluates:
  1. PCA dimensionality reduction with variance retention thresholds: 90%, 95%, 99%
  2. StandardScaler and PCA fit strictly on training fold only (zero leakage)
  3. LightGBM (400 trees, lr=0.05) evaluated on PCA components
  4. Component loadings and explained variance ratio extraction
  5. Subset-specific evaluation (Idealized vs Realistic, dropping subset_idealized)

Usage:
  python3 pipeline/S06b_feature_selection_AS/run_pca.py
  python3 pipeline/S06b_feature_selection_AS/run_pca.py --smoke
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
    fit_transform_pca,
    train_and_eval_lgbm,
)


def run_pca_experiments(
    smoke: bool = False,
    target: str = "U_mean",
    output_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = CURRENT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"S06b: PCA Experiments (Target: {target})")
    print(f"Mode: {'SMOKE TEST' if smoke else 'FULL EXPERIMENT'}")
    print("=" * 70)

    full_df, physics_features = load_dataset(REPO_ROOT)
    y_all = full_df[target].values
    folds = full_df["fold"].values.astype(int)
    n_folds = 2 if smoke else 5

    variance_thresholds = [0.90, 0.95, 0.99]

    fold_metrics_records = []
    explained_var_records = []
    loadings_records = []
    oof_predictions_records = []

    for var_thresh in variance_thresholds:
        method_label = f"PCA_{int(var_thresh * 100)}%_var"
        print(f"\nEvaluating {method_label} across {n_folds} folds...")

        for fold_idx in range(n_folds):
            train_mask = folds != fold_idx
            val_mask = folds == fold_idx

            X_tr_df = full_df.loc[train_mask, physics_features].fillna(0)
            X_val_df = full_df.loc[val_mask, physics_features].fillna(0)
            y_tr = y_all[train_mask]
            y_val = y_all[val_mask]
            val_cases = full_df.loc[val_mask, "case"].values

            t0_pca = time.time()
            X_tr_pca, X_val_pca, pca, scaler = fit_transform_pca(
                X_tr_df, X_val_df, variance_threshold=var_thresh, random_state=42
            )
            pca_time = time.time() - t0_pca
            n_components = pca.n_components_

            # Record explained variance per component
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            for comp_idx, (ev, cv) in enumerate(zip(pca.explained_variance_ratio_, cum_var)):
                explained_var_records.append({
                    "method": method_label,
                    "fold": fold_idx,
                    "component": comp_idx + 1,
                    "explained_variance_ratio": round(float(ev), 6),
                    "cumulative_variance_ratio": round(float(cv), 6),
                })

            # Record component loadings (eigenvectors)
            for comp_idx in range(n_components):
                for feat_idx, feat_name in enumerate(physics_features):
                    loadings_records.append({
                        "method": method_label,
                        "fold": fold_idx,
                        "component": comp_idx + 1,
                        "feature": feat_name,
                        "loading": round(float(pca.components_[comp_idx, feat_idx]), 6),
                    })

            # Train LightGBM on PCA components
            preds, metrics, train_time = train_and_eval_lgbm(
                X_tr_pca, y_tr, X_val_pca, y_val,
                n_estimators=400, learning_rate=0.05, random_state=42
            )

            fold_metrics_records.append({
                "method": method_label,
                "fold": fold_idx,
                "n_components": n_components,
                "pca_time_sec": round(pca_time, 4),
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

            print(f"  Fold {fold_idx}: comps={n_components:2d} | R2={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | PCA_time={pca_time:5.3f}s")

    pca_fold_metrics_df = pd.DataFrame(fold_metrics_records)
    explained_var_df = pd.DataFrame(explained_var_records)
    loadings_df = pd.DataFrame(loadings_records)
    pca_oof_df = pd.DataFrame(oof_predictions_records)

    # -----------------------------------------------------------------------
    # Subset evaluation for PCA (dropping subset_idealized)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PCA: Idealized vs Realistic Subsets (Dropping subset_idealized)")
    print("=" * 70)
    subset_features = [f for f in physics_features if f != "subset_idealized"]
    subset_pca_records = []

    for subset_name in ["idealized", "realistic"]:
        sub_mask = (full_df["subset"] == subset_name).values
        sub_df = full_df[sub_mask].reset_index(drop=True)
        sub_folds = sub_df["fold"].values.astype(int)
        sub_y = sub_df[target].values

        for var_thresh in variance_thresholds:
            method_label = f"PCA_{int(var_thresh * 100)}%_var"
            sub_r2s = []
            sub_maes = []
            sub_rmses = []
            comps_list = []

            for fold_idx in range(n_folds):
                tr_m = sub_folds != fold_idx
                val_m = sub_folds == fold_idx
                if val_m.sum() == 0:
                    continue

                X_tr_s = sub_df.loc[tr_m, subset_features].fillna(0)
                X_val_s = sub_df.loc[val_m, subset_features].fillna(0)
                y_tr_s = sub_y[tr_m]
                y_val_s = sub_y[val_m]

                X_tr_p, X_val_p, pca_s, _ = fit_transform_pca(
                    X_tr_s, X_val_s, variance_threshold=var_thresh, random_state=42
                )
                comps_list.append(pca_s.n_components_)

                preds_p, metrics_p, _ = train_and_eval_lgbm(
                    X_tr_p, y_tr_s, X_val_p, y_val_s,
                    n_estimators=400, learning_rate=0.05, random_state=42
                )
                sub_r2s.append(metrics_p["r2"])
                sub_maes.append(metrics_p["mae"])
                sub_rmses.append(metrics_p["rmse"])

            subset_pca_records.append({
                "subset": subset_name,
                "n_cases": len(sub_df),
                "method": method_label,
                "mean_n_components": round(float(np.mean(comps_list)), 1),
                "mean_r2": round(float(np.mean(sub_r2s)), 6),
                "std_r2": round(float(np.std(sub_r2s, ddof=0)), 6),
                "mean_mae": round(float(np.mean(sub_maes)), 6),
                "mean_rmse": round(float(np.mean(sub_rmses)), 6),
                "fold_r2s": ";".join([f"{r:.4f}" for r in sub_r2s]),
            })
            print(f"  Subset: {subset_name:9s} | {method_label:14s} | Comps={np.mean(comps_list):.1f} | Mean R2={np.mean(sub_r2s):.4f}")

    subset_pca_df = pd.DataFrame(subset_pca_records)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    pca_fold_metrics_df.to_csv(output_dir / "pca_fold_metrics.csv", index=False)
    explained_var_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    loadings_df.to_csv(output_dir / "pca_component_loadings.csv", index=False)
    pca_oof_df.to_csv(output_dir / "pca_oof_predictions.csv", index=False)
    subset_pca_df.to_csv(output_dir / "pca_subset_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("PCA Files Saved Successfully:")
    print(f"  {output_dir / 'pca_fold_metrics.csv'}")
    print(f"  {output_dir / 'pca_explained_variance.csv'}")
    print(f"  {output_dir / 'pca_component_loadings.csv'}")
    print(f"  {output_dir / 'pca_oof_predictions.csv'}")
    print(f"  {output_dir / 'pca_subset_metrics.csv'}")
    print("=" * 70)

    return {
        "pca_fold_metrics": pca_fold_metrics_df,
        "explained_var": explained_var_df,
        "loadings": loadings_df,
        "pca_oof": pca_oof_df,
        "subset_pca": subset_pca_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S06b PCA experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test (2 folds)")
    parser.add_argument("--target", type=str, default="U_mean", help="Target column name")
    args = parser.parse_args()

    run_pca_experiments(smoke=args.smoke, target=args.target)
