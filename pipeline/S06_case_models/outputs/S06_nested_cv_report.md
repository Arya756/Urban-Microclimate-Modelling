# Stage S06 — Leakage-Safe Nested Cross-Validation Report

## Executive Summary
This report presents generalization estimates for LightGBM trained on 34 physics features, evaluated using a **nested cross-validation protocol** to eliminate hyperparameter selection bias.

- **Target:** `U_mean`
- **Protocol:** 5 outer folds (GroupKFold by spatial `split_group`) × 3 inner folds (GroupKFold by spatial `split_group`)
- **Tuning Budget:** 10 Optuna trials per outer fold (total 50 optimization trials)
- **Total Duration:** 8.4 seconds
- **Frozen Test Groups:** 24 groups (53 cases) completely excluded and unopened.

## Key Comparison Against Historical Benchmark

| Evaluation Protocol | Mean Outer Fold R² | Pooled OOF R² | Fold R² Vector | Selection Bias Status |
|---|---|---|---|---|
| Historical Single-CV (Leaked Tuning) | **0.8331** | ~0.833 | [0.7372, 0.8789, 0.9045, 0.7822, 0.8625] | **Potentially Optimistic** (Validation folds reused for Optuna tuning) |
| **Nested CV (Corrected Protocol)** | **0.8167** | **0.8156** | [0.8180, 0.8019, 0.9095, 0.7077, 0.8463] | **Leakage-Safe Estimate** (Hyperparameters selected on training folds only) |

> **Scientific Interpretation:**
> The historical R² of 0.8331 was obtained by running Optuna on all 5 development folds simultaneously, meaning the hyperparameters were selected with prior exposure to the test partitions. The nested CV estimate (0.8167 mean fold R², 0.8156 pooled R²) is an honest generalization estimate obtained without reusing outer validation folds for tuning.

## Detailed Outer Fold Performance

| Outer Fold | Cases | Spatial Groups | Outer Val R² | MAE (m/s) | RMSE (m/s) | Best Inner CV R² | Tuning Duration |
|---|---|---|---|---|---|---|---|
| Fold 0 | 100 | 28 | 0.8180 | 0.0356 | 0.0613 | 0.7701 | 1.21s |
| Fold 1 | 100 | 27 | 0.8019 | 0.0353 | 0.0634 | 0.7550 | 0.88s |
| Fold 2 | 91 | 27 | 0.9095 | 0.0299 | 0.0423 | 0.7727 | 1.8s |
| Fold 3 | 114 | 27 | 0.7077 | 0.0438 | 0.0681 | 0.7982 | 3.29s |
| Fold 4 | 77 | 27 | 0.8463 | 0.0320 | 0.0490 | 0.7439 | 1.17s |

## Performance by Morphological Subset

| Subset | Cases | Pooled R² | MAE (m/s) | RMSE (m/s) |
|---|---|---|---|---|
| **Idealized Geometry** | 190 | 0.9643 | 0.0154 | 0.0271 |
| **Realistic Geometry** | 292 | 0.6629 | 0.0491 | 0.0720 |
| **All Development Cases** | 482 | 0.8156 | 0.0358 | 0.0586 |

## Downstream Experiments & Status

The following downstream experiments in `run_s06.py` and associated notebooks relied on the historical single-split `best_params`:
1. **Monotonic Constraints Experiment (Step 6):** Constrained `lambda_p <= 0` using historical params.
2. **Transfer Tests (Step 7):** Cross-subset models trained using historical params.
3. **Global Feature Importance (Step 10):** LightGBM fit on all 482 dev cases with historical params.

These historical experiments are preserved as exploratory artifacts, but their metrics should not be interpreted as validated by the nested protocol.

## Protocol Limitations
1. Nested CV estimates generalization under the outer split distribution; slight variance across outer fold choices persists due to the finite number of city-level spatial clusters (136 groups).
2. The outer-fold-specific hyperparameters vary slightly across folds, which reflects the natural sensitivity of hyperparameter selection to training group composition.
