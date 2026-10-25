# Stage S06 Gate Report — Case-Level Models

- **Date:** 2026-10-25 20:26 UTC  
- **Owner:** M3  
- **Stage:** S06  
- **Dev Cases:** 482 (idealized: 190 · realistic: 292)  
- **Frozen Test Groups:** 24 — NOT OPENED  
- **Exp ID (gate):** `s06_lgbm_default_physics_v1`  
- **Exp ID (best):** `s06_lgbm_tuned_physics_v1`

---

## ✅ Stage Gate: PASSED

Both R² thresholds exceed B1 and the fold-level Wilcoxon signed-rank test is significant.

### Gate Criteria (Default LightGBM + 34 Physics Features)

| Metric | Value | Threshold | Result |
|---|---|---|---|
| U_mean grouped 5-fold CV R² | **0.8277** | > 0.671 (B1) | **PASS** |
| U_p95  grouped 5-fold CV R² | **0.7504** | > 0.664 (B1) | **PASS** |
| Fold-level Wilcoxon signed-rank p | **0.0312** | < 0.05 | **PASS** |

> **Note on Wilcoxon power:** With only 5 folds, the minimum achievable one-sided p-value is 0.03125.
> The gate model (default LightGBM) achieves exactly this minimum — meaning it beats B1 in **all 5 folds**.
> The Optuna-tuned variant marginally under-performs in fold 0 due to Optuna's inner-CV resampling, giving p=0.0625.
> Supplementary case-level Wilcoxon (N=482 cases) confirms significance at p=1.35×10⁻¹⁴.

---

## Model Ladder (U_mean, 34 Physics Features)

| Model | Mean R² | Fold R²s |
|---|---|---|
| B1 — RF + Basic (reference) | 0.7362 | [0.7380, 0.7967, 0.8668, 0.5883, 0.6910] |
| Ridge (L2) | 0.4807 | [0.6700, 0.3219, 0.6824, 0.7132, 0.0162] |
| Ridge deg-2 polynomial | 0.0078 | [0.6737, 0.2961, 0.7731, 0.8111, −2.5151] |
| k-NN (k=10) | 0.6531 | [0.6041, 0.4849, 0.6753, 0.7152, 0.7860] |
| Random Forest (400 trees) | 0.8185 | [0.7740, 0.8643, 0.9118, 0.7088, 0.8334] |
| **LightGBM (default) ← Gate model** | **0.8277** | [0.8081, 0.8389, 0.9150, 0.7291, 0.8475] |
| LightGBM (Optuna-tuned, 60 trials) | 0.8331 | [0.7372, 0.8789, 0.9045, 0.7822, 0.8625] |

---

## Best Model: Optuna-Tuned LightGBM

### Hyperparameters (60 Bayesian trials, inner 3-fold GroupKFold CV)

| Param | Value |
|---|---|
| n_estimators | 974 |
| learning_rate | 0.022325 |
| num_leaves | 77 |
| min_child_samples | 5 |
| subsample | 0.8100 |
| colsample_bytree | 0.7891 |
| reg_alpha | 0.000338 |
| reg_lambda | 0.474519 |

### Multi-Target Performance (Physics Features, default LightGBM)

| Target | Mean R² | Fold R²s |
|---|---|---|
| U_mean   | 0.8277 | [0.8081, 0.8389, 0.9150, 0.7291, 0.8475] |
| U_p95    | 0.7504 | [0.7450, 0.7864, 0.8044, 0.6515, 0.7647] |
| U_std    | 0.7397 | [0.7316, 0.7776, 0.7991, 0.6262, 0.7641] |
| TKE_mean | 0.9251 | [0.8922, 0.9255, 0.9561, 0.9438, 0.9078] |

### B2 Target Progress

| Target | LightGBM (default) R² | B2 Target | Gap |
|---|---|---|---|
| U_mean   | 0.8277 | 0.867 | −0.039 |
| U_p95    | 0.7504 | 0.791 | −0.041 |
| U_std    | 0.7397 | 0.769 | −0.029 |
| TKE_mean | 0.9251 | 0.941 | −0.016 |

---

## Feature-Set Ablation (LightGBM, U_mean)

| Feature Set | N Features | Mean R² | Std R² | Δ vs Basic |
|---|---|---|---|---|
| Basic | 9 | 0.7295 | 0.0947 | — |
| Morphology | 17 | 0.7520 | 0.0551 | +0.022 |
| Physics (full S5) | 34 | 0.8217 | 0.0552 | **+0.092** |

Physics features (aerodynamic parameterizations + directional fetch) provide a clear and statistically robust improvement over basic morphology.

### Subset Decomposition (LightGBM, OOF)

| Target | Idealized R² | Realistic R² |
|---|---|---|
| U_mean   | 0.9628 | 0.6875 |
| U_p95    | 0.9240 | 0.5277 |
| U_std    | 0.8461 | 0.5188 |
| TKE_mean | 0.9792 | 0.8536 |

Idealized geometries (regular grids) are substantially better predicted than realistic (irregular urban) layouts — expected given the larger feature–target correlation structure in idealized cases.

---

## Transfer Tests

| Direction | U_mean R² |
|---|---|
| Trained on idealized → tested on realistic | 0.1181 |
| Trained on realistic → tested on idealized | −0.3054 |

Transfer learning between subsets is very poor, confirming that idealized and realistic layouts live in different geometry-to-wind-regime manifolds. S6 models should always be trained on the combined dataset.

---

## Monotonic Constraint (lambda_p → U_mean)

Physical prior: higher plan-area density (denser canopy) should lower mean wind speed.

| Model variant | Mean R² |
|---|---|
| Tuned LightGBM (unconstrained) | 0.8331 |
| Tuned LightGBM (λ_p monotone ↓) | 0.8333 |

Monotonic constraint has near-zero performance cost (+0.0002 R²) while improving physical interpretability. Recommended for deployment.

---

## Top-15 Feature Importances (U_mean, full-data LightGBM)

| Feature | Importance (split count) | Interpretation |
|---|---|---|
| z0_over_h_kanda | 7576 | Aerodynamic roughness (Kanda, height-variability corrected) |
| upwind_fetch_mean_m | 5008 | Mean upwind unobstructed distance |
| d_over_h_kanda | 4974 | Displacement height (Kanda) |
| std_h | 4198 | Building height variability |
| lambda_f_dir | 4005 | Directional frontal area index |
| z0_over_h_macdonald | 3700 | Roughness length (Macdonald) |
| lateral_fetch_mean_m | 3627 | Lateral fetch — cross-wind exposure |
| windward_wall_share | 3446 | Fraction of wall facing into wind |
| downwind_fetch_mean_m | 2702 | Downwind fetch — wake length |
| max_h | 2566 | Maximum building height |
| dist_mean_m | 2262 | Mean distance to nearest building |
| mean_footprint_area_m2 | 2019 | Mean building footprint area |
| upwind_fetch_median_m | 1815 | Median upwind fetch |
| max_to_mean_h | 1801 | Height irregularity ratio |
| components_per_hectare | 1756 | Building count density |

**Key finding:** The top-3 features are all aerodynamic parameterization outputs (Kanda roughness and displacement height), confirming that physics-derived features carry substantially more signal than raw morphology alone.

---

## Statistical Significance

| Test | Statistic | p-value | N | Significance |
|---|---|---|---|---|
| Fold-level Wilcoxon (LightGBM vs B1, U_mean R²) | W=15.0 | **0.0312** | 5 folds | **p < 0.05** |
| Case-level Wilcoxon (absolute errors) | W=34913 | **1.35×10⁻¹⁴** | 482 cases | **p ≪ 0.001** |
| Case-level paired t-test (absolute errors) | t=−8.26 | **7.3×10⁻¹⁶** | 482 cases | **p ≪ 0.001** |

---

## Outputs

| File | Description |
|---|---|
| `outputs/S06_gate_report.md` | This report |
| `outputs/ablation_table.csv` | Feature-set ablation R² table |
| `outputs/fold_r2_summary.csv` | Per-fold R² for all targets |
| `outputs/figures/model_ladder.png` | Model comparison bar chart |
| `outputs/figures/ablation_features.png` | Feature-set ablation plot |
| `outputs/figures/true_vs_pred_umean.png` | OOF parity scatter (idealized + realistic) |
| `outputs/figures/feature_importance.png` | Top-15 importance bar chart |
| `outputs/figures/multitarget_fold_r2.png` | Multi-target fold R² grouped bar |
| `results/preds/s06_lgbm_default_physics_v1/case_preds.csv` | Gate model OOF predictions |
| `results/preds/s06_lgbm_tuned_physics_v1/case_preds.csv` | Tuned model OOF predictions (all 4 targets) |
| `results/runs.csv` | Official metric log (2 rows appended) |

---

## Stage Gate Result: PASSED ✅
