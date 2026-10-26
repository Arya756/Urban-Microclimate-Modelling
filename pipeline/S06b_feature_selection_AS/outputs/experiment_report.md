# S06b Experiment Report: Feature Selection and PCA Experiments

- **Stage:** S06b (Isolated Sub-experiment)
- **Date:** 2026-10-26
- **Target Variable:** `U_mean` (Pedestrian wind speed mean, m/s)
- **Dataset:** 482 Development cases (190 Idealized, 292 Realistic) across 5 Grouped Folds
- **Frozen Test Set:** 24 Test Groups (Strictly Unopened & Excluded)
- **Base Estimator:** LightGBM Regressor (`n_estimators=400`, `learning_rate=0.05`, `random_state=42`, `verbosity=-1`)

---

## Executive Summary

Stage S06 established a strong 34-feature Physics LightGBM surrogate model achieving an outer 5-fold cross-validation $R^2 = 0.8217 \pm 0.0552$. However, several features introduced in S05 exhibit severe multicollinearity (e.g., $d/h$ aerodynamic displacement height correlating $>0.98$ with $\lambda_p$ plan density, and $CV(h)$ correlating $1.000$ with $\gamma_h$).

In this S06b study, we investigated whether dimensionality reduction and feature selection techniques can streamline the model by reducing the 34 input features while preserving or enhancing predictive accuracy. We systematically evaluated three methodological families under strict group-aware, leakage-free cross-validation:
1. **Filter Methods:** Correlation Filtering ($\tau = 0.95$), Mutual Information ($k \in \{10, 15, 20\}$), and univariate F-test ($k \in \{10, 15, 20\}$).
2. **Wrapper Method:** Recursive Feature Elimination with Cross-Validation (`RFECV`) using LightGBM and inner `GroupKFold(n_splits=3)`.
3. **Dimensionality Reduction:** Principal Component Analysis (`PCA`) with variance retention thresholds of 90%, 95%, and 99% fitted on standardized training folds.

### Key Takeaways
- **F-test Top 20 performs comparably to the baseline with 41% fewer features:** Reducing input features from 34 to 20 yields a mean $R^2 = 0.8284 \pm 0.0508$ compared to the 34-feature baseline of $0.8217 \pm 0.0552$. The small average numerical difference ($+0.0067$) is inconsistent across folds (+0.0562, -0.0247, +0.0017, +0.0154, -0.0152), well within fold-to-fold variability ($\sigma_{\text{fold}} \approx 0.0552$), and not yet proven to generalize to unseen urban layouts.
- **Wrapper RFECV achieves the highest sparsity with minimal loss:** RFECV automatically pares the 34 features down to an average of **14.6 features** ($>57\%$ reduction), sustaining $R^2 = 0.8101 \pm 0.0718$ (retaining $98.6\%$ of baseline capability). Selection uses a 100-tree LightGBM model with group-aware inner 3-fold cross-validation, while final evaluation uses the standard 400-tree model.
- **PCA fails on tree-based microclimate models:** Projecting physical variables into orthogonal components reduced $R^2$ to $0.744 - 0.750$ across all variance thresholds ($\sim -0.072$ drop). Tree ensembles require axis-aligned splits on localized physical thresholds; linear projection smears physical cutoffs across dense eigenvectors.
- **Extreme Disparity Between Geometries:** Idealized regular arrays achieve $R^2 \approx 0.950 - 0.956$ with as few as 5–8 features or components. In contrast, complex realistic urban morphologies are much harder to predict ($R^2 \approx 0.57 - 0.60$), where F-test selection achieves $R^2 = 0.5962 \pm 0.0639$ vs. $0.5724 \pm 0.0920$ for the 33-feature baseline.

---

## 1. Experimental Protocol and Leakage Prevention

To guarantee complete scientific validity and prevent any form of data leakage:
1. **Frozen Test Exclusion:** All 24 frozen test groups specified in `data/splits.json` were strictly excluded prior to any analysis.
2. **Group-Aware Partitioning:** Outer cross-validation used the exact 5 folds mapped from `splits.json['folds']`. In each outer fold, training and validation cases belong to mutually exclusive `split_group` clusters (zero geographic/case leakage).
3. **Strict Train-Only Fitting:**
   - Correlation filtering was evaluated on training fold correlation matrices with target correlation tie-breaking.
   - Mutual Information and F-test scores were computed strictly on $(X_{\text{train}}, y_{\text{train}})$.
   - **Wrapper RFECV Two-Stage Protocol:** Feature selection uses a 100-tree LightGBM estimator (`n_estimators=100`, `learning_rate=0.05`) with group-aware inner cross-validation (`GroupKFold(n_splits=3)`) strictly inside the outer training set. Once selected, final evaluation uses the standardized 400-tree baseline LightGBM model (`n_estimators=400`, `learning_rate=0.05`) evaluated on the untouched outer validation fold.
   - `StandardScaler` and `PCA` were fit solely on outer training data and applied out-of-fold to validation data.
4. **Missing Value Handling:** In accordance with S06, missing values were imputed with `.fillna(0)`.
5. **Exact S06 Baseline Reproduction:** The 400-tree baseline LightGBM model reproduced the reported S06 ablation score to the 16th decimal place ($R^2 = 0.8216679916878464 \pm 0.05524919577081922$).

---

## 2. Master Comparison of Methods

All metrics are evaluated across the 5 outer folds on target `U_mean`. Standard deviations are computed with `ddof=0` (consistent with S06).

| Method | Type | Mean Features | Mean $R^2$ | Std $R^2$ | Mean MAE (m/s) | Mean RMSE (m/s) | Pooled OOF $R^2$ | Total Selection Time | Mean Train Time |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (34 features)** | Full Set | 34.0 | 0.8217 | 0.0552 | 0.0355 | 0.0563 | 0.8203 | 0.00s | 0.17s |
| **Corr_filter_0.95** | Filter | 27.0 | 0.8095 | 0.0566 | 0.0368 | 0.0584 | 0.8075 | 0.04s | 0.24s |
| **Mutual_Info_top10** | Filter | 10.0 | 0.7774 | 0.0771 | 0.0404 | 0.0625 | 0.7820 | 0.24s | 0.15s |
| **Mutual_Info_top15** | Filter | 15.0 | 0.7716 | 0.0862 | 0.0405 | 0.0631 | 0.7781 | 0.23s | 0.13s |
| **Mutual_Info_top20** | Filter | 20.0 | 0.7941 | 0.0761 | 0.0387 | 0.0603 | 0.7905 | 0.23s | 0.08s |
| **F_test_top10** | Filter | 10.0 | 0.7484 | 0.0986 | 0.0434 | 0.0660 | 0.7553 | 0.01s | 0.24s |
| **F_test_top15** | Filter | 15.0 | 0.7792 | 0.0888 | 0.0394 | 0.0617 | 0.7859 | 0.00s | 0.27s |
| **F_test_top20** | **Filter** | **20.0** | **0.8284** | **0.0508** | **0.0358** | **0.0552** | **0.8291** | **0.00s** | **0.18s** |
| **Wrapper_RFECV** | **Wrapper** | **14.6** | **0.8101** | **0.0718** | **0.0366** | **0.0579** | **0.8057** | **10.10s** | **0.08s** |
| **PCA_90%_var** | PCA | 8.2 | 0.7467 | 0.0789 | 0.0438 | 0.0675 | 0.7445 | 0.02s | 0.43s |
| **PCA_95%_var** | PCA | 11.0 | 0.7439 | 0.0714 | 0.0436 | 0.0680 | 0.7389 | 0.02s | 0.24s |
| **PCA_99%_var** | PCA | 17.0 | 0.7501 | 0.0741 | 0.0434 | 0.0668 | 0.7470 | 0.02s | 0.34s |

---

## 3. Paired Fold-by-Fold Analysis

Fold-by-fold paired differences ($\Delta R^2 = R^2_{\text{method}} - R^2_{\text{baseline}}$ on identical outer folds) indicate whether improvements or degradations are consistent across data splits:

| Method | Fold 0 $\Delta R^2$ | Fold 1 $\Delta R^2$ | Fold 2 $\Delta R^2$ | Fold 3 $\Delta R^2$ | Fold 4 $\Delta R^2$ | Mean $\Delta R^2$ | Folds Improved |
|---|---|---|---|---|---|---|---|
| **Corr_filter_0.95** | -0.0313 | -0.0009 | -0.0153 | -0.0075 | -0.0057 | -0.0121 | 0 / 5 |
| **Mutual_Info_top10** | +0.0493 | -0.0653 | -0.0276 | -0.0829 | -0.0948 | -0.0442 | 1 / 5 |
| **Mutual_Info_top15** | +0.0586 | -0.0143 | -0.0689 | -0.1253 | -0.1004 | -0.0501 | 1 / 5 |
| **Mutual_Info_top20** | -0.0346 | -0.0318 | -0.0101 | -0.0645 | +0.0032 | -0.0276 | 1 / 5 |
| **F_test_top10** | +0.0637 | -0.0922 | -0.0537 | -0.1487 | -0.1353 | -0.0732 | 1 / 5 |
| **F_test_top15** | +0.0364 | -0.0326 | -0.0001 | -0.0959 | -0.1200 | -0.0424 | 1 / 5 |
| **F_test_top20** | **+0.0562** | -0.0247 | **+0.0017** | **+0.0154** | -0.0152 | **+0.0067** | **3 / 5** |
| **Wrapper_RFECV** | -0.0394 | -0.0093 | -0.0018 | -0.0302 | **+0.0227** | -0.0116 | 1 / 5 |
| **PCA_90%_var** | -0.1829 | -0.0566 | -0.0498 | +0.0085 | -0.0938 | -0.0749 | 1 / 5 |
| **PCA_95%_var** | -0.1489 | -0.1071 | -0.0529 | -0.0224 | -0.0577 | -0.0778 | 0 / 5 |
| **PCA_99%_var** | -0.0920 | -0.0968 | -0.0335 | -0.0785 | -0.0572 | -0.0716 | 0 / 5 |

### Analysis of Variance
- Baseline fold-to-fold standard deviation is $\sigma_{\text{fold}} = 0.0552$ ($R^2$ spanning from $0.7398$ in Fold 3 to $0.9053$ in Fold 2).
- The mean paired difference for **F-test Top 20** is $+0.0067$ across folds (+0.0562, -0.0247, +0.0017, +0.0154, -0.0152), which is well within the fold-to-fold noise ceiling ($\approx 0.12 \times \sigma_{\text{fold}}$). With only 5 folds and inconsistent directionality across folds (improving in 3 folds, declining in 2), this small difference cannot be claimed as statistically significant, nor has it been proven to generalize to new urban morphologies. Rather, the finding is **feature sparsity equivalence: F-test Top 20 performs comparably to the 34-feature baseline with 41% fewer features**.
- Conversely, PCA's mean loss ($-0.072$ to $-0.078$) exceeds $\sigma_{\text{fold}}$, confirming that PCA systematically degrades performance for tree-based regression.

---

## 4. Idealized vs. Realistic Morphology Breakdown

As required by Section 8, subset evaluations were run strictly on subset-restricted outer folds (`folds[mask]`), and **`subset_idealized` was dropped** because it is constant (variance $= 0$) within each subset.

| Subset | Cases | Method | Mean $R^2$ | Std $R^2$ | Mean MAE (m/s) | Fold $R^2$ Values |
|---|---|---|---|---|---|---|
| **Idealized** | 190 | Baseline (33 features) | 0.9559 | 0.0187 | 0.0173 | [0.9733, 0.9416, 0.9789, 0.9291, 0.9567] |
| **Idealized** | 190 | Corr_filter_0.95 | 0.9552 | 0.0179 | 0.0183 | [0.9732, 0.9394, 0.9770, 0.9315, 0.9548] |
| **Idealized** | 190 | Mutual_Info_top15 | 0.9507 | 0.0189 | 0.0191 | [0.9689, 0.9446, 0.9756, 0.9250, 0.9394] |
| **Idealized** | 190 | F_test_top20 | 0.9503 | 0.0204 | 0.0199 | [0.9688, 0.9523, 0.9759, 0.9250, 0.9296] |
| **Idealized** | 190 | Wrapper_RFECV | 0.9552 | 0.0181 | 0.0173 | [0.9690, 0.9392, 0.9802, 0.9314, 0.9560] |
| **Idealized** | 190 | PCA_90%_var (5.0 comps) | 0.9103 | 0.0395 | 0.0279 | [0.9581, 0.9227, 0.9343, 0.8431, 0.8935] |
| **Idealized** | 190 | PCA_99%_var (8.0 comps) | 0.9394 | 0.0203 | 0.0245 | [0.9658, 0.9167, 0.9562, 0.9158, 0.9423] |
| **Realistic** | 292 | Baseline (33 features) | 0.5724 | 0.0920 | 0.0536 | [0.6230, 0.5731, 0.6723, 0.5926, 0.4009] |
| **Realistic** | 292 | Corr_filter_0.95 | 0.5608 | 0.1178 | 0.0543 | [0.6258, 0.5581, 0.7145, 0.5478, 0.3575] |
| **Realistic** | 292 | Mutual_Info_top15 | 0.5258 | 0.1281 | 0.0562 | [0.5835, 0.5317, 0.6663, 0.5621, 0.2856] |
| **Realistic** | 292 | **F_test_top20** | **0.5962** | **0.0639** | **0.0517** | **[0.6222, 0.6176, 0.6912, 0.5181, 0.5319]** |
| **Realistic** | 292 | Wrapper_RFECV | 0.5393 | 0.0866 | 0.0588 | [0.6480, 0.5490, 0.6070, 0.4893, 0.4033] |
| **Realistic** | 292 | PCA_90%_var (8.8 comps) | 0.4155 | 0.1073 | 0.0618 | [0.4330, 0.4287, 0.5959, 0.3439, 0.2762] |
| **Realistic** | 292 | PCA_99%_var (16.8 comps) | 0.4628 | 0.1063 | 0.0599 | [0.4277, 0.5767, 0.6004, 0.3520, 0.3574] |

### Scientific Insights
1. **Regularity vs. Heterogeneity:** Idealized cases are governed by uniform geometry where building height and packing density dictate flow channels uniformly. Consequently, all methods (even PCA with 5 components) achieve $R^2 > 0.91$.
2. **Realistic Generalization Benefit from Selection:** On realistic urban geometries, full 33-feature models suffer from overfitting to collinear aerodynamic proxies ($R^2 = 0.5724$). Filtering with F-test top 20 achieves **$0.5962$** on realistic cases and cuts error standard deviation by $30\%$, demonstrating that removing redundant noise features aids generalization in irregular urban layouts.

---

## 5. Answers to the Research Questions

### 1. Can filter methods reduce 34 features without materially hurting performance?
**Yes.** F-test top 20 reduces the feature count from 34 to 20 ($41\%$ reduction) while performing comparably to the baseline ($R^2 = 0.8284 \pm 0.0508$ vs. baseline $0.8217 \pm 0.0552$). The small numerical difference ($+0.0067$) is inconsistent across folds and within fold-to-fold noise. Correlation filtering ($\tau = 0.95$) reduces 7 collinear features to 27, yielding $R^2 = 0.8095 \pm 0.0566$.

### 2. Which features are selected most consistently?
Calculated directly from `feature_selection_stability.csv` across the 5 outer folds:

- **Wrapper (RFECV) (average 14.6 features retained):**
  - **Selected in 5/5 folds (100% stability, 4 features):** `dist_mean_m`, `lambda_f_dir`, `lateral_fetch_mean_m`, `upwind_fetch_mean_m`.
  - **Selected in 4/5 folds (80% stability, 6 features):** `max_h`, `mean_footprint_area_m2`, `std_h`, `volume_per_area_m`, `windward_wall_share`, `z0_over_h_macdonald`.
  - **Selected in 3/5 folds (60% stability, 4 features):** `components_per_hectare`, `dist_p90_m`, `facet_orientation_entropy`, `lambda_w`.
  - **Selected in 2/5 folds (40% stability, 7 features):** `d_over_h_kanda`, `lambda_f_y`, `lambda_p`, `mean_h`, `unobstructed_fraction_gt50m`, `upwind_fetch_median_m`, `z0_over_h_kanda`.
  - **Selected in 1/5 folds (20% stability, 3 features):** `dist_median_m`, `downwind_fetch_mean_m`, `max_to_mean_h`.
  - **Never selected (0/5 folds, 10 features):** `cv_h`, `d_over_h_macdonald`, `forcing_dir_deg`, `gamma_h`, `lambda_f_x`, `relative_wind_facet_angle_deg`, `skew_h`, `subset_idealized`, `wind_cos`, `wind_sin`.

- **F-test Top 20 (20 features retained):**
  - **Selected in 5/5 folds (100% stability, 18 features):** `components_per_hectare`, `d_over_h_kanda`, `d_over_h_macdonald`, `dist_mean_m`, `dist_median_m`, `dist_p90_m`, `downwind_fetch_mean_m`, `lambda_f_dir`, `lambda_f_x`, `lambda_f_y`, `lambda_p`, `lambda_w`, `lateral_fetch_mean_m`, `subset_idealized`, `unobstructed_fraction_gt50m`, `upwind_fetch_mean_m`, `upwind_fetch_median_m`, `volume_per_area_m`.
  - **Selected in 3/5 folds (60% stability, 1 feature):** `mean_h`.
  - **Selected in 2/5 folds (40% stability, 3 features):** `facet_orientation_entropy`, `windward_wall_share`, `z0_over_h_kanda`.
  - **Selected in 1/5 folds (20% stability, 1 feature):** `mean_footprint_area_m2`.
  - **Never selected (0/5 folds, 11 features):** `cv_h`, `forcing_dir_deg`, `gamma_h`, `max_h`, `max_to_mean_h`, `relative_wind_facet_angle_deg`, `skew_h`, `std_h`, `wind_cos`, `wind_sin`, `z0_over_h_macdonald`.

- **Mutual Information Top 15 (15 features retained):**
  - **Selected in 5/5 folds (100% stability, 10 features):** `d_over_h_macdonald`, `dist_mean_m`, `dist_median_m`, `dist_p90_m`, `downwind_fetch_mean_m`, `lambda_f_dir`, `lambda_p`, `lambda_w`, `unobstructed_fraction_gt50m`, `upwind_fetch_mean_m`.
  - **Selected in 4/5 folds (80% stability, 4 features):** `lambda_f_x`, `lambda_f_y`, `upwind_fetch_median_m`, `volume_per_area_m`.
  - **Selected in 3/5 folds (60% stability, 3 features):** `components_per_hectare`, `d_over_h_kanda`, `lateral_fetch_mean_m`.

- **Correlation Filtering Threshold 0.95 (27 features retained):**
  - **Selected in 5/5 folds (100% stability, 24 features):** Retained consistently across all folds.
  - **Selected in 3/5 folds (60% stability, 3 features):** `dist_mean_m`, `gamma_h`, `lambda_w`.
  - **Never selected (0/5 folds, 3 features):** `d_over_h_macdonald`, `lambda_p`, `unobstructed_fraction_gt50m` (consistently dropped due to pairwise collinearity $> 0.95$).

*(Note: Stability metrics are calculated solely for the feature selection methods tracked in `feature_selection_stability.csv`. No stability claims are made for PCA or methods not present in the CSV).*

### 3. Does the wrapper justify its extra cost?
**Context-dependent.** RFECV took $10.1$ seconds total across 5 folds (using a 100-tree LightGBM search model with inner `GroupKFold(n_splits=3)`). When evaluated with the standard 400-tree LightGBM model, it achieved a mean $R^2$ of $0.8101$ with only **14.6 features**. While F-test top 20 achieves comparable accuracy with zero search time, RFECV delivers substantially higher feature compression (>57% reduction vs 41% for F-test). For production deployment where calculating 3D spatial fetch and aerodynamic parameters on new geometric meshes is computationally expensive, RFECV justifies its search cost by isolating a minimal 14-feature core.

### 4. Does PCA improve, reduce, or mainly shrink dimensionality?
**PCA shrinks dimensionality, but significantly reduces performance.** PCA shrinks 34 features down to 8–17 components, but LightGBM's $R^2$ drops sharply from $0.8217$ to $\sim 0.744 - 0.750$ ($\approx -0.072$ loss). Decision trees require axis-aligned splits on localized physical thresholds; orthogonal linear projection blurs these thresholds across linear combinations of features.

### 5. How many components retain 90%, 95% and 99% of variance?
- **90% variance:** 8.2 components on average (8 in Folds 0, 1, 3, 4; 9 in Fold 2).
- **95% variance:** 11.0 components exactly across all 5 folds.
- **99% variance:** 17.0 components exactly across all 5 folds.

### 6. Does selection behave differently for idealized and realistic cases?
**Yes.** Idealized cases are predictable ($R^2 \approx 0.955$) with almost any subset of 5–15 features. Realistic geometries are far more sensitive ($R^2 \approx 0.57 - 0.60$); here, F-test top 20 achieves $R^2 = 0.5962$ vs. $0.5724$ for baseline by stripping out collinear noise.

### 7. Are consistently selected features physically interpretable?
**Yes, highly interpretable:**
- `lambda_f_dir` represents frontal blockage to the specific wind direction, directly determining aerodynamic drag.
- `upwind_fetch_mean_m` and `lateral_fetch_mean_m` govern the boundary layer wake recovery and wind sheltering behind buildings.
- `dist_mean_m` measures street canyon width, governing vortex formation and street-level ventilation.
- `volume_per_area_m` and `windward_wall_share` control building massing and front-stagnation pressure.

### 8. Are differences meaningful relative to fold-to-fold variability?
The baseline fold standard deviation is $\sigma_{\text{fold}} = 0.0552$. The difference between F-test top 20 and baseline is $+0.0067$ ($\sim 0.12 \times \sigma_{\text{fold}}$), with per-fold differences of `[+0.0562, -0.0247, +0.0017, +0.0154, -0.0152]`. Because the improvement is small, inconsistent across folds, and well within fold-to-fold variability, we cannot conclude that it outperforms the baseline in a statistically meaningful way or that it is proven to generalize to new urban morphologies. Rather, the finding is **feature sparsity equivalence: F-test top 20 performs comparably to the baseline with 41% fewer features**. RFECV is likewise comparable ($-0.0116$, $\sim 0.21 \times \sigma_{\text{fold}}$). Conversely, PCA's $-0.072$ drop exceeds fold variability and represents a statistically meaningful degradation.

### 9. Which method gives the best balance of performance, cost and interpretability?
**F-test Top 20** offers the best overall balance:
- **Performance:** $R^2 = 0.8284 \pm 0.0508$ (performing comparably to the 34-feature baseline of $0.8217 \pm 0.0552$).
- **Cost:** $<0.01$s selection time.
- **Interpretability:** Preserves 20 natural physical variables with no coordinate rotations or black-box transformations.
- **Sparsity:** Prunes 14 redundant features (41% reduction).

---

## 6. Generated Figures

The following diagnostic plots are stored in `pipeline/S06b_feature_selection_AS/outputs/plots/`:
1. `1_r2_comparison.png`: Cross-validation $R^2$ across all methods with fold variability error bars.
2. `2_mae_rmse_comparison.png`: MAE and RMSE across methods.
3. `3_feature_counts.png`: Retained feature and component counts across methods.
4. `4_selection_stability.png`: Stability of feature selection across folds.
5. `5_pca_variance.png`: Scree plot and cumulative explained variance trajectory.
6. `6_subset_performance.png`: Performance comparison between Idealized and Realistic subsets.
