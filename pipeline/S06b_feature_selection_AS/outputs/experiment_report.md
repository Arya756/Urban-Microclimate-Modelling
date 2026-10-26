# S06b Experiment Report: Feature Selection and PCA Experiments

- **Stage:** S06b (isolated sub-experiment)
- **Target variable:** `U_mean`, the mean pedestrian-level wind speed over open ground, normalised by each case's friction velocity (U/u*)
- **Dataset:** 482 development cases (190 idealized, 292 realistic) across 5 grouped folds
- **Frozen test set:** 24 test groups, never opened
- **Base estimator:** LightGBM regressor (`n_estimators=400`, `learning_rate=0.05`, `random_state=42`, `verbosity=-1`)

> **Update.** The tables below come from a rerun on the corrected inputs. In the first run, `U_mean` was in m/s and its open-ground mask came from NaN in the wind files; it is now U/u* with the open ground taken from the building map (`pipeline/S02_preprocessing/outputs/case_targets.csv`). The S05 features now use the published Kanda (2013) parameterisation. The main conclusion changed: in the first run F-test top 20 edged the baseline; on the corrected data no selection method beats the full feature set.

---

## Executive Summary

The 34-feature physics LightGBM model reaches an outer 5-fold $R^2 = 0.8170 \pm 0.0627$ (pooled out-of-fold $R^2 = 0.8199$). Several S05 features are strongly collinear (for example $CV(h)$ and $\gamma_h$, or Macdonald $d/H$ and $\lambda_p$). This study asks whether feature selection or dimensionality reduction can shrink the 34 inputs without losing accuracy, under strict group-aware, leakage-free cross-validation:
1. **Filter methods:** correlation filtering ($\tau = 0.95$), mutual information ($k \in \{10, 15, 20\}$) and the univariate F-test ($k \in \{10, 15, 20\}$).
2. **Wrapper method:** recursive feature elimination with cross-validation (`RFECV`) using LightGBM and inner `GroupKFold(n_splits=3)`.
3. **Dimensionality reduction:** principal component analysis (`PCA`) keeping 90%, 95% and 99% of the variance of the standardised training features.

### Key Takeaways
- **No method beats the full feature set.** The best reductions are statistically equivalent to it.
- **Correlation filtering is the safe reduction:** it drops about 6 collinear features (27.8 kept) with $R^2 = 0.8162 \pm 0.0503$, a mean paired difference of -0.0008.
- **RFECV keeps half the features with a small loss:** 16.6 features on average, $R^2 = 0.8099 \pm 0.0792$ (-0.0070). The corrected Kanda roughness and displacement height are selected in every fold.
- **F-test top 20** reaches $R^2 = 0.8068 \pm 0.0458$ (-0.0102). Mutual-information subsets lose more.
- **PCA hurts:** $R^2 = 0.721$ to $0.756$ across the three variance thresholds. Tree ensembles split on individual physical thresholds, and linear projections mix those thresholds together.
- **Idealized and realistic cases differ sharply:** idealized arrays reach $R^2 \approx 0.95$ with any selection, while realistic neighbourhoods reach only $0.43$ to $0.58$ when modelled on their own.

---

## 1. Experimental Protocol and Leakage Prevention

1. **Frozen test exclusion:** the 24 frozen test groups in `data/splits.json` are excluded before any analysis.
2. **Group-aware partitioning:** the outer cross-validation uses the 5 folds of `splits.json['folds']`, so training and validation cases never share a `split_group` (city or layout family).
3. **Train-only fitting:**
   - Correlation filtering uses the training-fold correlation matrix, breaking ties by correlation with the target.
   - Mutual information and F-test scores are computed on $(X_{train}, y_{train})$ only.
   - **RFECV:** selection uses a 100-tree LightGBM model with inner `GroupKFold(n_splits=3)` inside the outer training set; the selected features are then scored with the 400-tree baseline model on the untouched outer validation fold.
   - `StandardScaler` and `PCA` are fitted on the outer training data only.
4. **Missing values:** imputed with `.fillna(0)`, as in S06 (the corrected feature table has none).
5. **Baseline:** the 400-tree LightGBM configuration of the S06 ablation, rerun on the corrected targets and features.

---

## 2. Master Comparison of Methods

Metrics over the 5 outer folds on `U_mean`; standard deviations use `ddof=0`.

| Method | Type | Mean features | Mean R² | Std R² | Mean MAE (U/u*) | Mean RMSE (U/u*) | Pooled OOF R² | Selection time (total) | Mean train time |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (34 features) | Full set | 34.0 | 0.8170 | 0.0627 | 0.1695 | 0.2688 | 0.8199 | 0.00 s | 0.70 s |
| Corr_filter_0.95 | Filter | 27.8 | 0.8162 | 0.0503 | 0.1742 | 0.2721 | 0.8160 | 0.04 s | 0.72 s |
| Mutual_Info_top10 | Filter | 10.0 | 0.7729 | 0.0964 | 0.1900 | 0.2965 | 0.7754 | 0.13 s | 0.62 s |
| Mutual_Info_top15 | Filter | 15.0 | 0.7548 | 0.0784 | 0.1945 | 0.3117 | 0.7601 | 0.13 s | 0.64 s |
| Mutual_Info_top20 | Filter | 20.0 | 0.7966 | 0.0732 | 0.1814 | 0.2844 | 0.7934 | 0.12 s | 0.64 s |
| F_test_top10 | Filter | 10.0 | 0.7434 | 0.0979 | 0.2110 | 0.3165 | 0.7507 | 0.00 s | 0.62 s |
| F_test_top15 | Filter | 15.0 | 0.7739 | 0.0676 | 0.1940 | 0.2999 | 0.7804 | 0.00 s | 0.64 s |
| F_test_top20 | Filter | 20.0 | 0.8068 | 0.0458 | 0.1827 | 0.2797 | 0.8078 | 0.00 s | 0.66 s |
| Wrapper_RFECV | Wrapper | 16.6 | 0.8099 | 0.0792 | 0.1725 | 0.2718 | 0.8116 | 62.86 s | 0.63 s |
| PCA_90%_var | PCA | 8.2 | 0.7208 | 0.1164 | 0.2190 | 0.3327 | 0.7179 | 0.01 s | 0.61 s |
| PCA_95%_var | PCA | 11.2 | 0.7482 | 0.0928 | 0.2076 | 0.3169 | 0.7442 | 0.00 s | 0.63 s |
| PCA_99%_var | PCA | 18.0 | 0.7557 | 0.0752 | 0.2016 | 0.3127 | 0.7551 | 0.00 s | 0.64 s |

---

## 3. Paired Fold-by-Fold Analysis

$\Delta R^2 = R^2_{method} - R^2_{baseline}$ on identical outer folds:

| Method | Fold 0 ΔR² | Fold 1 ΔR² | Fold 2 ΔR² | Fold 3 ΔR² | Fold 4 ΔR² | Mean ΔR² | Folds improved |
|---|---|---|---|---|---|---|---|
| Corr_filter_0.95 | -0.0008 | -0.0288 | -0.0240 | +0.0147 | +0.0351 | -0.0008 | 2 / 5 |
| Mutual_Info_top10 | +0.0116 | -0.0996 | -0.0043 | -0.0938 | -0.0343 | -0.0441 | 1 / 5 |
| Mutual_Info_top15 | -0.0664 | -0.0428 | -0.0380 | -0.0785 | -0.0848 | -0.0621 | 0 / 5 |
| Mutual_Info_top20 | -0.0586 | -0.0378 | -0.0164 | -0.0389 | +0.0501 | -0.0203 | 1 / 5 |
| F_test_top10 | +0.0426 | -0.1327 | -0.0665 | -0.1047 | -0.1067 | -0.0736 | 1 / 5 |
| F_test_top15 | -0.0097 | -0.0581 | -0.0378 | -0.0251 | -0.0846 | -0.0431 | 0 / 5 |
| F_test_top20 | +0.0116 | -0.0428 | -0.0478 | +0.0114 | +0.0168 | -0.0102 | 3 / 5 |
| Wrapper_RFECV | +0.0044 | -0.0161 | +0.0080 | -0.0460 | +0.0145 | -0.0070 | 3 / 5 |
| PCA_90%_var | -0.3024 | -0.0870 | -0.0340 | +0.0284 | -0.0855 | -0.0961 | 1 / 5 |
| PCA_95%_var | -0.2184 | -0.0927 | -0.0196 | +0.0219 | -0.0351 | -0.0688 | 1 / 5 |
| PCA_99%_var | -0.1511 | -0.0862 | -0.0199 | +0.0037 | -0.0531 | -0.0613 | 1 / 5 |

### Analysis of Variance
- The baseline fold-to-fold standard deviation is $\sigma_{fold} = 0.0627$.
- Correlation filtering (-0.0008), RFECV (-0.0070) and F-test top 20 (-0.0102) change $R^2$ by less than a quarter of $\sigma_{fold}$, with gains and losses mixed across folds. They are equivalent to the baseline, not better.
- PCA loses 0.069 (95% variance) on average and loses in 4 of 5 folds at every threshold, a consistent degradation.

---

## 4. Idealized vs. Realistic Breakdown

Each subset is modelled on its own with subset-restricted outer folds; `subset_idealized` is dropped because it is constant within a subset.

| Subset | Cases | Method | Mean R² | Std R² | Mean MAE (U/u*) | Fold R² values |
|---|---|---|---|---|---|---|
| Idealized | 190 | Baseline (33 features) | 0.9488 | 0.0234 | 0.0894 | [0.9663, 0.9340, 0.9799, 0.9137, 0.9500] |
| Idealized | 190 | Corr_filter_0.95 | 0.9474 | 0.0228 | 0.0932 | [0.9656, 0.9311, 0.9783, 0.9149, 0.9471] |
| Idealized | 190 | Mutual_Info_top15 | 0.9471 | 0.0221 | 0.0903 | [0.9666, 0.9444, 0.9765, 0.9161, 0.9321] |
| Idealized | 190 | F_test_top20 | 0.9451 | 0.0232 | 0.0938 | [0.9682, 0.9365, 0.9760, 0.9155, 0.9295] |
| Idealized | 190 | Wrapper_RFECV | 0.9497 | 0.0245 | 0.0882 | [0.9705, 0.9360, 0.9827, 0.9138, 0.9454] |
| Idealized | 190 | PCA_90%_var (5.0 components) | 0.9066 | 0.0313 | 0.1371 | [0.9599, 0.9044, 0.8918, 0.8643, 0.9125] |
| Idealized | 190 | PCA_95%_var (6.0 components) | 0.9306 | 0.0296 | 0.1157 | [0.9681, 0.9107, 0.9507, 0.8846, 0.9389] |
| Idealized | 190 | PCA_99%_var (9.0 components) | 0.9403 | 0.0169 | 0.1124 | [0.9648, 0.9301, 0.9529, 0.9167, 0.9371] |
| Realistic | 292 | Baseline (33 features) | 0.5544 | 0.1554 | 0.2465 | [0.6142, 0.5773, 0.7580, 0.5417, 0.2806] |
| Realistic | 292 | Corr_filter_0.95 | 0.5596 | 0.1580 | 0.2437 | [0.6250, 0.5624, 0.7838, 0.5309, 0.2961] |
| Realistic | 292 | Mutual_Info_top15 | 0.4298 | 0.1727 | 0.2802 | [0.4231, 0.5265, 0.5943, 0.5027, 0.1023] |
| Realistic | 292 | F_test_top20 | 0.5799 | 0.1431 | 0.2328 | [0.5436, 0.6294, 0.8133, 0.5396, 0.3737] |
| Realistic | 292 | Wrapper_RFECV | 0.5480 | 0.1684 | 0.2529 | [0.5525, 0.5811, 0.8013, 0.5332, 0.2717] |
| Realistic | 292 | PCA_90%_var (9.0 components) | 0.3987 | 0.1545 | 0.3188 | [0.4985, 0.2215, 0.6218, 0.4194, 0.2322] |
| Realistic | 292 | PCA_95%_var (11.8 components) | 0.4529 | 0.1798 | 0.2826 | [0.5846, 0.2846, 0.7220, 0.4275, 0.2457] |
| Realistic | 292 | PCA_99%_var (17.6 components) | 0.5059 | 0.1679 | 0.2687 | [0.6195, 0.6347, 0.6695, 0.3361, 0.2697] |

### Insights
1. **Regular arrays are easy:** for idealized cases every method, even PCA with 5 components, reaches $R^2 > 0.90$.
2. **Real neighbourhoods are hard on their own:** the realistic-only baseline reaches $R^2 = 0.5544$, lower than the realistic-case score of the models trained on both subsets in S06c (about 0.69). F-test top 20 does best here ($R^2 = 0.5799$), but the fold spread ($\pm 0.14$) is too large to call the difference real.

---

## 5. Answers to the Research Questions

### 1. Can filter methods reduce 34 features without materially hurting performance?
**Yes, modestly.**
- Correlation filtering at $\tau = 0.95$ keeps 27.8 features (folds: [28, 28, 28, 28, 27]) with no loss ($R^2 = 0.8162$ against $0.8170$).
- F-test top 20 keeps 20 features at a small cost ($R^2 = 0.8068$).
- Smaller filter subsets (10 or 15 features) lose 0.04 to 0.07.

### 2. Which features are selected most consistently?
From `feature_selection_stability.csv`, across the 5 outer folds:

- **Wrapper (RFECV), 16.6 features on average (folds: [15, 19, 18, 14, 17]):**
  - **Selected in 5/5 folds (8):** `d_over_h_kanda`, `dist_mean_m`, `lambda_f_dir`, `lateral_fetch_mean_m`, `upwind_fetch_mean_m`, `volume_per_area_m`, `z0_over_h_kanda`, `z0_over_h_macdonald`.
  - **Selected in 4/5 folds (8):** `dist_p90_m`, `downwind_fetch_mean_m`, `facet_orientation_entropy`, `lambda_f_y`, `max_h`, `mean_footprint_area_m2`, `std_h`, `windward_wall_share`.
  - **Selected in 3/5 folds (2):** `components_per_hectare`, `unobstructed_fraction_gt50m`.
  - **Selected in 2/5 folds (1):** `lambda_p`.
  - **Selected in 1/5 folds (3):** `lambda_w`, `mean_h`, `upwind_fetch_median_m`.
  - **Never selected (12):** `cv_h`, `d_over_h_macdonald`, `dist_median_m`, `forcing_dir_deg`, `gamma_h`, `lambda_f_x`, `max_to_mean_h`, `relative_wind_facet_angle_deg`, `skew_h`, `subset_idealized`, `wind_cos`, `wind_sin`.
- **F-test top 20:**
  - **Selected in 5/5 folds (18):** `components_per_hectare`, `d_over_h_kanda`, `d_over_h_macdonald`, `dist_mean_m`, `dist_median_m`, `dist_p90_m`, `downwind_fetch_mean_m`, `lambda_f_dir`, `lambda_f_x`, `lambda_f_y`, `lambda_p`, `lambda_w`, `lateral_fetch_mean_m`, `subset_idealized`, `unobstructed_fraction_gt50m`, `upwind_fetch_mean_m`, `upwind_fetch_median_m`, `volume_per_area_m`.
  - **Selected in 3/5 folds (3):** `facet_orientation_entropy`, `mean_h`, `windward_wall_share`.
  - **Selected in 1/5 folds (1):** `mean_footprint_area_m2`.
  - **Never selected (12):** `cv_h`, `forcing_dir_deg`, `gamma_h`, `max_h`, `max_to_mean_h`, `relative_wind_facet_angle_deg`, `skew_h`, `std_h`, `wind_cos`, `wind_sin`, `z0_over_h_kanda`, `z0_over_h_macdonald`.
- **Mutual information top 15:**
  - **Selected in 5/5 folds (10):** `d_over_h_macdonald`, `dist_mean_m`, `dist_median_m`, `dist_p90_m`, `downwind_fetch_mean_m`, `lambda_f_dir`, `lambda_p`, `lambda_w`, `unobstructed_fraction_gt50m`, `upwind_fetch_mean_m`.
  - **Selected in 4/5 folds (5):** `components_per_hectare`, `lambda_f_x`, `lambda_f_y`, `upwind_fetch_median_m`, `volume_per_area_m`.
  - **Selected in 3/5 folds (1):** `lateral_fetch_mean_m`.
  - **Selected in 2/5 folds (1):** `z0_over_h_macdonald`.
  - **Never selected (17):** `cv_h`, `d_over_h_kanda`, `facet_orientation_entropy`, `forcing_dir_deg`, `gamma_h`, `max_h`, `max_to_mean_h`, `mean_footprint_area_m2`, `mean_h`, `relative_wind_facet_angle_deg`, `skew_h`, `std_h`, `subset_idealized`, `wind_cos`, `wind_sin`, `windward_wall_share`, `z0_over_h_kanda`.
- **Correlation filter 0.95:**
  - **Selected in 5/5 folds (25):** `components_per_hectare`, `cv_h`, `d_over_h_kanda`, `d_over_h_macdonald`, `downwind_fetch_mean_m`, `facet_orientation_entropy`, `forcing_dir_deg`, `lambda_f_dir`, `lateral_fetch_mean_m`, `max_h`, `max_to_mean_h`, `mean_footprint_area_m2`, `mean_h`, `relative_wind_facet_angle_deg`, `skew_h`, `std_h`, `subset_idealized`, `upwind_fetch_mean_m`, `upwind_fetch_median_m`, `volume_per_area_m`, `wind_cos`, `wind_sin`, `windward_wall_share`, `z0_over_h_kanda`, `z0_over_h_macdonald`.
  - **Selected in 4/5 folds (2):** `dist_mean_m`, `dist_median_m`.
  - **Selected in 3/5 folds (1):** `lambda_w`.
  - **Selected in 1/5 folds (3):** `dist_p90_m`, `lambda_f_x`, `lambda_f_y`.
  - **Never selected (3):** `gamma_h`, `lambda_p`, `unobstructed_fraction_gt50m`.

### 3. Does the wrapper justify its extra cost?
**Only for compression.** RFECV took 63 s in total and keeps 16.6 features at $R^2 = 0.8099$, about the accuracy of the full set with half the inputs. It is useful where computing fetch and roughness features on new geometries is expensive. It does not improve accuracy.

### 4. Does PCA improve, reduce, or mainly shrink dimensionality?
**It shrinks dimensionality and reduces accuracy.** PCA keeps 8 to 18 components, and $R^2$ falls from $0.8170$ to $0.721$–$0.756$.

### 5. How many components retain 90%, 95% and 99% of the variance?
- **90%:** [8, 8, 9, 8, 8] components per fold.
- **95%:** [11, 11, 12, 11, 11] components per fold.
- **99%:** [18, 18, 18, 18, 18] components per fold.

### 6. Does selection behave differently for idealized and realistic cases?
**Yes.** Idealized cases reach $R^2 \approx 0.95$ with any subset. Realistic cases range from 0.40 to 0.58 depending on the method, with large fold-to-fold spread.

### 7. Are the consistently selected features physically interpretable?
**Yes.**
- `lambda_f_dir`: the frontal blockage facing the wind, which sets the drag.
- `upwind_fetch_mean_m` and `lateral_fetch_mean_m`: how far the wind travels unobstructed.
- `dist_mean_m`: how open the streets are.
- `volume_per_area_m`: the built mass.
- The Kanda and Macdonald roughness length and displacement height: the aerodynamic summary of the canopy.

### 8. Are differences meaningful relative to fold-to-fold variability?
- The selection methods differ from the baseline by less than a quarter of $\sigma_{fold} = 0.0627$ and in both directions across folds. They are equivalent, not better or worse.
- PCA's loss is consistent across folds.

### 9. Which method gives the best balance of performance, cost and interpretability?
- **The full 34-feature set is the accuracy choice.**
- **Correlation filtering** (about 28 features, no loss, negligible cost) is the cleanest reduction.
- **RFECV** (about 17 features, small loss) gives the most compression.
- PCA is not recommended for these tree models.

---

## 6. Generated Figures

Stored in `pipeline/S06b_feature_selection_AS/outputs/plots/`:
1. `1_r2_comparison.png`: cross-validation $R^2$ across methods with fold error bars.
2. `2_mae_rmse_comparison.png`: MAE and RMSE across methods.
3. `3_feature_counts.png`: retained features or components.
4. `4_selection_stability.png`: selection stability across folds.
5. `5_pca_variance.png`: scree plot and cumulative explained variance.
6. `6_subset_performance.png`: idealized vs. realistic performance.
