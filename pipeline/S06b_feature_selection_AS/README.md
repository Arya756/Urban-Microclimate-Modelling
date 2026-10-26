# Stage S06b: Feature Selection and PCA Experiments

**Stage:** S06b (Isolated Sub-experiment)  
**Repository:** [Urban-Microclimate-Modelling](https://github.com/mahir-m01/Urban-Microclimate-Modelling)  
**Branch:** `feature/s06b-feature-selection-pca`  

---

## 📖 Overview for Beginners

In Stage S06, our team built a LightGBM machine learning model to predict the mean pedestrian wind speed of each case (`U_mean`, in units of the case's friction velocity u*). That model used **34 physics-informed features** (such as building height, street canyon width, wind direction, and aerodynamic roughness parameters) and achieves an $R^2$ score of approximately **0.82**.

> **Update.** All results below come from a rerun on the corrected targets (U/u* with each case's own u*, open ground taken from the building map) and the S05 features with the published Kanda (2013) parameterisation. The conclusions changed: no selection method now beats the full feature set.

However, having 34 features introduces two main challenges:
1. **Redundancy and Multicollinearity:** Several features measure almost the exact same physical phenomenon. For example, building height variation $CV(h)$ and $\gamma_h$ have a correlation of $1.000$ (perfect collinearity), and aerodynamic displacement height $d/h$ correlates $>0.98$ with building density $\lambda_p$.
2. **Computational and Extraction Cost:** Extracting 34 complex 3D spatial features on large city grids requires significant compute. If we can achieve the same accuracy with 15 or 20 features, future microclimate simulations will run much faster.

This **S06b experiment** explores three standard machine learning techniques to see if we can safely prune these 34 features down without hurting predictive performance.

---

## 🔍 The Three Approaches Explained

### 1. Filter Methods (Fast & Independent of the Model)
Filter methods evaluate each feature using statistical properties before any machine learning model is trained:
- **Correlation Filtering:** If two features correlate above 0.95, they carry redundant information. We drop the one that correlates less with the target wind speed.
- **Mutual Information (MI):** Measures how much knowing a feature reduces uncertainty about the target wind speed, capturing non-linear relationships.
- **F-test (Univariate Regression):** Uses linear statistical hypothesis testing to rank features based on the strength of their relationship with the target.

### 2. Wrapper Methods (Model-Guided Search)
Instead of looking at features in isolation, wrapper methods use the machine learning model itself to test subsets of features:
- **RFECV (Recursive Feature Elimination with Cross-Validation):** Evaluates feature subsets by iteratively eliminating the least important features. In our implementation:
  - **Feature Selection Phase:** Uses a lightweight 100-tree LightGBM estimator (`n_estimators=100`, `learning_rate=0.05`) with group-aware inner cross-validation (`GroupKFold(n_splits=3)`) strictly on outer training data to find the optimal feature subset.
  - **Evaluation Phase:** Trains the standardized 400-tree baseline LightGBM model (`n_estimators=400`, `learning_rate=0.05`) on the selected features and evaluates on the untouched outer validation fold.
- *Pros:* Captures complex non-linear interactions between features and automatically prunes redundant variables.
- *Cons:* Slower because it evaluates models iteratively across inner CV splits.

### 3. Dimensionality Reduction (PCA)
- **Principal Component Analysis (PCA):** Rather than selecting a subset of original features, PCA rotates and combines all 34 features into new, uncorrelated mathematical axes called "principal components".
- We keep only enough components to preserve 90%, 95%, or 99% of the total dataset variance.
- *Important Distinction:* PCA does **not** select features; it creates composite features. As a result, physical interpretability is lost.

---

## ⚖️ How S06b Differs from the S06 Ablation

In Stage S06 (step 3), the team performed a **domain-expert feature ablation**:
- **Basic (9 features):** Simple building footprint and wind angle.
- **Morphology (17 features):** Basic features + building shape and height distributions.
- **Physics (34 features):** Morphology + aerodynamic formulas and directional fetch distances.

In contrast, **S06b does algorithmic feature selection** starting from the full 34 Physics features. We ask: *Can an algorithm discover an even leaner, better subset of these 34 features without needing manual human groupings?*

---

## 🛡️ Preventing Data Leakage with Grouped Cross-Validation

In urban climate datasets, multiple simulation cases often share the same underlying urban geometry (for example, the same street layout tested at different wind angles, $0^\circ$ vs $45^\circ$). 

If we used standard random K-fold cross-validation:
- Case `A_angle_0` might end up in the training fold while `A_angle_45` ends up in the validation fold.
- The model would simply memorize the building layout of geometry `A`, causing artificial inflation of test scores (**data leakage**).

**How S06b ensures zero leakage:**
1. **Frozen Test Exclusion:** All 24 frozen test groups in `data/splits.json` were strictly excluded before running any models.
2. **Outer GroupKFold:** All 482 development cases were split strictly by `split_group`. No urban layout appears in both training and validation sets of any fold.
3. **Training-Only Feature Selection:** All statistics (correlation, Mutual Information, F-test, RFECV inner folds, StandardScaler, and PCA) were fitted **strictly on the training fold**. The validation fold is kept completely untouched until final evaluation.

---

## 🚀 How to Run

All scripts can be executed directly from the repository root.

### 1. Run Unit Tests
Verifies leakage prevention, baseline reproduction, and group-aware CV logic:
```bash
PYTHONPATH=. pytest pipeline/S06b_feature_selection_AS/tests/test_feature_selection.py
```

### 2. Run a Quick Smoke Test (2 folds, coarse step)
Runs a fast diagnostic sanity check in ~15 seconds:
```bash
PYTHONPATH=. python3 pipeline/S06b_feature_selection_AS/run_feature_selection.py --smoke
PYTHONPATH=. python3 pipeline/S06b_feature_selection_AS/run_pca.py --smoke
PYTHONPATH=. python3 pipeline/S06b_feature_selection_AS/plot_and_consolidate.py
```

### 3. Run the Full Experiment (5 outer folds)
Runs the full 5-fold cross-validation experiments and generates all tables and plots:
```bash
# Step 1: Run filter and wrapper feature selection
PYTHONPATH=. python3 pipeline/S06b_feature_selection_AS/run_feature_selection.py

# Step 2: Run PCA dimensionality reduction
PYTHONPATH=. python3 pipeline/S06b_feature_selection_AS/run_pca.py

# Step 3: Consolidate master tables and generate plots
PYTHONPATH=. python3 pipeline/S06b_feature_selection_AS/plot_and_consolidate.py
```

---

## 📂 Folder Structure and Generated Outputs

All code and experiment artifacts are completely contained inside `pipeline/S06b_feature_selection_AS/`:

```
pipeline/S06b_feature_selection_AS/
├── README.md                      # This documentation guide
├── config.yaml                    # Experiment parameters, seeds, and paths
├── feature_selection_utils.py     # Leakage-free feature selection and evaluation utility library
├── run_feature_selection.py       # Main pipeline for Filter & Wrapper methods
├── run_pca.py                     # Main pipeline for PCA experiments
├── plot_and_consolidate.py        # Generates unified comparison tables and 6 figures
├── tests/
│   └── test_feature_selection.py  # Comprehensive test suite (8 pytest tests)
└── outputs/
    ├── comparison.csv             # Master comparison table across all 11 methods
    ├── fold_metrics.csv           # Detailed metrics (R², MAE, RMSE) per fold and method
    ├── paired_fold_comparisons.csv# Fold-by-fold paired differences (Method R² - Baseline R²)
    ├── selected_features.csv      # Exact list of features selected in each fold
    ├── feature_selection_stability.csv # Selection frequency across outer folds
    ├── pca_explained_variance.csv # Individual and cumulative variance per principal component
    ├── pca_component_loadings.csv # Eigenvector loadings linking features to components
    ├── oof_predictions.csv        # Out-of-fold prediction values for all 482 cases
    ├── subset_metrics.csv         # Performance breakdown on Idealized vs Realistic subsets
    ├── experiment_report.md       # Comprehensive scientific report answering all key questions
    └── plots/
        ├── 1_r2_comparison.png         # Cross-validation R² with error bars across all methods
        ├── 2_mae_rmse_comparison.png   # MAE and RMSE across all methods
        ├── 3_feature_counts.png        # Retained feature/component counts
        ├── 4_selection_stability.png   # Feature selection stability across folds
        ├── 5_pca_variance.png          # PCA scree plot and cumulative variance
        └── 6_subset_performance.png    # Idealized vs Realistic subset comparison
```

---

## 📊 Summary of Results

### Overall Performance on Target `U_mean` (5 Outer Folds)

| Method | Approach | Features / Components | Mean $R^2 \pm \text{std}$ | MAE (U/u*) | RMSE (U/u*) |
|---|---|---|---|---|---|
| Baseline | Full Set | 34.0 | $0.8170 \pm 0.0627$ | 0.1695 | 0.2688 |
| Corr_filter_0.95 | Filter | 27.8 | $0.8162 \pm 0.0503$ | 0.1742 | 0.2721 |
| Mutual_Info_top20 | Filter | 20.0 | $0.7966 \pm 0.0732$ | 0.1814 | 0.2844 |
| F_test_top20 | Filter | 20.0 | $0.8068 \pm 0.0458$ | 0.1827 | 0.2797 |
| Wrapper_RFECV | Wrapper | 16.6 | $0.8099 \pm 0.0792$ | 0.1725 | 0.2718 |
| PCA_90%_var | PCA | 8.2 | $0.7208 \pm 0.1164$ | 0.2190 | 0.3327 |
| PCA_95%_var | PCA | 11.2 | $0.7482 \pm 0.0928$ | 0.2076 | 0.3169 |
| PCA_99%_var | PCA | 18.0 | $0.7557 \pm 0.0752$ | 0.2016 | 0.3127 |

---

## 🎯 Final Conclusions and Recommendations

1. **Keep the full feature set for accuracy.** No selection method beats the 34-feature baseline ($R^2 = 0.8170 \pm 0.0627$). The best reductions are statistically equivalent to it: their mean paired differences are under a quarter of the fold-to-fold spread, with gains and losses mixed across folds.

2. **Cleanest reduction: correlation filtering.**
   - It removes about 6 collinear features (27.8 kept) with no loss ($R^2 = 0.8162 \pm 0.0503$).
   - It takes milliseconds, and the kept features stay physically interpretable.

3. **Most compression: wrapper RFECV.**
   - It keeps 16.6 features on average, about half, at $R^2 = 0.8099$.
   - It selects these in every fold:
     - directional frontal area (`lambda_f_dir`);
     - upwind and lateral fetch;
     - mean distance to walls;
     - built volume;
     - the Kanda displacement height and roughness length;
     - the Macdonald roughness length.

4. **F-test top 20** reaches $R^2 = 0.8068 \pm 0.0458$. In the first run (old targets) it edged the baseline; on the corrected data it does not.

5. **Avoid PCA for these tree models.** Keeping 90 to 99% of the variance with 8 to 18 components lowers $R^2$ to 0.721–0.756, because linear projections blur the physical thresholds the trees split on.

For the full detailed scientific findings and answers to the research questions, see [`outputs/experiment_report.md`](outputs/experiment_report.md).
