# S06 Case-level models

Predicts four summary statistics of the pedestrian-level wind field of each case from the S05 morphology features:
- U_mean, U_p95 and U_std: the mean, 95th percentile and standard deviation of U/u* over open ground;
- TKE_mean: the mean of TKE/u*² over open ground.

| Part | Files | Status |
| --- | --- | --- |
| S06a | `run_s06.py`, `nested_tuning.py`, `S06a_case_models.ipynb`, files directly under `outputs/` | First model ladder and LightGBM gate model; superseded by S06c (see below) |
| S06b | `../S06b_feature_selection_AS/` | Feature selection and PCA experiments |
| S06c | `s06c_model_comparison.py`, `S06c_case_model_comparison.ipynb`, `outputs/s06c/` | Corrected rerun and comparison of eight regressors; current results |

## S06c protocol

- **Data.**
  - 482 development cases: 190 idealized and 292 realistic, in 93 groups (splits version 2).
  - Targets come from `pipeline/S02_preprocessing/outputs/case_targets.csv` and are normalised by each case's own u*. Open ground is taken from the building map.
  - The 34 S05 case features include the published Kanda (2013) parameterisation.
  - The frozen test groups are never loaded.
- **Outer loop.** The five fixed folds of `data/splits.json`. Each fold holds out whole cities and whole idealized layout families.
- **Inner loop.** 5-fold GroupKFold on the outer-training cases only, used for hyperparameter search with Optuna (TPE, 30 trials per model, fold and target).
  - XGBoost and LightGBM share one search space: tree count, learning rate, depth, minimum leaf size from 1 to 40, row and column subsampling, L2 penalty.
  - The Gaussian process (Matérn 5/2 with one length scale per feature) is fitted by marginal likelihood instead.
- **Scores.** R², MAE and RMSE on the pooled out-of-fold predictions, overall and per subset, plus the spread of R² across the five folds.
- **Model comparison.** Differences between models come from a group bootstrap: whole cities and layout families are resampled 2,000 times.

## S06c results

Scores below come from `outputs/s06c/` after the pipeline audit of 27 October 2026 (corrected features and targets, development groups merged by city in splits version 2).

### U_mean (out-of-fold, tuned)

| Model | R² | R² idealized | R² realistic | MAE (U/u*) | Fold R² |
| --- | --- | --- | --- | --- | --- |
| LightGBM | **0.876** | 0.961 | **0.787** | **0.143** | 0.865 ± 0.033 |
| Random forest | 0.828 | 0.958 | 0.694 | 0.158 | 0.829 ± 0.025 |
| Gaussian process | 0.816 | **0.976** | 0.655 | 0.164 | 0.780 ± 0.141 |
| XGBoost | 0.784 | 0.956 | 0.608 | 0.163 | 0.792 ± 0.051 |
| SVR (RBF) | 0.717 | 0.958 | 0.472 | 0.187 | 0.711 ± 0.065 |
| kNN | 0.655 | 0.930 | 0.373 | 0.219 | 0.669 ± 0.065 |
| Ridge | 0.197 | 0.864 | -0.480 | 0.344 | -0.004 ± 0.936 |
| Training mean | -0.011 | -0.115 | -0.071 | 0.470 | |
| B1: random forest, 9 basic features, untuned | 0.742 | 0.881 | 0.587 | 0.230 | 0.730 ± 0.025 |

Fold R² is the mean ± sample standard deviation over the five folds.

### All targets (R², tuned, full feature set)

| Model | U_mean | U_p95 | U_std | TKE_mean |
| --- | --- | --- | --- | --- |
| LightGBM | 0.876 | 0.767 | 0.694 | 0.925 |
| Random forest | 0.828 | 0.725 | 0.683 | 0.898 |
| Gaussian process | 0.816 | 0.710 | 0.685 | 0.947 |
| XGBoost | 0.784 | 0.723 | 0.741 | 0.904 |
| B1 | 0.742 | 0.660 | 0.584 | 0.860 |

### Findings

1. **LightGBM leads.**
   - It beats B1 on U_mean by 0.135 R² [0.095, 0.178] (95% group-bootstrap interval), and the random forest beats B1 by 0.086 [0.039, 0.137].
   - The gains of tuned XGBoost (0.042 [-0.028, 0.113]) and the Gaussian process (0.075 [-0.042, 0.157]) are not significant.
   - LightGBM beats the random forest by 0.048 [0.005, 0.099] and XGBoost by 0.092 [0.023, 0.167].
2. **Tuning can help or hurt.**
   - With default settings the random forest scores 0.831, LightGBM 0.829 and XGBoost 0.851, and default XGBoost and LightGBM are not significantly different (0.022 [-0.031, 0.064]).
   - Tuning raises LightGBM to 0.876 but lowers XGBoost to 0.784. The loss lies almost entirely in one fold (R² 0.727 tuned against 0.874 with defaults), where the chosen setting strongly underpredicts two very open neighbourhoods, CA-Sunset and US-WestPhoenix, whose cities are held out with them.
   - Hyperparameters chosen on the training cities need not carry over to unseen cities. Ridge regression is the least stable model: one fold scores -1.48.
3. **Real cities are the hard part.**
   - Idealized arrays are predicted almost perfectly (R² about 0.96).
   - Realistic neighbourhoods reach 0.79 for U_mean and 0.57 for U_p95 with LightGBM.
   - Peak winds depend on local street geometry that case-level summaries do not describe; this is the gap S07 measures.
4. **Physics features help most.** With the 9 basic features the tuned ensembles reach 0.72 to 0.74 (realistic 0.51 to 0.55). Adding fetch, distance-to-wall, displacement-height and roughness features raises LightGBM to 0.876 and the random forest to 0.828; the 8 morphology features alone change little.
5. **What drives mean wind.** TreeSHAP for LightGBM on the held-out folds (`outputs/s06c/figures/shap_u_mean.png`) ranks these features highest:
   - mean and 90th-percentile distance to the nearest wall (more open space, more wind);
   - frontal area density λf (more frontal area, less wind);
   - mean upwind and lateral fetch;
   - built volume per unit area and λp.
6. **Random splits flatter models.** Scored on realistic cases, random 5-fold CV overstates R² by 0.00 for the random forest, 0.04 for LightGBM and kNN, and 0.61 for ridge, compared with folds that hold out whole cities (`leakage_random_vs_grouped.png`).
7. **No transfer between subsets.** Untuned LightGBM trained only on idealized arrays reaches R² 0.12 on real cities (0.70 when real cities are in training); trained only on real cities, it reaches -0.51 on idealized arrays (0.95). Both subsets must be in the training data.
8. **Flagged cases do not drive the results.** Leaving out the six flagged cases changes R² by at most 0.007 (`r2_without_flagged` in `model_comparison.csv`).

## Why S06a is superseded

S06a (first run) predicted the same statistics in m/s:
- they were taken from the S04 master table, with open ground defined by NaN in the wind files;
- for 59 development cases that biases the mean wind by more than 1% (up to 14.6%) compared with the raster mask used now;
- its Kanda (2013) features used a form that is not the published one;
- its first runs tuned hyperparameters on the outer validation folds.

`nested_tuning.py` (added later) fixed the tuning protocol. Its numbers, and the S06a gate report (LightGBM U_mean R² 0.828), describe that first setup and are kept for reference.

## Case readiness and exclusions

- **Excluded (3 cases):** CN-BE-V2_d00, CN-CD-V1_d00 and CN-SH-V1_d00. Their downloadable building maps do not match the simulated domain.
- **Kept and flagged (6 cases):**
  - JP-Tok-V4_d00 stopped at 4.9 h. Its time series is stationary (final TKE coefficient of variation 2%), but its fields average 0.9 h instead of 8 h.
  - UG-UG-U1/U2/U3 and AU-Mel-U9_d00: after the vertical flip, the building map and wind field agree on 90% to 97% of cells; the rest are building-edge interpolation artefacts.
  - NP-NP-U1_d00 stores values inside 3,120 building cells, which the building-map mask removes.
- All rules live in `urbanwind.data` (`EXCLUDE`, `FLAG`, `load_case`).

## Files

```text
pipeline/S06_case_models/
├── s06c_model_comparison.py           # S06c experiment (about 20 minutes on 10 cores)
├── S06c_case_model_comparison.ipynb   # S06c results with commentary
├── run_s06.py, nested_tuning.py       # S06a scripts
├── S06a_case_models.ipynb             # S06a notebook
└── outputs/
    ├── s06c/
    │   ├── model_comparison.csv       # every model, feature set and target; overall, per subset, per fold
    │   ├── oof_predictions.csv        # out-of-fold predictions behind every row above
    │   ├── best_params.json           # tuned hyperparameters per model, target and outer fold
    │   ├── bootstrap_r2_differences.csv
    │   ├── leakage_random_vs_grouped.csv
    │   ├── transfer.csv
    │   ├── shap_importance_u_mean.csv
    │   └── figures/                   # model ladder, all targets, ablation, leakage, parity, SHAP
    └── (S06a reports, tables and figures)

results/preds/s06c_<model>/case_preds.csv   # contract-format predictions, scored into results/runs.csv
```

## Reproduce

```bash
uv sync
uv run python pipeline/S06_case_models/s06c_model_comparison.py
for m in mean ridge knn svr gaussian_process random_forest xgboost lightgbm; do
  uv run python -m urbanwind.evaluate results/preds/s06c_$m --stage S06c
done
```
