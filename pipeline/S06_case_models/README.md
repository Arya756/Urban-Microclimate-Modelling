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
  - 482 development cases: 190 idealized and 292 realistic, in 136 groups.
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

### U_mean (out-of-fold, tuned)

| Model | R² | R² idealized | R² realistic | MAE (U/u*) | Fold R² |
| --- | --- | --- | --- | --- | --- |
| XGBoost | **0.836** | 0.970 | **0.700** | 0.161 | 0.836 ± 0.053 |
| LightGBM | 0.828 | 0.967 | 0.686 | 0.164 | 0.828 ± 0.076 |
| Gaussian process | 0.819 | **0.982** | 0.656 | **0.159** | 0.820 ± 0.100 |
| Random forest | 0.817 | 0.966 | 0.666 | 0.173 | 0.817 ± 0.052 |
| kNN | 0.690 | 0.935 | 0.439 | 0.212 | 0.700 ± 0.085 |
| SVR (RBF) | 0.617 | 0.958 | 0.275 | 0.180 | 0.552 ± 0.411 |
| Ridge | 0.383 | 0.885 | -0.128 | 0.314 | 0.367 ± 0.249 |
| Training mean | -0.004 | -0.107 | -0.061 | 0.467 | |
| B1: random forest, 9 basic features, untuned | 0.735 | 0.890 | 0.565 | 0.233 | 0.730 ± 0.100 |

### All targets (R², tuned, full feature set)

| Model | U_mean | U_p95 | U_std | TKE_mean |
| --- | --- | --- | --- | --- |
| XGBoost | 0.836 | 0.754 | 0.724 | 0.916 |
| LightGBM | 0.828 | 0.751 | 0.696 | 0.921 |
| Random forest | 0.817 | 0.749 | 0.721 | 0.911 |
| Gaussian process | 0.819 | 0.666 | 0.683 | 0.934 |
| B1 | 0.735 | 0.670 | 0.595 | 0.865 |

### Findings

1. **Tree ensembles lead, and they are tied.**
   - XGBoost, LightGBM and the random forest beat B1 on U_mean by 0.08 to 0.10 R². The 95% bootstrap intervals of each difference exclude zero.
   - The differences among the three are within resampling noise: XGBoost minus LightGBM is 0.008 [-0.019, 0.032], and XGBoost minus the random forest is 0.019 [-0.019, 0.057].
   - On U_p95 all three beat B1 by about 0.08 (intervals exclude zero).
2. **Tuning adds little.** With fixed default settings the random forest scores 0.815, LightGBM 0.827 and XGBoost 0.811 on U_mean; only XGBoost gains noticeably from tuning.
   - An earlier run, with 3 inner folds and a narrower leaf-size range for LightGBM than for XGBoost, made LightGBM look 0.06 worse. Matching the search spaces removed the gap, so the comparison is sensitive to how each model is tuned.
   - Kernel and linear models are the least stable. SVR scores 0.62 to 0.84 on four folds but -0.26 on the fifth, because the hyperparameters chosen on the training folds did not carry over to the held-out cities.
3. **Real cities are the hard part.**
   - Idealized arrays are predicted almost perfectly (R² 0.97).
   - Realistic neighbourhoods reach 0.70 for U_mean and 0.55 for U_p95.
   - Peak winds depend on local street geometry that case-level summaries do not describe; this is the gap S07 measures.
4. **Physics features help most on real cities.**
   - Adding fetch, distance-to-wall, displacement-height and roughness features raises U_mean R² from 0.71–0.77 (basic or morphology features) to 0.82–0.84.
   - On realistic cases R² rises from 0.52–0.61 to 0.67–0.70.
   - The 8 extra morphology features add between 0 and 0.05 over the 9 basic ones.
5. **What drives mean wind.** TreeSHAP on the held-out folds (`outputs/s06c/figures/shap_u_mean.png`) ranks these features highest:
   - frontal area density λf (more frontal area, less wind);
   - mean and 90th-percentile distance to the nearest wall (more open space, more wind);
   - mean upwind fetch;
   - built volume per unit area;
   - lateral fetch;
   - the Kanda roughness length.
6. **Random splits flatter models.** Scored on realistic cases, random 5-fold CV overstates R² by 0.03 for the random forest, 0.04 for LightGBM and 0.45 for ridge, compared with folds that hold out whole cities (`leakage_random_vs_grouped.png`).
7. **No transfer between subsets.** A model trained only on idealized arrays reaches R² 0.09 on real cities; trained only on real cities, it reaches -0.47 on idealized arrays. Both subsets must be in the training data.
8. **Flagged cases do not drive the results.** Leaving out the six flagged cases changes R² by less than 0.01 (`r2_without_flagged` in `model_comparison.csv`).

## Why S06a is superseded

S06a (first run) predicted the same statistics in m/s:
- they were taken from the S04 master table, with open ground defined by NaN in the wind files;
- for 52 cases that biased the mean wind by more than 1% (up to 11.7%);
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
