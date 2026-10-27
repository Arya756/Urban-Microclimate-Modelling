# S09 Phase 1 evaluation on the frozen test set

One-time scoring of the 53 cases in the 24 frozen test groups (34 idealized, 19 realistic) with the final Phase 1 models.

| | |
| --- | --- |
| Inputs | final model settings from S06c and S07; development cases only for fitting |
| Outputs | `outputs/test_case_level.csv`, `outputs/test_case_level_params.json`, `outputs/test_field_level.csv`, `outputs/test_field_per_case.csv`, `outputs/test_scored.json`; predictions in `results/preds/s09_*` (not tracked) |

## Protocol

- Every modelling choice was made on the development folds before this stage was run.
- The case-level models are re-tuned on all 482 development cases with the S06c search (Optuna, 30 trials, five grouped inner folds) and refitted on all of them.
- The pixel-level model and the hybrid are refitted with the S07 settings. The hybrid takes its level from the S09 XGBoost prediction.
- The test cases are scored once. `outputs/test_scored.json` records the date and commit, and `score_test_set.py` refuses to run while it exists.
- Two test groups come from cities that also have development neighbourhoods: AU-SURREYHILLS (Melbourne) and FI-MS (Helsinki). Every score is therefore also reported without them.

## Results

**Case level**, $U_\mathrm{mean}$ $R^2$:

| Model | All (53) | Idealized (34) | Realistic (19) | Without shared-city cases (51) |
| --- | --- | --- | --- | --- |
| SVR (RBF) | 0.884 | 0.976 | 0.737 | 0.871 |
| kNN | 0.878 | 0.957 | 0.745 | 0.867 |
| Random forest | 0.871 | 0.984 | 0.690 | 0.850 |
| LightGBM | 0.867 | 0.989 | 0.673 | 0.890 |
| Gaussian process | 0.860 | 0.990 | 0.654 | 0.877 |
| XGBoost | 0.848 | 0.990 | 0.623 | 0.847 |
| Ridge | 0.703 | 0.836 | 0.458 | 0.721 |
| B1: random forest, 9 basic features | 0.790 | 0.918 | 0.571 | 0.757 |

For $U_\mathrm{p95}$, XGBoost reaches 0.893 and LightGBM 0.869, against 0.836 for B1. All targets are in `test_case_level.csv`.

**Field level:**

| Prediction | Pooled $R^2$ | Pooled $R^2$ idealized / realistic | Median per-case $R^2$ |
| --- | --- | --- | --- |
| True case mean everywhere (oracle) | 0.377 | 0.415 / 0.365 | 0 |
| S09 case-level mean everywhere | 0.274 | 0.414 / 0.232 | -0.003 |
| Pixel-level LightGBM | **0.676** | 0.892 / 0.612 | 0.334 |
| Hybrid | 0.665 | 0.882 / 0.600 | **0.349** |

## Reading the test scores

1. **Generalisation holds.**
   - Every non-linear case-level model beats the baseline B1 on the unseen test groups, as on the development folds.
   - The pixel model and hybrid reproduce the development picture: the level is recovered, the pattern only partly.
2. **The ranking among non-linear models is not resolved by 53 cases.**
   - The test set contains only 19 realistic cases, and it is 64% idealized against 39% in development, so the all-case $R^2$ is higher than in development.
   - The order of models that differ by 0.01 to 0.04 changes with the case mix. On the test set SVR and kNN come first, on the development folds the tree ensembles do.
   - The development group bootstrap over 482 cases is the appropriate tool for ranking, and it finds the leading models statistically tied.
3. **Shared cities do not inflate the scores.** Removing the two test cases from cities with development neighbourhoods changes $R^2$ by -0.03 to +0.02.

## Reproduce

The scoring is deliberately one-time: `score_test_set.py` (about 25 minutes) writes `outputs/test_scored.json` before anything is scored and refuses to run again while it exists. A second scoring is possible only with `--rescore "<reason>"`, for a fix that changes no modelling choice; the file then records the reason and keeps every earlier scoring under `history`. Test-case targets are computed by this script through `urbanwind.data.load_case`; the S02 target table never contains them. The final scores are also logged as the S09 rows of `results/runs.csv`, written with `python -m urbanwind.evaluate <pred_dir> --stage S09 --final-test`. The `--final-test` flag lifts the evaluator's guard against test groups, and is meant only for this stage.
