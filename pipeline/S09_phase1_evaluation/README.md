# S09 Phase 1 evaluation on the frozen test set

Scoring of the 53 cases in the 24 frozen test groups (34 idealized, 19 realistic) with the final Phase 1 models.

| | |
| --- | --- |
| Inputs | final model settings from S06c and S07; development cases only for fitting |
| Outputs | `outputs/test_case_level.csv`, `outputs/test_case_level_params.json`, `outputs/test_field_level.csv`, `outputs/test_field_per_case.csv`, `outputs/test_scored.json`; predictions in `results/preds/s09_*` (not tracked) |

## Protocol

- Every modelling choice was made on the development folds before this stage was run.
- The case-level models are re-tuned on all 482 development cases with the S06c search (Optuna, 30 trials, five grouped inner folds) and refitted on all of them.
- The pixel-level model and the hybrid are refitted with the S07 settings. The hybrid takes its level from the S09 prediction of the case-level model chosen in S07 on the development folds (`level_model` in `pipeline/S07_pixel_model/outputs/run_info.json`, LightGBM).
- The test set was scored on 26 October 2026 and, after the pipeline audit of 27 October 2026 corrected the features, targets and development groups, a second time on 27 October 2026. The models were re-tuned and re-selected on the development folds; no test score was used for any choice. `outputs/test_scored.json` records both runs and the reason.
- Two test groups come from cities that also have development neighbourhoods: AU-SURREYHILLS (Melbourne) and FI-MS (Helsinki). Every score is therefore also reported without them.

## Results

**Case level**, $U_\mathrm{mean}$ $R^2$:

| Model | All (53) | Idealized (34) | Realistic (19) | Without shared-city cases (51) |
| --- | --- | --- | --- | --- |
| Gaussian process | 0.931 | 0.985 | 0.843 | 0.946 |
| XGBoost | 0.898 | 0.975 | 0.772 | 0.892 |
| Random forest | 0.883 | 0.971 | 0.740 | 0.861 |
| kNN | 0.870 | 0.959 | 0.723 | 0.843 |
| LightGBM | 0.865 | 0.974 | 0.690 | 0.875 |
| SVR (RBF) | 0.857 | 0.974 | 0.669 | 0.865 |
| Ridge | 0.724 | 0.840 | 0.507 | 0.747 |
| B1: random forest, 9 basic features | 0.827 | 0.919 | 0.666 | 0.804 |

For $U_\mathrm{p95}$ the margins over B1 are small (XGBoost 0.858, LightGBM 0.850, against 0.843 for B1) and the Gaussian process falls below it (0.792). All targets are in `test_case_level.csv`.

**Field level:**

| Prediction | Pooled $R^2$ | Pooled $R^2$ idealized / realistic | Median per-case $R^2$ |
| --- | --- | --- | --- |
| True case mean everywhere (oracle) | 0.378 | 0.415 / 0.367 | 0 |
| S09 case-level mean everywhere | 0.295 | 0.412 / 0.260 | -0.007 |
| Pixel-level LightGBM | 0.680 | **0.893** / 0.617 | 0.351 |
| Hybrid | **0.685** | 0.876 / **0.628** | **0.366** |

## Reading the test scores

1. **Generalisation holds.**
   - For the mean wind, every non-linear case-level model beats the baseline B1 on the unseen test groups, including kNN and SVR, which were below it on the development folds.
   - The pixel model and hybrid reproduce the development picture: the level is recovered, the pattern only partly.
2. **The ranking among non-linear models is not resolved by 53 cases.**
   - The test set contains only 19 realistic cases, and it is 64% idealized against 39% in development, so the all-case $R^2$ is higher than in development.
   - The order of the leading models changes with the case mix: on the test set the Gaussian process comes first and LightGBM fifth; on the development folds LightGBM leads.
   - The development comparison over 482 cases, with its group bootstrap, is the appropriate evidence for ranking.
3. **Shared cities do not inflate the scores.** Removing the two test cases from cities with development neighbourhoods changes case-level $R^2$ by -0.03 to +0.02 and leaves the median per-case field $R^2$ unchanged; pooled field $R^2$ falls by 0.06 to 0.08 for every field model, the oracle included, which reflects the changed case mix.

## Reproduce

The scoring is deliberately one-time: `score_test_set.py` (about 25 minutes) writes `outputs/test_scored.json` before anything is scored and refuses to run again while it exists. A second scoring is possible only with `--rescore "<reason>"`, for a fix that changes no modelling choice; the file then records the reason and keeps every earlier scoring under `history`. Test-case targets are computed by this script through `urbanwind.data.load_case`; the S02 target table never contains them. The final scores are also logged as the S09 rows of `results/runs.csv`, written with `python -m urbanwind.evaluate <pred_dir> --stage S09 --final-test`. The `--final-test` flag lifts the evaluator's guard against test groups, and is meant only for this stage.
