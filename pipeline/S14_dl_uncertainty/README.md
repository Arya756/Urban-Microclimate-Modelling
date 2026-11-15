# S14 Uncertainty and calibration

Gives every predicted wind map of the final S13 model an uncertainty map, and turns it into intervals whose coverage holds on cities and layout families the model has never seen.

| | |
| --- | --- |
| Inputs | the out-of-fold predictions of the three seeds of the 200-epoch U-Net (`results/preds/s13_unet_tuned_e200_seed{0,1,2}/`, not tracked); for MC dropout, the seed-0 weights (`results/checkpoints/s13_unet_tuned_e200_seed0/`) and the S10 cache; the building rasters (`data/raw/`) for the distance to the nearest building |
| Outputs | `outputs/coverage_summary.csv`, `outputs/reliability.csv`, `outputs/coverage_by_distance.csv`, `outputs/per_case_coverage_<method>.csv`, `outputs/spearman_selective.csv`; `figures/`; `results/preds/s14_<method>/<case>.npz` with `lo` and `hi` (not tracked) |
| Code | `src/urbanwind/dl/uq.py`, `run_s14_pipeline.py`; tests in `src/urbanwind/dl/tests/test_uq.py` |

## Method

**Uncertainty maps.**
- **Deep ensemble.** The three training seeds of the final model. The prediction is their mean and the uncertainty sigma is their standard deviation, cell by cell. Both come straight from the saved, scored predictions, so no network is run again.
- **MC dropout.** The seed-0 network of each fold with only its dropout layers active, 30 passes per case, so every case is still predicted out of fold. The passes run through the project's own prediction code: mirror averaging, bilinear return to 0.5 m grids, and the building mask from the cache. GroupNorm keeps no running statistics, so switching dropout on changes nothing else.

**Intervals (90%).** Split conformal on whole groups. The quantile q for the cases of outer fold k comes only from the cells of the other four folds, so a city or layout family never calibrates its own intervals.

| Interval | Conformity score | Interval |
| --- | --- | --- |
| `constant` | \|y − pred\| | pred ± q, the same width everywhere |
| `normalised` | \|y − pred\| / (sigma + 0.05) | pred ± q (sigma + 0.05), wider where the members disagree |
| `normalised_by_subset` (main) | the same | the same, with idealized and realistic cases calibrated separately, as their error scales differ |

**Checks.**
- Coverage overall, by subset, per case and by distance to the nearest building. The distance is measured from the building raster on each case's own grid (0.5 m or 1 m), wrapping round the periodic domain.
- An out-of-fold reliability diagram: nominal 50% to 95%, each calibrated on the other folds.
- The Spearman correlation between sigma and the absolute error: pooled, over 2,000,000 random cells, and within each case.
- Selective prediction: the MAE after dropping the 10%, 20% and 30% most uncertain cells.

## Results: deep ensemble

All 482 development cases and 87,788,708 open-ground cells (`outputs/coverage_summary.csv`).

| 90% interval | Coverage, all | Idealized | Realistic | Mean width, idealized (u*) | Mean width, realistic (u*) | Cases with coverage in 85% to 95% |
| --- | --- | --- | --- | --- | --- | --- |
| `constant` | 0.8991 | 0.9972 | 0.8799 | 2.4347 | 2.4354 | 43% |
| `normalised` | 0.8992 | 0.9839 | 0.8826 | 1.5477 | 2.8470 | 58% |
| `normalised_by_subset` | 0.8995 | 0.9002 | 0.8994 | 0.8568 | 3.0579 | 63% |

- **All three reach 90% overall,** as split conformal guarantees, but only the main interval holds 90% in each subset. The constant width over-covers idealized cases (99.7%) with intervals almost three times wider than they need to be, and under-covers realistic ones (88.0%).
- **Per case,** the main interval's coverage has a median of 0.908, with the 10th to 90th percentiles at 0.764 to 0.955 (`outputs/per_case_coverage_ensemble.csv`). Cells within a case are strongly correlated, so a whole case can sit above or below its group's quantile. The lowest single cases cover 31% (realistic) and 41% (idealized).
- **Reliability.** Calibrated on the other folds, the coverage matches the nominal level from 50% to 95% within 0.002 for the main interval (`outputs/reliability.csv`, `figures/reliability_diagram.png`).
- **By distance to the nearest building** (`outputs/coverage_by_distance.csv`), the main interval covers 89.4% to 90.5% from 1 m to 32 m. It covers 87.7% beyond 32 m and 78.7% within 1 m of a wall, where few cells (0.35% of the total) and the sharpest gradients are.

| Uncertainty against error (`outputs/spearman_selective.csv`) | Ensemble |
| --- | --- |
| Spearman, pooled over 2,000,000 random cells | 0.2956 |
| Spearman, median within a case | 0.1233 |
| MAE of all cells (u*) | 0.5378 |
| MAE after dropping the 10% most uncertain cells | 0.5065 |
| MAE after dropping 20% | 0.4861 |
| MAE after dropping 30% | 0.4664 |

- The ensemble spread is larger where the error is larger across the dataset (0.30 pooled, just below the 0.3 target), but only weakly within a case (median 0.12). It tells fast, complex regions from calm ones better than it ranks the cells inside one neighbourhood.
- Dropping the most uncertain cells lowers the error steadily: by 6% at 10% dropped, and by 13% at 30%.

## Results: MC dropout

The same analysis with the MC-dropout spread (method `mc_dropout` in every output file). Its prediction is the mean of the 30 passes of one network, so it is less accurate than the three-seed mean (MAE over all cells 0.5538 against 0.5378 u*).

| 90% interval, MC dropout | Coverage, all | Idealized | Realistic | Mean width, idealized (u*) | Mean width, realistic (u*) | Cases with coverage in 85% to 95% |
| --- | --- | --- | --- | --- | --- | --- |
| `constant` | 0.8991 | 0.9972 | 0.8799 | 2.4987 | 2.5004 | 43% |
| `normalised` | 0.8970 | 0.9955 | 0.8778 | 2.1334 | 2.4733 | 38% |
| `normalised_by_subset` | 0.8974 | 0.8987 | 0.8972 | 0.8839 | 2.6486 | 44% |

| Uncertainty against error | Ensemble | MC dropout |
| --- | --- | --- |
| Spearman, pooled over 2,000,000 random cells | 0.2956 | 0.2583 |
| Spearman, median within a case | 0.1233 | 0.1679 |
| MAE after dropping the 10% / 20% / 30% most uncertain cells (u*) | 0.5065 / 0.4861 / 0.4664 | 0.5215 / 0.5032 / 0.4869 |

- **Coverage holds for both,** because the conformal step sets the width. The main MC-dropout interval covers 0.8974 overall and 0.8987 / 0.8972 in the two subsets, and stays within 0.003 of nominal from 50% to 95% (`outputs/reliability.csv`).
- **The ensemble spread is the better uncertainty map across the dataset.** It ranks errors better over all cells (0.30 against 0.26), and its per-case coverage is more even: 63% of cases fall in 85% to 95% against 44%, and its 10th percentile is 0.764 against 0.665 (`outputs/coverage_summary.csv`).
- **MC dropout ranks errors slightly better within a case** (median 0.17 against 0.12) and its realistic intervals are narrower (2.65 against 3.06 u* on average), at the cost of a less accurate prediction and less even coverage.
- **Cost.** The ensemble reuses the three trained seeds. MC dropout needs 30 network passes per case (about 55 minutes for all 482 cases on an Apple M3 Pro GPU).

## Reproducibility

The analysis was run twice, on two machines, each rebuilding the seed predictions from the saved weights. The ensemble results agree to within 0.0003 in every per-case value and reliability point, and the MC-dropout results, which depend on random dropout draws, agree to within 0.003 in coverage and 0.004 in Spearman correlation. The tables here are from the run with the wall-distance correction.

## Running it

```bash
uv run python pipeline/S14_dl_uncertainty/run_s14_pipeline.py                # ensemble only: about 5 minutes on a CPU
uv run python pipeline/S14_dl_uncertainty/run_s14_pipeline.py --mc-dropout   # adds MC dropout: 30 passes x 482 cases, about an hour on a GPU
uv run pytest -q src/urbanwind/dl/tests/test_uq.py
```

The seed predictions are rebuilt from the weights in the Kaggle dataset `mahirm01/s13-unet-e200-weights` with `python -m urbanwind.dl.train --config pipeline/S13_hybrid/configs/unet_tuned_e200.yaml --seed <s> --repredict`.

## Definition of done (issue #39)

- 90% intervals cover 85% to 95% of held-out cells: met. Ensemble overall 0.8995, idealized 0.9002, realistic 0.8994; MC dropout 0.8974, 0.8987, 0.8972.
- Spearman correlation between uncertainty and absolute error above 0.3: not met. Ensemble 0.2956 pooled and 0.1233 within a case; MC dropout 0.2583 and 0.1679.

## Limitations

- Three ensemble members give a rough spread. The conformal step sets the interval width, so coverage does not depend on how well sigma is scaled, only on how well it ranks cells.
- Coverage is marginal over cells. Per-case coverage varies widely, because the cells of a case share its errors.
- The epsilon of 0.05 u* in the normalised score was fixed in advance, not tuned.
