# S07 Pixel-level model

Predicts the pedestrian-level wind speed $U/u_*$ at every open-ground cell with gradient-boosted trees. It also measures what such a tabular model cannot capture, the gap that motivates the convolutional models of Phase 2.

| | |
| --- | --- |
| Inputs | building raster and wind direction (through `urbanwind.data`), `pipeline/S05_feature_engineering/outputs/case_features.csv`, S06c out-of-fold predictions of `U_mean` |
| Outputs | `outputs/summary.csv`, `outputs/per_case_metrics.csv`, `outputs/feature_importance.csv`, `outputs/error_vs_wall_distance.csv`, `outputs/diagnostics_*.csv`, figures; predicted fields in `results/preds/s07_<variant>/<case>.npz` (not tracked) |

## Model

**Features.** 30 per open cell, computed from the building raster and the wind direction only:
- **20 S05 channels:**
  - local plan-area density and mean height within 8, 16, 32 and 64 m;
  - distance to the nearest wall;
  - upwind, downwind and lateral fetch;
  - tallest upwind building within 1–10, 11–30 and 31–60 m;
  - canyon width and aspect ratio;
  - wind direction.
- **4 sheltering features added here:**
  - the shelter ratio $h/x$ (tallest upwind building height over its distance band);
  - upwind fetch at ±30° off the wind;
  - upwind fetch divided by the mean building height.
- **6 case descriptors.**

**Training.**
- 3,000 random open cells per training case (1.45 million cells), so every case carries equal weight.
- LightGBM with fixed settings (800 trees, learning rate 0.05, 127 leaves).
- The five grouped folds of `data/splits.json`, the same as S06c. Every scored case belongs to a city or layout family the model never saw.

**Variants scored on every open cell of the held-out cases.**

| Variant | Prediction |
| --- | --- |
| `pixel_gbm_s05` | $U/u_*$ from the 20 S05 channels and case descriptors only (ablation) |
| `pixel_gbm` | $U/u_*$ from all 30 features |
| `hybrid_gbm` | the pattern $U/U_\mathrm{mean}$ from all 30 features, times the S06c XGBoost out-of-fold `U_mean` |
| `level_s06c` | the S06c `U_mean` on every cell |
| `level_oracle` | the true case mean on every cell (uses the truth; the best any flat field can do) |

## Results (482 development cases, held-out folds)

| Variant | Pooled $R^2$ | Pooled $R^2$ idealized | Pooled $R^2$ realistic | Median per-case $R^2$ idealized | Median per-case $R^2$ realistic | Mean MAE | Mean $\Delta P_{95}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `level_oracle` | 0.310 | 0.394 | 0.297 | 0 | 0 | 0.817 | -1.75 |
| `level_s06c` | 0.266 | 0.389 | 0.247 | -0.003 | -0.018 | 0.839 | -1.75 |
| `pixel_gbm_s05` | 0.564 | 0.837 | 0.525 | 0.422 | 0.207 | 0.630 | -0.78 |
| `pixel_gbm` | **0.593** | **0.880** | **0.552** | 0.457 | 0.253 | **0.607** | -0.73 |
| `hybrid_gbm` | 0.581 | 0.867 | 0.540 | **0.478** | **0.254** | 0.614 | -0.73 |

Pooled $R^2$ is computed over all open cells of all held-out cases, so it rewards getting each case's level right. Per-case $R^2$ is computed inside one case and then the median is taken over cases; it rewards only the pattern. All errors are in units of $u_*$.

1. **Level and pattern.**
   - A flat field at the true case mean explains 31% of the pooled variance, so about two thirds of the variation at pedestrian height lies within neighbourhoods.
   - The S06c prediction of the mean comes close to that oracle (0.266).
2. **The pixel model adds part of the pattern.** It raises the pooled $R^2$ to 0.593 and cuts each case's mean squared error by 62% (idealized) and 31% (realistic) against the oracle flat field.
3. **Sheltering features help.** They raise the realistic per-case $R^2$ from 0.207 to 0.253 and the idealized one from 0.422 to 0.457. Upwind fetch divided by mean height becomes the most important feature (32% of the total split gain).
4. **The hybrid matches the direct model.** The pixel model already learns each case's level from its case descriptors (level error 0.047 against 0.069 for the S06c mean painted flat), so multiplying by the case-level prediction adds nothing.
5. **Peaks are missed.**
   - The 95th percentile is underpredicted by 0.73 $u_*$ on average, and the maximum by 2.5 $u_*$.
   - Absolute error grows from 0.57 next to walls to 0.83 more than 32 m from them, where the fast streams run.
6. **The worst case is a flow structure, not a bug.** The lowest per-case $R^2$ is AU-Mel-U15_d90. There the wind blows across rows of buildings, and pedestrian-level flow is a set of canyon vortices whose cross-stream velocity changes sign from one street to the next; straight-line fetch features cannot describe it. The same layout with the wind along the streets (`_d00`) also scores below the median, but far better ($R^2$ -0.08 against -0.71).

## Diagnostics: where the limit comes from

`s07_diagnostics.py` scores the same model under two other protocols (`outputs/diagnostics_summary.csv`).

| Protocol | Median per-case $R^2$, idealized | Median per-case $R^2$, realistic | Pooled $R^2$ |
| --- | --- | --- | --- |
| Grouped folds (above) | 0.457 | 0.253 | 0.593 |
| Random 80/20 case split, as in Lu et al. (2025) | 0.438 | 0.258 | 0.448 |
| 40 of the model's own training cases | 0.781 | 0.371 | 0.658 |

- The random split does not help (0.438 and 0.258 against 0.457 and 0.253), so the grouped protocol is not what limits the model.
- On real neighbourhoods even the training cases reach a median per-case $R^2$ of only 0.371, so the features cannot describe those patterns however much data is added. On one fold, eight times more training cells per case left the score unchanged.
- The published U-Net reaches per-case $R^2$ of about 0.94 (idealized) and 0.65 (full set) under a random split. Most of that difference is therefore the representation: a convolutional network reads the geometry around every cell directly.

## Figures

- `outputs/figures/error_maps_realistic.png`: LES, prediction and errors for the worst, median and best held-out realistic case.
- `outputs/figures/level_pattern_error.png`: per-case mean squared error split into level and pattern.
- `outputs/figures/error_vs_wall_distance.png`: absolute error against distance to the nearest wall.

## Reproduce

```bash
uv run python pipeline/S07_pixel_model/s07_pixel_model.py      # about 35 minutes on 10 cores
uv run python pipeline/S07_pixel_model/s07_diagnostics.py      # about 6 minutes
for v in pixel_gbm hybrid_gbm level_s06c level_oracle; do
  uv run python -m urbanwind.evaluate results/preds/s07_$v --stage S07
done
```
