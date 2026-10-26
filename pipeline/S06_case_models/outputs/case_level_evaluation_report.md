# Stage S06: Corrected Case-Level Evaluation Report

## 1. Executive Summary
This report documents the correction of the **case-level evaluation logic** in the Urban Microclimate Modelling repository. The evaluation harness in `src/mmnscfd/evaluate.py` has been updated to evaluate scalar case-level models across independent cases per target, completely resolving the historical bug where metrics were calculated across heterogeneous targets within each case.

- **Primary Target:** `U_mean`
- **Model:** LightGBM Regressor (34 physics features)
- **Validation Protocol:** 5-fold outer GroupKFold × 3-fold inner GroupKFold (leakage-safe nested Optuna CV)
- **Development Sample Size:** 482 unique cases (190 idealized, 292 realistic) across 136 spatial groups
- **Frozen Test Exclusion:** All 24 frozen test groups (53 cases) verified 100% excluded

---

## 2. Root Cause Analysis of the Historical Evaluation Bug

### The Mechanism
In historical runs (e.g. `pipeline/S06_case_models/run_s06.py`), `case_preds.csv` was written with 4 scalar targets per case (`U_mean`, `U_p95`, `U_std`, `TKE_mean`), yielding $4 \times 482 = 1928$ rows.
The CLI script `src/mmnscfd/evaluate.py` implemented:
```python
for case, grp in df.groupby("case"):
    m = case_metrics(y_true=grp["y_true"].values, y_pred=grp["y_pred"].values)
```
and then averaged these case dicts with `pooled_metrics()`.

### The Mathematical Flaw
1. For each case, `grp["y_true"].values` was an array of the 4 targets:
   $$\mathbf{y}_{\text{true}} = [U_{\text{mean}}, U_{p95}, U_{\text{std}}, \text{TKE}_{\text{mean}}] \approx [0.38, 0.80, 0.14, 0.05]$$
2. `case_metrics` calculated:
   $$SS_{\text{tot}} = \sum_{t=1}^4 (y_{\text{true}, t} - \bar{y}_{\text{true}})^2 \approx 0.30$$
   $$SS_{\text{res}} = \sum_{t=1}^4 (y_{\text{pred}, t} - y_{\text{true}, t})^2 \approx 0.001$$
   $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} \approx 0.94 - 0.99$$
3. Averaging these across the 482 cases produced `pooled_r2 = 0.9440` and `median_ssim = 0.9993` in `results/runs.csv`.
4. This metric measured only that $U_{p95} > U_{\text{mean}} > U_{\text{std}} > \text{TKE}_{\text{mean}}$ within a case; it did not evaluate whether the model could predict wind speed across different urban morphologies.
5. If `case_preds.csv` contained only 1 target per case, $SS_{\text{tot}} = 0 \implies R^2 = \text{NaN}$.

---

## 3. Corrected Evaluation Results

All metrics below are computed across independent cases for `U_mean` using the nested out-of-fold predictions ([`nested_oof_predictions.csv`](outputs/nested_oof_predictions.csv)).

### Overall Development and Subset Metrics
| Partition | Cases ($N$) | Pooled $R^2$ | MAE (m/s) | RMSE (m/s) | Ground Truth Mean | Prediction Mean | Ground Truth Std | Prediction Std |
|---|---|---|---|---|---|---|---|---|
| **Overall Development** | 482 | **0.8156** | **0.0358** | **0.0586** | 0.3836 | 0.3776 | 0.1364 | 0.1155 |
| **Idealized Subset** | 190 | **0.9643** | **0.0154** | **0.0271** | 0.3407 | 0.3398 | 0.1433 | 0.1389 |
| **Realistic Subset** | 292 | **0.6629** | **0.0491** | **0.0720** | 0.4114 | 0.4022 | 0.1240 | 0.0890 |

### Outer Fold Supplementary Diagnostics
| Outer Fold | Cases | Spatial Groups | Outer Val $R^2$ | MAE (m/s) | RMSE (m/s) |
|---|---|---|---|---|---|
| **Fold 0** | 100 | 28 | 0.8180 | 0.0356 | 0.0613 |
| **Fold 1** | 100 | 27 | 0.8019 | 0.0353 | 0.0634 |
| **Fold 2** | 91 | 27 | 0.9095 | 0.0299 | 0.0423 |
| **Fold 3** | 114 | 27 | 0.7077 | 0.0438 | 0.0681 |
| **Fold 4** | 77 | 27 | 0.8463 | 0.0320 | 0.0490 |
| **Fold Mean ± Std** | — | — | **0.8167 ± 0.0657** | **0.0353 ± 0.0047** | **0.0568 ± 0.0096** |

---

## 4. Comparison with Historical Benchmarks

| Metric / Artifact | Reported Value | Evaluation Protocol | Validated Status |
|---|---|---|---|
| **`results/runs.csv` (row 3)** | $R^2 = 0.9440$ | Multi-target within-case pooling | **Invalid / Confirmed Bug** (heterogeneous targets pooled within cases) |
| **`S06_gate_report.md`** | Mean $R^2 = 0.8331$ | Single 5-fold CV with leaked Optuna tuning | **Potentially Optimistic** (tuning reused outer validation folds) |
| **`case_level_evaluation_metrics.csv`** | **Pooled $R^2$ = 0.8156**<br>**Mean Fold $R^2$ = 0.8167** | Leakage-safe nested cross-validation across cases | **Honest Generalization Estimate** (unreused validation folds, across-case regression) |

---

## 5. Historical Artifact Preservation
The following historical artifacts remain untouched and preserved for auditability:
1. `results/runs.csv` (Rows 1–4 preserved without overwriting)
2. `pipeline/S06_case_models/outputs/S06_gate_report.md`
3. `pipeline/S06_case_models/outputs/ablation_table.csv`
4. `pipeline/S06_case_models/outputs/fold_r2_summary.csv`

---

## 6. Limitations and Assumptions
1. **Target Specificity:** Scalar evaluation requires evaluating each target independently. The S06 nested cross-validation specifically optimized the primary project target `U_mean`. Additional targets (`U_p95`, `U_std`, `TKE_mean`) should be evaluated using separate dedicated nested runs if needed.
2. **Spatial Metric Inapplicability:** Spatial field metrics (`median_ssim`, `mean_pattern_err`, `delta_p95`, `delta_max`) are physically meaningless for scalar models and are logged as `NaN`.
