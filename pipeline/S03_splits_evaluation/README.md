# S03 Splits and Evaluation Protocol

Defines the frozen train/val/test split strategy and establishes the shared evaluation protocol for all downstream stages.

| | |
| --- | --- |
| **Inputs** | `pipeline/S04_eda/S04g_combined_report/model_ready_cases.csv` |
| **Outputs** | `data/splits.json` (hash committed), `results/runs.csv` (header), `src/urbanwind/splits.py`, `src/urbanwind/metrics.py`, `src/urbanwind/evaluate.py` |
| **Gate Requirement** | splits hash committed; leakage test (no group in two places) passes; dummy predictor scored through `evaluate.py` gives case-level R² ≈ 0 |
| **Gate Status** | ✅ PASSED |

---

## Contents

| File | Description |
| --- | --- |
| `S03_Splits_and_Evaluation.ipynb` | Stage notebook — loads case table, builds GroupKFold splits, writes `splits.json`, runs all gate tests |
| `README.md` | This file |
| `S03_gate_report.md` | Formal gate evidence |

## Split Design

### Groups
- **Idealized**: 112 layout families (case name without `_dNN` wind-direction suffix)
- **Realistic**: 50 city groups (city code from case name)

### Frozen Test Set (24 groups, ~53 cases)
Opened **only** in S9 (Phase 1) and S16 (Phase 2). Never used for tuning.

- **Realistic (7 cities)**: AU-HOB, AU-SURREYHILLS, ES-BAR, FI-MS, US-CHI, US-NEW, VAL-US-CHI
  - Stratified by density class; spans Australia, Europe (ES, FI) and North America
- **Idealized (17 families)**: All λp = 0.1111 families (interpolation), all λp = 0.5289 families (extrapolation), plus representative families at other λp values

### Development Set (136 groups, 482 cases)
- GroupKFold(n=5) with fixed fold per group; seed=42
- Fold assignments saved in `data/splits.json`
- Note: S04g catalog contains 136 development groups partitioned across 5 folds (an earlier text mention of 119 groups was reconciled).

## Evaluation Protocol

All results are computed via:
```
python -m urbanwind.evaluate results/preds/<exp_id> --exp-id <id> --stage <Sxx>
```

Metrics (open-ground cells only, case-weighted):
| Metric | Code Key | Description |
|---|---|---|
| Pooled MAE | `pooled_mae` | Mean of per-case MAE values |
| Pooled RMSE | `pooled_rmse` | Mean of per-case RMSE values |
| Macro-averaged within-case R² | `pooled_r2` | Arithmetic mean of within-case R² values (`_nanmean("r2")`), **not** a globally pooled R² computed from global sum of squares. |
| median nRMSE | `median_nrmse` | Median of (per-case RMSE ÷ mean target) |
| within-case R² | `within_r2` | Pattern agreement inside a case (identical to per-case R² for flat 1D/2D arrays) |
| Pattern correlation | `median_corr` | Median over cases of the Pearson correlation between the true and predicted open-ground fields |
| ΔP95, Δmax | `mean_delta_p95`, `mean_delta_max` | Mean difference in 95th percentile and maximum wind speed |
| level err, pattern err | `mean_level_err`, `mean_pattern_err` | Mean bias squared and pattern variance error decomposition |

### Baseline Predictor Notes & Recommendations

- **Existing Gate Predictor (`dummy_s03_gate`):** The initial smoke test uses a **per-case mean predictor** ($\hat{y}_i = \bar{y}_i$). Because it predicts the mean of ground-truth test case targets, $\text{SS}_{\text{res}, i} = \text{SS}_{\text{tot}, i}$, mathematically forcing each case's $R^2_i = 1 - 1 = 0.0$ and yielding `pooled_r2 = 0.0000`. This validates pipeline plumbing, logging to `results/runs.csv`, and execution integrity, but **is not a valid out-of-sample baseline** because it uses ground-truth case targets (target leakage).
- **Recommended Follow-up Baseline:** A true out-of-sample baseline should predict the **training-set global mean** ($\hat{y} = \bar{y}_{\text{train}}$) across all evaluation cases. Implementing this training-set global-mean baseline is recommended as a separate follow-up task; evaluation code is kept unchanged in this documentation task.

## Gate Evidence

```
splits.json SHA-256: 3a28b842d4f9d1249a483049c99d5abbb9268f1e6262f77fc3018ad4526ba7a7
Leakage test: PASS — 24 test groups, 136 dev groups, 0 overlap
Dummy predictor (per-case mean): pooled_r2 = 0.0000 (validates metric plumbing; see baseline note above)
evaluate.py smoke test: PASS — 1 row appended to results/runs.csv
```
