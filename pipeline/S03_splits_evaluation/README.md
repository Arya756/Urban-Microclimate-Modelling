# S03 Splits and Evaluation Protocol

Defines the frozen train/val/test split strategy and establishes the shared evaluation protocol for all downstream stages.

| | |
| --- | --- |
| **Inputs** | `pipeline/S04_eda/S04g_combined_report/model_ready_cases.csv` |
| **Outputs** | `data/splits.json` (hash committed), `results/runs.csv` (header), `src/urbanwind/splits.py`, `src/urbanwind/metrics.py`, `src/urbanwind/evaluate.py` |
| **Gate Requirement** | splits hash committed; leakage test (no group in two places) passes; dummy predictor scored through `evaluate.py` gives case-level R² ≈ 0 |
| **Gate Status** | PASSED |

---

## Contents

| File | Description |
| --- | --- |
| `S03_Splits_and_Evaluation.ipynb` | Stage notebook: loads case table, builds GroupKFold splits, writes `splits.json` (version 1), runs all gate tests |
| `regroup_dev_groups.py` | Version 2: merges development groups that are not independent and reassigns the folds; test groups unchanged |
| `README.md` | This file |
| `S03_gate_report.md` | Formal gate evidence |

> **Do not re-run the notebook's export cells.** They rebuild `data/splits.json` from scratch (version 1 groups, new fold assignment) and start a new `results/runs.csv`, which silently changes every downstream result. The frozen splits are version 2, written by `regroup_dev_groups.py`; read them with `urbanwind.splits.load_splits`.

## Split Design

### Groups
Version 1 grouped cases by the case-name prefix: idealized cases by layout family (both wind directions), realistic cases by city code. A city code is not always a city, so version 2 (`regroup_dev_groups.py`) merges development groups that are not independent:
- one city under several codes: Melbourne (AU-MEL, AU-PRESTON and the 0.5 m copies VAL-AU-MEL), Vancouver, Tokyo, Amsterdam, Lodz, Singapore and Phoenix;
- idealized families that differ only in height spread (`-28` and `-56`).

The split_group column of `data/catalog.csv` holds the merged groups; every stage reads groups from there.

### Frozen Test Set (24 groups, ~53 cases)
Opened **only** in S9 (Phase 1) and S16 (Phase 2). Never used for tuning.

- **Realistic (7 cities)**: AU-HOB, AU-SURREYHILLS, ES-BAR, FI-MS, US-CHI, US-NEW, VAL-US-CHI
  - Stratified by density class; spans Australia, Europe (ES, FI) and North America
- **Idealized (17 families)**: families at seven of the eight density levels; every one of those levels also occurs in development, so the idealized test cases test interpolation
- The test groups were fixed in version 1 and keep their names and members. Two test cases come from cities that also have development cases (AU-SURREYHILLS in Melbourne, FI-MS in Helsinki), and most idealized test families have a development family with the same footprint and another height spread.

### Development Set (93 groups, 482 cases)
- GroupKFold(n=5) over cases, so the folds hold 96 or 97 cases each and every group sits in one fold
- Fold assignments saved in `data/splits.json` (version 2; `previous_hash` records version 1)

## Evaluation Protocol

All results are computed via:
```
python -m urbanwind.evaluate results/preds/<exp_id> --exp-id <id> --stage <Sxx>
```

Metrics (open-ground cells only). "Pooled" pools every predicted item: cases for case-level (scalar) predictions, every open cell of every case for field predictions. Columns that only exist for fields stay empty for scalar rows.
| Metric | Code Key | Description |
|---|---|---|
| Pooled MAE | `pooled_mae` | Mean absolute error over all items |
| Pooled RMSE | `pooled_rmse` | Root mean squared error over all items |
| Pooled R² | `pooled_r2` | One R² over all items (one sum of squares) |
| Per-case R² | `mean_case_r2`, `median_case_r2` | Mean and median over cases of the R² inside each case (fields only) |
| median nRMSE | `median_nrmse` | Median over cases of (per-case RMSE ÷ mean target) (fields only) |
| Pattern correlation | `median_corr` | Median over cases of the Pearson correlation between the true and predicted open-ground fields |
| ΔP95, Δmax | `mean_delta_p95`, `mean_delta_max` | Mean difference in 95th percentile and maximum wind speed |
| level err, pattern err | `mean_level_err`, `mean_pattern_err` | Mean bias squared and pattern variance error decomposition |

### Baseline Predictor Notes & Recommendations

- **Existing Gate Predictor (`dummy_s03_gate`):** The initial smoke test uses a **per-case mean predictor** ($\hat{y}_i = \bar{y}_i$). Because it predicts the mean of ground-truth test case targets, $\text{SS}_{\text{res}, i} = \text{SS}_{\text{tot}, i}$, mathematically forcing each case's $R^2_i = 1 - 1 = 0.0$, so `mean_case_r2 = 0` (in `results/runs_v1.csv` this column was still named `pooled_r2`). This validates pipeline plumbing, logging to `results/runs.csv`, and execution integrity, but **is not a valid out-of-sample baseline** because it uses ground-truth case targets (target leakage).
- **Recommended Follow-up Baseline:** A true out-of-sample baseline should predict the **training-set global mean** ($\hat{y} = \bar{y}_{\text{train}}$) across all evaluation cases. Implementing this training-set global-mean baseline is recommended as a separate follow-up task; evaluation code is kept unchanged in this documentation task.

## Gate Evidence

```
splits.json SHA-256: 3a28b842d4f9d1249a483049c99d5abbb9268f1e6262f77fc3018ad4526ba7a7
Leakage test: PASS: 24 test groups, 136 dev groups, 0 overlap
Dummy predictor (per-case mean): pooled_r2 = 0.0000 (validates metric plumbing; see baseline note above)
evaluate.py smoke test: PASS: 1 row appended to results/runs.csv
```
