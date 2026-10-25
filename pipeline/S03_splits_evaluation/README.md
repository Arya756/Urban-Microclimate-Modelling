# S03 Splits and Evaluation Protocol

Defines the frozen train/val/test split strategy and establishes the shared evaluation protocol for all downstream stages.

| | |
| --- | --- |
| **Owner** | M5 |
| **Inputs** | `pipeline/S04_eda/S04g_combined_report/model_ready_cases.csv` |
| **Outputs** | `data/splits.json` (hash committed), `results/runs.csv` (header), `src/mmnscfd/splits.py`, `src/mmnscfd/metrics.py`, `src/mmnscfd/evaluate.py` |
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
  - Stratified by density class; includes non-Western cities (ES, FI)
- **Idealized (17 families)**: All λp = 0.1111 families (interpolation), all λp = 0.5289 families (extrapolation), plus representative families at other λp values

### Development Set (119 groups, 482 cases)
- GroupKFold(n=5) with fixed fold per group; seed=42
- Fold assignments saved in `data/splits.json`

## Evaluation Protocol

All results are computed via:
```
python -m mmnscfd.evaluate results/preds/<exp_id> --exp-id <id> --stage <Sxx> --owner <Mx>
```

Metrics (open-ground cells only, case-weighted):
| Metric | Description |
|---|---|
| MAE, RMSE, R² | Pooled over cases |
| median nRMSE | Per-case RMSE ÷ mean U, median |
| within-case R² | Pattern agreement inside a case |
| SSIM | Structural similarity |
| ΔP95, Δmax | Error on windiest 5% and maximum |
| level err, pattern err | MSE decomposition |

## Gate Evidence

```
splits.json SHA-256: 3a28b842d4f9d1249a483049c99d5abbb9268f1e6262f77fc3018ad4526ba7a7
Leakage test: PASS — 24 test groups, 136 dev groups, 0 overlap
Dummy predictor (per-case mean): pooled_r2 = 0.0000
evaluate.py smoke test: PASS — 1 row appended to results/runs.csv
```
