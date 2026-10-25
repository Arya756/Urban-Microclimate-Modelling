# Stage S03 Gate Report — Splits and Evaluation Protocol

- **Total Cases (model-ready):** 535 (of 538 total; 3 excluded: known domain-size mismatch)
- **Total Groups:** 160 (112 idealized families + 50 realistic cities, minus 2 excluded topo-fail groups)
- **Frozen Test Groups:** 24 (7 realistic cities + 17 idealized families)
- **Dev Groups:** 136 (assigned to 5 GroupKFold folds)
- **GroupKFold n_splits:** 5  |  seed: 42
- **splits.json SHA-256:** `3a28b842d4f9d1249a483049c99d5abbb9268f1e6262f77fc3018ad4526ba7a7`
- **Leakage Test:** PASS — 0 groups in both test and dev sets
- **Dummy Predictor Gate:** PASS (pipeline plumbing) — per-case mean predictor → macro-averaged within-case R² = 0.0000 (Note: uses ground-truth case targets to verify logging; training-set global-mean recommended as follow-up out-of-sample baseline)
- **evaluate.py Smoke Test:** PASS — row appended to results/runs.csv
- **results/runs.csv:** EXISTS (header + 1 dummy gate run)

## Stage Gate Result: PASSED
