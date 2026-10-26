# Stage S06 — Case-Level Regression Surrogate Models

**Stage:** S06 (Phase 1 — AML)
**Status:** ✅ Gate Passed
**Date:** 2026-10-25

---

## Overview

Stage S06 trains and evaluates a hierarchy of case-level regression surrogates that predict scalar microclimate metrics from 79 building geometry and aerodynamic physics features produced by Stage S05.

### What This Stage Does

| Step | Description |
|---|---|
| B1 reproduction | Random Forest on basic morphology features — validates the CV harness against the reference baseline (R² ≈ 0.671) |
| Model ladder | Ridge, Poly-Ridge (deg-2), k-NN, Random Forest, LightGBM — all evaluated on physics features with identical grouped 5-fold CV |
| Feature ablation | LightGBM on Basic / Morphology / Physics feature sets — quantifies benefit of aerodynamic parameterizations |
| Multi-target | LightGBM trained separately on U_mean, U_p95, U_std, TKE_mean |
| Hyperparameter tuning | Optuna Bayesian optimization (60 trials, inner 3-fold GroupKFold) |
| Monotonic constraints | λ_p → U_mean constrained negative — validates physical monotonicity |
| Transfer tests | Cross-subset inference (idealized→realistic and vice versa) |
| Statistical significance | Fold-level Wilcoxon signed-rank test (N=5) + case-level Wilcoxon (N=482) |
| Contract predictions | OOF predictions saved to `results/preds/` in `evaluate.py` contract format |
| Scoring | `mmnscfd.evaluate` scores and appends to `results/runs.csv` |
| Gate report | Full markdown report with all numbers, tables and outputs |

---

## Gate Criteria

| Criterion | Threshold | Achieved | Status |
|---|---|---|---|
| U_mean grouped CV R² | > 0.671 (B1) | **0.8277** | ✅ PASS |
| U_p95  grouped CV R² | > 0.664 (B1) | **0.7504** | ✅ PASS |
| Fold-level Wilcoxon p | < 0.05 | **0.0312** | ✅ PASS |

Gate model: `LightGBM + 34 Physics Features` (default hyperparameters)
Exp ID: `s06_lgbm_default_physics_v1`

---

## Key Results

### Model Ladder (U_mean, Physics Features)

| Model | Mean R² |
|---|---|
| B1 — RF + Basic | 0.736 |
| Ridge | 0.481 |
| Ridge (deg-2 poly) | 0.008 |
| k-NN | 0.653 |
| Random Forest | 0.819 |
| **LightGBM (gate)** | **0.828** |
| LightGBM (Optuna-tuned) | **0.833** |

### Feature-Set Ablation (LightGBM, U_mean)

| Feature Set | Features | Mean R² | Δ vs Basic |
|---|---|---|---|
| Basic | 9 | 0.730 | — |
| Morphology | 17 | 0.752 | +0.022 |
| Physics | 34 | 0.822 | **+0.092** |

### Multi-Target Summary (LightGBM, Physics)

| Target | Mean R² | B2 Target |
|---|---|---|
| U_mean | 0.828 | 0.867 |
| U_p95 | 0.750 | 0.791 |
| U_std | 0.740 | 0.769 |
| TKE_mean | **0.925** | 0.941 |

### Top Feature Importances (U_mean)

1. `z0_over_h_kanda` — aerodynamic roughness (Kanda)
2. `upwind_fetch_mean_m` — upstream unobstructed distance
3. `d_over_h_kanda` — displacement height (Kanda)

---

## Directory Structure

```
pipeline/S06_case_models/
├── run_s06.py                        # Reproducible pipeline script (all 14 steps)
├── S06a_case_models.ipynb            # Jupyter notebook (interactive version)
└── outputs/
    ├── S06_gate_report.md            # Full gate report (all numbers, all tables)
    ├── ablation_table.csv            # Feature-set ablation results
    ├── fold_r2_summary.csv           # Per-fold R² for all 4 targets × 5 folds
    └── figures/
        ├── model_ladder.png          # Model comparison bar chart
        ├── ablation_features.png     # Feature-set ablation chart
        ├── true_vs_pred_umean.png    # OOF parity scatter (idealized + realistic)
        ├── feature_importance.png    # Top-15 feature importance bars
        └── multitarget_fold_r2.png   # Multi-target fold-level grouped bars

results/preds/
├── s06_lgbm_default_physics_v1/     # Gate model OOF predictions
│   └── case_preds.csv               # 964 rows: U_mean + U_p95, 482 cases each
└── s06_lgbm_tuned_physics_v1/       # Best model OOF predictions
    └── case_preds.csv               # 1928 rows: 4 targets × 482 cases

results/runs.csv                     # Official metric log (S06 rows appended)
```

---

## How to Reproduce

```bash
# From repo root
python3 pipeline/S06_case_models/run_s06.py
```

The script is self-contained. It reads from:
- `pipeline/S05_feature_engineering/outputs/case_features.csv`
- `pipeline/S04_eda/S04g_combined_report/master_case_table.csv`
- `data/splits.json`

It writes to:
- `pipeline/S06_case_models/outputs/` (reports, CSVs, figures)
- `results/preds/` (contract-format predictions)
- `results/runs.csv` (metric log via `mmnscfd.evaluate`)

---

## Dependencies

```bash
pip install lightgbm optuna mapie shap matplotlib
```

All other dependencies (`numpy`, `pandas`, `scikit-learn`, `scipy`) are already in the project environment.

---

## Key Findings & Notes for S7

1. **Physics features are essential** — the +9.2 R² point lift from physics over basic morphology is the single largest performance driver at case level.
2. **Realistic subsets are harder** — U_mean OOF R² is 0.96 (idealized) vs 0.69 (realistic), suggesting realistic urban forms need more expressive or spatial models (→ S7 pixel-level).
3. **Transfer learning fails completely** — trained on idealized, tested on realistic gives R²=0.12. The two subsets must always be trained together.
4. **Monotonic constraints are free** — enforcing λ_p↑ → U↓ changes R² by +0.0002. Always enable for physical interpretability.
5. **TKE_mean is much easier** — R²=0.925, well within reach of B2 (0.941). The aerodynamic roughness features strongly predict canopy turbulence.

---

## Case Readiness, Topography Alignment, and Exclusions

### Overview & Root-Cause Resolution
During dataset auditing, concerns were raised regarding six simulation cases:
- **`JP-Tok-V4_d00` (Simulation Duration):** The simulation ran for 4.91 hours (17,689 s, 8,834 timesteps) before stopping (due to an HPC walltime cutoff relative to the 12-hour nominal setup). Verification of the turbulence time series demonstrates that the flow field achieved turbulent stationarity well before termination (final TKE CV = 1.99%, well below the 10% threshold; duration exceeds the 2.0 h spin-up limit by 2.45x). Complete 34 physics features exist in S05, valid ground targets load without error, and out-of-fold prediction is physically sound (residual: -0.024 m/s).
- **`UG-UG-U1_d00`, `UG-UG-U2_d00`, `UG-UG-U3_d00` & `AU-Mel-U9_d00` (Topography Mismatches):** Topography and pedestrian target grid dimensions match identically. Before the `np.flipud(H)` correction, top-row-first file storage inverted the y-axis relative to NetCDF arrays, causing apparent mask divergence (~50%). With vertical orientation corrected, mask agreement reaches 90.2%–96.6%. The remaining differences are sub-grid CFD boundary interpolation artefacts at building perimeters.
- **`NP-NP-U1_d00` ("Buildings Hold Wind Values"):** NetCDF stored numerical zeros or face-staggered velocities within 3,120 building footprint cells. Enforcing `topo_H == 0.0` as the authoritative ground mask unconditionally zeroes and masks all 81,724 building cells, yielding 116,543 valid pedestrian cells.

### Auditable Exclusions
Only three cases across all 538 dataset cases remain excluded from model training:
- `CN-BE-V2_d00`, `CN-CD-V1_d00`, `CN-SH-V1_d00` (`qc_topo_valid = False` due to an upstream bounding-box metadata mismatch where domain size != coordinate span).

### Dataset and Split Integrity
- **All six investigated cases are retained** as `is_model_ready_spatial = True`.
- **482 development cases** across 136 spatial groups and 5 outer folds remain completely unchanged.
- **53 frozen test cases** across 24 test groups remain strictly isolated and untouched.
