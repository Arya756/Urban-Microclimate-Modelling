# Stage S05 — Feature Engineering Notebook

## `notebooks/S05_feature_engineering.ipynb`

This notebook executes and validates the complete feature-engineering pipeline for the
**Urban Microclimate Modelling via Neural Surrogates and CFD** project.

---

## Prerequisites

| Item | Required version |
|---|---|
| Python | ≥ 3.10 |
| numpy | any recent |
| pandas | any recent |
| scipy | any recent |
| h5py | any recent |
| matplotlib | any recent |
| jupyter / nbconvert | any recent |

---

## How to run on Google Colab

1. Open a new Colab notebook and run:

```python
!git clone https://github.com/mahir-m01/Urban-Microclimate-Modelling.git
import os; os.chdir('Urban-Microclimate-Modelling')
```

2. Open `notebooks/S05_feature_engineering.ipynb` via the Colab file browser.

3. **If your raw data (`*_topo`, `*_ped.nc`) lives on Google Drive**, mount Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then set `RAW_DATA_DIR_OVERRIDE` in Section 2, Cell 4 of the notebook:

```python
RAW_DATA_DIR_OVERRIDE = '/content/drive/MyDrive/urbantales/raw'
```

4. Run all cells top-to-bottom (`Runtime → Run all`).

---

## How to run locally

```bash
cd path/to/Urban-Microclimate-Modelling
jupyter notebook notebooks/S05_feature_engineering.ipynb
```

Or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute \
  --output S05_feature_engineering_executed.ipynb \
  --output-dir notebooks/ \
  notebooks/S05_feature_engineering.ipynb
```

---

## Feature-Engineering Modules Reused

| Module | Role |
|---|---|
| `pipeline/S05_feature_engineering/config.py` | Constants, paths, channel ordering |
| `pipeline/S05_feature_engineering/validation.py` | Height-map validation, domain geometry, synthetic benchmark |
| `pipeline/S05_feature_engineering/morphology.py` | Height statistics, connected components, canopy volume |
| `pipeline/S05_feature_engineering/orientation_features.py` | λ_f, λ_w, facet orientation, wind vector encoding |
| `pipeline/S05_feature_engineering/aerodynamic.py` | Macdonald (1998) and Kanda et al. (2013) parameterisations |
| `pipeline/S05_feature_engineering/spatial_features.py` | Periodic distance transform, FFT multi-scale convolutions, canyon geometry |
| `pipeline/S05_feature_engineering/fetch_features.py` | Periodic ray-marching for upwind / downwind / lateral fetch |
| `pipeline/S05_feature_engineering/feature_assembly.py` | Case-level record assembly, 22-channel pixel tensor, target loader |
| `pipeline/S05_feature_engineering/process_cases.py` | `build_feature_dictionary`, `generate_gate_report` |

---

## Generated Outputs

| File | Description |
|---|---|
| `pipeline/S05_feature_engineering/outputs/case_features.csv` | Case-level feature table (535 rows × 79 columns, full dataset) |
| `pipeline/S05_feature_engineering/outputs/pixel_features/<case>_features.npz` | 22-channel float32 tensor `X` + targets `Y`, `TKE_norm`, `valid_mask` (535 files) |
| `pipeline/S05_feature_engineering/outputs/feature_dictionary.json` | Machine-readable feature catalogue |
| `pipeline/S05_feature_engineering/outputs/S05_gate_report.md` | Gate report: NaN/Inf counts, leakage guarantee, elapsed time |
| `pipeline/S05_feature_engineering/outputs/<case>_channel_grid.png` | Visual grid of all 22 tensor channels |

---

## Validation Results (Full Dataset Execution: 535 Cases)

| Check | Result |
|---|---|
| Model-ready cases processed | **535 / 535** (100% complete) |
| Topo & Ped.nc input verification | ✅ 535 / 535 present & verified |
| Numeric NaN in case_features.csv | **0** |
| Numeric Inf in case_features.csv | **0** |
| Physical plausibility (20 checks) | ✅ all pass |
| Pixel NPZ shape (22 channels) / dtype / finite / caps | ✅ 535 / 535 pass |
| Target leakage (X vs Y separation) | ✅ NONE detected |
| Split group & metadata preserved | ✅ |
| Preprocessing outputs intact | ✅ |
| Unit tests (32 tests across 7 suites) | ✅ 32/32 pass |

---

## Dataset & Pipeline Status

- **Full Dataset Secured:** All 535 model-ready cases have both `*_topo` building elevation rasters and `*_ped.nc` pedestrian wind fields stored locally in `data/raw/`.
- **Pipeline Frozen:** Feature extraction logic, 22-channel spatial tensor assembly, and aerodynamic parameterizations are verified, tested, and frozen for subsequent neural surrogate training (Stage S06).
