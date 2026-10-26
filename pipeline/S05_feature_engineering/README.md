# Stage S05 — Urban Microclimate Feature Engineering

Transforms preprocessed building height rasters, domain geometry, and atmospheric forcing parameters into scientifically validated case-level and pixel-level features for machine learning and deep learning surrogates.

| Attribute | Details |
| --- | --- |
| **Inputs** | `data/raw/<case>/<case>_topo` and `_p3d` (read through `urbanwind.data`), `pipeline/S04_eda/S04g_combined_report/master_case_table.csv` (case descriptors) |
| **Outputs** | `outputs/case_features.csv` (535 cases), `outputs/feature_dictionary.json`, `outputs/S05_gate_report.md`, example figures; 22-channel pixel tensors are built on demand (`assemble_pixel_feature_tensor`) and written to `outputs/pixel_features/` only with `--save_pixel` |
| **Gate Requirement** | 100% test pass rate, exact periodic boundary handling, Macdonald (1998) & Kanda (2013) parameterizations verified, zero target leakage. |

---

## Architecture & Module Breakdown

The pipeline is organized into modular scientific components under `pipeline/S05_feature_engineering/`:

```text
pipeline/S05_feature_engineering/
├── config.py                 # Dataclasses, constants, channel ordering, default paths
├── validation.py             # Domain geometry validation and synthetic 40x40 benchmark generator
├── morphology.py             # Height statistics, volume, plan area density, connected components
├── orientation_features.py   # Frontal area index (lambda_f), wall index (lambda_w), facet orientation
├── aerodynamic.py            # Macdonald (1998) and Kanda et al. (2013) parameterizations
├── spatial_features.py       # Periodic Euclidean distance, FFT circular convolutions for local density & height
├── fetch_features.py         # Periodic raymarch for upwind, downwind, and lateral fetch
├── feature_assembly.py       # Case-level record assembly, 22-channel pixel tensor, separate target loader
├── process_cases.py          # Batch CLI runner, progress tracking, gate reporting
├── plot_feature_examples.py  # Channel grid and fetch figures for example cases
├── S05a_feature_engineering.ipynb  # First full run (superseded by process_cases.py, see below)
└── tests/                    # Comprehensive unit tests
    ├── test_validation.py
    ├── test_morphology.py
    ├── test_aerodynamic.py
    ├── test_spatial_features.py
    ├── test_fetch_features.py
    ├── test_feature_assembly.py
    ├── test_canyon_geometry.py
    ├── test_readiness.py
    ├── test_topo_orientation.py
    └── test_valid_mask.py
```

---

## Mathematical Formulations

### 1. Domain & Plan-Area Density ($\lambda_p$)
$$\lambda_p = \frac{\sum_{i \in \text{bldg}} dx \cdot dy}{L_x \cdot L_y} = \frac{N_{\text{bldg}}}{N_{\text{total}}}$$
- Evaluated with strict periodic boundaries. Ground is $0.0\text{ m}$; buildings have $H > 0.0\text{ m}$.

### 2. Directional Frontal Area Index ($\lambda_f(\theta)$)
Projected facade area facing incoming wind at angle $\theta$ (in degrees, with $0^\circ$ blowing along $+x$):
$$\lambda_f(\theta) = |\cos\theta| \lambda_{f, x} + |\sin\theta| \lambda_{f, y}$$
where:
$$\lambda_{f, x} = \frac{1}{A_{\text{total}}} \sum_{y} \sum_{x} \max\left(0, H(y, x) - H(y, (x-1)\pmod{N_x})\right) dy$$
$$\lambda_{f, y} = \frac{1}{A_{\text{total}}} \sum_{y} \sum_{x} \max\left(0, H(y, x) - H((y-1)\pmod{N_y}, x)\right) dx$$

### 3. Total Wall Area Index ($\lambda_w$)
$$\lambda_w = 2 \left(\lambda_{f, x} + \lambda_{f, y}\right)$$
Windward wall share: $\frac{\lambda_f(\theta)}{\lambda_w}$.

### 4. Aerodynamic Parameterizations
#### Macdonald et al. (1998)
$$\frac{d}{H} = 1 + \alpha^{-\lambda_p} (\lambda_p - 1)$$
$$\frac{z_0}{H} = \left(1 - \frac{d}{H}\right) \exp\left( -\left[ 0.5 \beta \frac{C_D}{\kappa^2} \left(1 - \frac{d}{H}\right) \lambda_f \right]^{-0.5} \right)$$
- Empirical parameters: $\alpha = 4.43$, $\kappa = 0.40$, $C_D = 1.20$, $\beta = 1.00$.

#### Kanda et al. (2013)
Fitted to large-eddy simulations of real city blocks; accounts for height variability ($\sigma_H$, $H_{\text{max}}$):
$$\frac{d}{H_{\text{max}}} = c_0 X^2 + \left(a_0 \lambda_p^{b_0} - c_0\right) X, \qquad X = \frac{\sigma_H + \bar{H}}{H_{\text{max}}} \le 1$$
$$z_0 = \left(b_1 Y^2 + c_1 Y + a_1\right) z_{0,\text{Macdonald}}, \qquad Y = \frac{\lambda_p \sigma_H}{\bar{H}}$$
- Constants: $a_0 = 1.29$, $b_0 = 0.36$, $c_0 = -0.17$, $a_1 = 0.71$, $b_1 = 20.21$, $c_1 = -0.77$ (as implemented in UMEP).
- For uniform heights $X = 1$ and $Y = 0$, so $d = a_0 \lambda_p^{b_0} \bar{H}$ and $z_0 = a_1 z_{0,\text{Macdonald}}$. For $X > 1$ (outside the fitted range) $d = a_0 \lambda_p^{b_0} \bar{H}$.
- $d$ is capped just below $H_{\text{max}}$: the fit exceeds it for $\lambda_p$ above about 0.55 (20 dense idealized arrays and 3 realistic cases).

### 5. Multi-Scale Spatial Convolutions
- Evaluated via exact 2D FFT circular convolution on the periodic torus:
$$\lambda_{p, R} = \mathcal{F}^{-1}\left( \mathcal{F}(M_{\text{bldg}}) \cdot \mathcal{F}(K_R) \right)$$
- Guarantees exact conservation of total mass: domain spatial mean of $\lambda_{p, R}$ equals global $\lambda_p$.

### 6. Directional Fetch Raymarching
- **Upwind fetch:** distance to building opposite flow direction ($-\vec{u}_{\text{wind}}$), capped at $150\text{ m}$.
- **Downwind fetch:** distance along flow direction ($+\vec{u}_{\text{wind}}$), capped at $60\text{ m}$.
- **Lateral fetch:** distances at $\pm 90^\circ$ relative to flow, capped at $60\text{ m}$.

---

## Pixel-Level Multichannel Tensor Channels (22 Channels)

| Channel | Identifier | Description | Units |
| --- | --- | --- | --- |
| 0 | `building_height` | Raw 2D elevation raster | m |
| 1 | `fluid_mask` | Open-space binary mask ($1 = \text{fluid}, 0 = \text{building}$) | - |
| 2 | `dist_to_building` | Periodic Euclidean distance to nearest building facade | m |
| 3–6 | `lambda_p_{8,16,32,64}m` | Local plan-area density at radii $R \in [8, 16, 32, 64]\text{ m}$ | - |
| 7–10 | `h_mean_{8,16,32,64}m` | Local mean raster height at radii $R \in [8, 16, 32, 64]\text{ m}$ | m |
| 11 | `upwind_fetch` | Upwind unobstructed distance (capped at 150 m) | m |
| 12 | `downwind_fetch` | Downwind unobstructed distance (capped at 60 m) | m |
| 13–14 | `lateral_{left,right}_fetch`| Lateral unobstructed distances (capped at 60 m) | m |
| 15–17 | `upwind_h_{1_10,11_30,31_60}m` | Tallest building upwind within 1–10, 11–30 and 31–60 m | m |
| 18 | `canyon_width` | Local street canyon width ($2 \times \text{distance}$) | m |
| 19 | `aspect_ratio_hw` | Local canyon aspect ratio ($H / W$) | - |
| 20–21 | `wind_{cos,sin}` | Prevailing wind vector components $(\cos\theta, \sin\theta)$ | - |

---

## Leakage Prevention Protocol

1. **Target Isolation & Valid Masking:** Target wind speed ($Y = U(z=1.75\text{ m}) / u_\tau$) and turbulence ($TKE / u_\tau^2$) are loaded strictly through `load_normalized_target(ped_path, u_tau, topo_H=H)` into separate `'Y'` and `'TKE_norm'` arrays. The valid mask unconditionally excludes building cells (`topo_H > 0`) to prevent CFD sub-grid boundary artefacts from leaking into the target space, while preserving legitimate zero-velocity predictions on open ground.
2. **Predictor Integrity:** Predictor feature extraction functions (`assemble_case_features`, `assemble_pixel_feature_tensor`) have zero access to simulation output fields.
3. **Split Preservation:** Preserves `recommended_split` ('train', 'val', 'test'), `split_group`, and `fold` from `splits.json` and `model_ready_cases.csv`.

---

## Reproducible Execution & CLI

Run all unit tests:
```bash
python -m pytest pipeline/S05_feature_engineering/tests -v
```

Process specific cases:
```bash
python pipeline/S05_feature_engineering/process_cases.py --cases UA0625_d00 CA-Sunset
```

Process all usable cases (about 80 s; add `--save_pixel` to also write the pixel tensors, about 5 GB):
```bash
python pipeline/S05_feature_engineering/process_cases.py
python pipeline/S05_feature_engineering/plot_feature_examples.py
```

## Data handling

- Building maps are read through `urbanwind.data.load_topo`, which flips the top-row-first `_topo` raster onto the row order of the wind fields. Rows are +y and columns +x; the wind direction is measured from +x towards +y.
- The three cases in `urbanwind.data.EXCLUDE` are skipped, leaving 535 cases.
- `S05a_feature_engineering.ipynb` records the first full run. It predates the orientation fix and the corrected Kanda parameterisation, so its outputs are historical; `case_features.csv` and the figures in `outputs/` come from `process_cases.py` and `plot_feature_examples.py`.
