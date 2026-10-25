# Stage S05 — Urban Microclimate Feature Engineering

Transforms preprocessed building height rasters, domain geometry, and atmospheric forcing parameters into scientifically validated case-level and pixel-level features for machine learning and deep learning surrogates.

| Attribute | Details |
| --- | --- |
| **Inputs** | `data/catalog.csv`, `data/metadata.csv`, `data/splits.json`, `data/raw/<case>/<case>_topo`, `pipeline/S04_eda/S04g_combined_report/master_case_table.csv` |
| **Outputs** | `outputs/case_features.csv`, `outputs/pixel_features/<case>_features.npz`, `outputs/feature_dictionary.json`, `outputs/S05_gate_report.md` |
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
├── feature_assembly.py       # Case-level record assembly, 19-channel pixel tensor, separate target loader
├── process_cases.py          # Batch CLI runner, memory-safe execution, progress tracking, gate reporting
└── tests/                    # Comprehensive unit tests
    ├── test_validation.py
    ├── test_morphology.py
    ├── test_aerodynamic.py
    ├── test_spatial_features.py
    ├── test_fetch_features.py
    └── test_feature_assembly.py
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
Accounts for height variability ($\sigma_H, H_{\text{max}}$):
$$\frac{d_{\text{kanda}}}{H_{\text{max}}} = c_0 X^2 + \left(a_0 \lambda_p^{b_0} - c_0 X^2\right) \left[ 1 - a_1 \exp(-b_1 Y) \right]$$
$$\frac{z_{0, \text{kanda}}}{H_{\text{max}}} = \left(1 - \frac{d}{H_{\text{max}}}\right) \left(a_2 \lambda_f^{b_2}\right) \exp(-c_2 \lambda_f) \cdot \left[1 + 0.8 \frac{\sigma_H}{\bar{H}}\right]$$
where $X = (\sigma_H + \bar{H}) / H_{\text{max}}$ and $Y = \lambda_f (\sigma_H / \bar{H})$.

### 5. Multi-Scale Spatial Convolutions
- Evaluated via exact 2D FFT circular convolution on the periodic torus:
$$\lambda_{p, R} = \mathcal{F}^{-1}\left( \mathcal{F}(M_{\text{bldg}}) \cdot \mathcal{F}(K_R) \right)$$
- Guarantees exact conservation of total mass: domain spatial mean of $\lambda_{p, R}$ equals global $\lambda_p$.

### 6. Directional Fetch Raymarching
- **Upwind fetch:** distance to building opposite flow direction ($-\vec{u}_{\text{wind}}$), capped at $150\text{ m}$.
- **Downwind fetch:** distance along flow direction ($+\vec{u}_{\text{wind}}$), capped at $60\text{ m}$.
- **Lateral fetch:** distances at $\pm 90^\circ$ relative to flow, capped at $60\text{ m}$.

---

## Pixel-Level Multichannel Tensor Channels (19 Channels)

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
| 15 | `canyon_width` | Local street canyon width ($2 \times \text{distance}$) | m |
| 16 | `aspect_ratio_hw` | Local canyon aspect ratio ($H / W$) | - |
| 17–18 | `wind_{cos,sin}` | Prevailing wind vector components $(\cos\theta, \sin\theta)$ | - |

---

## Leakage Prevention Protocol

1. **Target Isolation:** Target wind speed ($Y = U(z=1.75\text{ m}) / u_\tau$) and turbulence ($TKE / u_\tau^2$) are loaded strictly through `load_normalized_target()` into separate `'Y'` and `'TKE_norm'` arrays.
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

Process all model-ready cases:
```bash
python pipeline/S05_feature_engineering/process_cases.py --subset all --skip_existing
```
