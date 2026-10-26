# Stage S05 Gate Report — Urban Microclimate Feature Engineering

- **Date:** 2026-10-26 16:17:21 UTC
- **Total Cases Processed:** 535
- **Numeric NaN Values:** 0
- **Numeric Inf Values:** 0
- **Total Bad Numeric Values (NaN + Inf):** 0
- **Target Leakage Guarantee:** VERIFIED PASS (Wind-speed targets kept completely isolated from predictor feature extractors)
- **Periodic Boundary Verification:** PASS (Cyclic wrapping verified across periodic distance, multi-scale convolutions, and directional fetch)
- **Macdonald & Kanda Parameterizations:** PASS (Empirical formulas verified on uniform and variable building canopies)
- **Multi-channel Spatial Tensor Output:** 22 channels per case saved as compressed `.npz` files in `outputs/pixel_features/`
- **Elapsed Pipeline Runtime:** 385.47 seconds

## Key Feature Range Summary

| Feature | Min | Mean | Max | Units |
| --- | --- | --- | --- | --- |
| $\lambda_p$ (Plan Area Density) | 0.0571 | 0.2817 | 0.6400 | - |
| $\bar{H}$ (Mean Building Height) | 2.41 | 17.29 | 40.83 | m |
| $\lambda_f$ (Directional Frontal Area) | 0.0393 | 0.3413 | 1.4083 | - |
| $\lambda_w$ (Wall Area Index) | 0.1586 | 1.2055 | 5.6308 | - |
| $d / H$ (Macdonald Displacement) | 0.1339 | 0.4957 | 0.8611 | - |
| $z_0$ (Macdonald Roughness) | 0.0417 | 2.2371 | 10.3529 | m |
| $d$ (Kanda Displacement) | 3.8218 | 17.5916 | 121.4816 | m |
| $z_0$ (Kanda Roughness) | 0.1553 | 1.9991 | 28.5826 | m |
| Distance to Building (Mean) | 1.31 | 9.17 | 30.21 | m |
| Upwind Fetch (Mean) | 3.67 | 57.20 | 125.24 | m |

## Stage Gate Result: PASSED
