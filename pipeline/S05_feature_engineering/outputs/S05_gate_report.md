# Stage S05 Gate Report: Urban Microclimate Feature Engineering

- **Date:** 2026-10-27 09:55:39 UTC
- **Total Cases Processed:** 535
- **Numeric NaN Values:** 0
- **Numeric Inf Values:** 0
- **Total Bad Numeric Values (NaN + Inf):** 0
- **Target Leakage Guarantee:** VERIFIED PASS (Wind-speed targets kept completely isolated from predictor feature extractors)
- **Periodic Boundary Verification:** PASS (Cyclic wrapping verified across periodic distance, multi-scale convolutions, and directional fetch)
- **Macdonald & Kanda Parameterizations:** PASS (Macdonald et al. 1998; Kanda et al. 2013 in its published form, verified analytically on uniform and variable canopies)
- **Multi-channel Spatial Tensor:** 22 channels per case, built on demand by `assemble_pixel_feature_tensor` (written to `outputs/pixel_features/` only with `--save_pixel`)
- **Elapsed Pipeline Runtime:** 80.17 seconds

## Key Feature Range Summary

| Feature | Min | Mean | Max | Units |
| --- | --- | --- | --- | --- |
| $\lambda_p$ (Plan Area Density) | 0.0571 | 0.2817 | 0.6400 | - |
| $\bar{H}$ (Mean Building Height) | 2.41 | 17.29 | 40.83 | m |
| $\lambda_f$ (Directional Frontal Area) | 0.0393 | 0.3430 | 1.4083 | - |
| $\lambda_w$ (Wall Area Index) | 0.1586 | 1.2123 | 5.6308 | - |
| $d / H$ (Macdonald Displacement) | 0.1339 | 0.4957 | 0.8611 | - |
| $z_0$ (Macdonald Roughness) | 0.0417 | 2.2467 | 10.3529 | m |
| $d$ (Kanda Displacement) | 2.7842 | 16.8443 | 53.3060 | m |
| $z_0$ (Kanda Roughness) | 0.0327 | 1.7756 | 8.5384 | m |
| Distance to Building (Mean) | 1.31 | 9.10 | 30.21 | m |
| Upwind Fetch (Mean) | 3.67 | 56.97 | 125.24 | m |

## Stage Gate Result: PASSED
