# Urban Microclimate Modelling

Neural surrogates for pedestrian-level urban wind, trained on large-eddy simulations of idealized and real neighbourhoods.

## Overview

Urban form controls how wind moves through streets. Deep street canyons trap heat and pollution, while open corridors and tall buildings create uncomfortable or unsafe gusts. High-fidelity large-eddy simulation (LES) resolves these effects, but costs thousands of CPU-hours per neighbourhood.

This project develops data-driven surrogate models. Given a building-height map and the prevailing wind direction, they predict the time-averaged wind field at pedestrian height (1.75 m) in milliseconds.

The study focuses on surrogate accuracy and uncertainty. It is not intended for safety-critical design approval.

## Objectives

- Predict neighbourhood-scale wind statistics from engineered urban-morphology features using classical machine learning.
- Predict full pedestrian-level wind fields from building-height maps using deep learning.
- Combine both approaches in a hybrid model, and quantify predictive uncertainty with calibrated intervals.
- Evaluate generalisation on held-out cities and layout families.

## Dataset

This work uses UrbanTALES (Nazarian et al., 2025), a public collection of 538 PALM large-eddy simulations of neutral atmospheric flow over urban layouts.

| Subset | Cases | Description |
| --- | --- | --- |
| Idealized | 224 | Aligned and staggered building arrays; eight plan-area densities; uniform and variable heights; two wind directions; 0.5 m grid |
| Realistic | 314 | Neighbourhoods from 50 cities in 22 countries, derived from OpenStreetMap; uniform and real building heights; 1 m grid |

Each case provides:
- a building-height raster;
- time-averaged pedestrian-level velocity and turbulence fields;
- horizontally averaged vertical profiles;
- the simulation configuration.

The data is distributed by its authors under CC BY-NC 4.0. The simulation fields are not redistributed in this repository; `data/` holds only the case catalog and the official per-case metadata table.

## Approach

| Stage | Focus | Methods |
| --- | --- | --- |
| Data | Ingestion, quality control, preprocessing | Alignment checks, resampling to a common 1 m grid, wind-aligned canonical frame, grouped data splits |
| Phase 1: Machine learning | Area-averaged and pixel-level wind prediction | Morphology features (plan and frontal area density, street aspect ratio, aerodynamic roughness, wind connectivity); random forests, gradient boosting, Gaussian processes; SHAP; conformal prediction |
| Phase 2: Deep learning | Full-field wind prediction | U-Net with periodic padding, Fourier neural operators, attention and adversarial refinement, deep ensembles |
| Hybrid | Combining both | Feature-conditioned networks (FiLM), level-anchored field prediction, residual stacking |

All models are evaluated with grouped cross-validation, so that entire cities and layout families are held out from training. Reported metrics include:
- field MAE, normalised RMSE and R²;
- structural similarity;
- error on peak wind speeds;
- interval coverage.

## Repository structure

```text
Urban-Microclimate-Modelling/
├── README.md
├── data/                             case catalog, metadata and download script; raw and processed data are not tracked
├── src/mmnscfd/                      shared data, feature, metric and model code
├── pipeline/
│   ├── S00_setup/
│   ├── S01_data_ingestion/
│   ├── S02_preprocessing/
│   ├── S03_splits_evaluation/
│   ├── S04_eda/
│   ├── S05_feature_engineering/
│   ├── S06_case_models/
│   ├── S07_pixel_model/
│   ├── S08_interpretability_uncertainty/
│   ├── S09_phase1_report/
│   ├── S10_dl_data_pipeline/
│   ├── S11_unet_baseline/
│   ├── S12_architecture_search/
│   ├── S13_hybrid/
│   ├── S14_dl_uncertainty/
│   ├── S15_ablations_validation/
│   └── S16_phase2_report/
├── results/                          experiment log and figures
├── reports/                          phase reports
└── tests/
```

Each stage folder contains its notebooks, a short stage summary and the outputs that later stages build on.

## Citation

This project builds on the UrbanTALES dataset:

> Nazarian, N., Lu, J., Lipson, M., Hart, M. A., Liu, S., Krayenhoff, E. S., Blunn, L., and Martilli, A. (2025). UrbanTALES: A Large-Eddy Simulation Dataset for Urban Canopy Layer Turbulence and Parameterization. *Bulletin of the American Meteorological Society*, 106(12). https://doi.org/10.1175/BAMS-D-25-0061.1

Related work:

> Lu, J., Nazarian, N., et al. (2025). Machine learning predicts pedestrian wind flow from urban morphology and prevailing wind direction. *Environmental Research Letters*, 20, 054006. https://doi.org/10.1088/1748-9326/adc148

## Acknowledgements

Developed as a semester project in Advanced Machine Learning and Deep Learning. Simulation data courtesy of the UrbanTALES team at UNSW Sydney.
