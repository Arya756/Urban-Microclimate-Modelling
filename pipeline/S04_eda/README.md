# S04 Exploratory data analysis and data quality

Exploratory analysis and quality checks of every UrbanTALES file type, before any modelling. Each notebook covers one kind of file; S04g joins them into one table per case.

| | |
| --- | --- |
| Inputs | the raw case files in `data/raw/` (profiles, `_topo`, `_ped.nc`, `_p3d`, `_ts.nc`) and `data/catalog.csv` |
| Outputs | per-notebook output folders (inventories, quality flags, summaries, figures); `S04g_combined_report/master_case_table.csv` and `model_ready_cases.csv` |

## Notebooks

| Notebook | Files examined | What it checks | Main outputs |
| --- | --- | --- | --- |
| `S04a_idealized_profiles/` | vertical profiles, 224 idealized cases | inventory, missing values, the z = 0 and top-row artefacts, wind at 1.75 m, the sectional drag (`cdz`) gap, correlations | `urbantales_idealized_eda_outputs/` |
| `S04b_realistic_profiles/` | vertical profiles, 314 realistic cases | the same checks for real neighbourhoods, direction suffixes, physics checks (momentum flux identity), roughness rows | `realistic_eda_outputs/` |
| `S04c_topography_eda.ipynb` | building rasters (`_topo`) | completeness, plan-area density against the catalog, frontal-area density, height distributions, idealized layout design | `S04c_topography_outputs/` |
| `S04d_pedestrian_wind_eda.ipynb` | pedestrian wind fields (`_ped.nc`) | variables, building masking, agreement with the profiles at 1.75 m, speed and TKE distributions | `S04d_pedestrian_wind_outputs/` |
| `S04e_p3d_forcing_eda.ipynb` | run settings (`_p3d`) | wind direction from the pressure-gradient vector against the case names, forcing magnitude, grid settings | `S04e_p3d_outputs/` |
| `S04f_timeseries_eda.ipynb` | time series (`_ts.nc`) | run duration, time step, stationarity of TKE and friction velocity, $u_*$ against the metadata | `S04f_timeseries_outputs/` |
| `S04g_combined_eda_report.ipynb` | all of the above | one row per case with every quality flag, and the model-ready case list | `S04g_combined_report/` |

## Findings used downstream

- The wind direction comes from the pressure-gradient vector `dpdxy` in `_p3d`, not from the case name: the 13 cases of the CN-BE-V1 series are named 90° away from their forcing, and 10 realistic cases have no direction suffix.
- Three realistic cases (CN-BE-V2_d00, CN-CD-V1_d00, CN-SH-V1_d00) are excluded: their downloadable rasters do not match the simulated domain.
- JP-Tok-V4_d00 stopped early (4.9 h, so its fields average 0.9 h of stationary flow); it is kept and flagged.
- In the profiles, the z = 0 row (empty) and the top row (a boundary artefact) are dropped.
- The runs are long enough to be stationary, and the friction velocity from the time series matches the metadata.

## Later corrections

Some notebook statements predate the data audit in `urbanwind.data.load_case`, which is authoritative for every modelling stage:
- the building rasters are stored upside down relative to the wind fields and must be flipped;
- open ground is taken from the raster (`H == 0`), not from missing values in the wind file, and open-ground cells with a speed of exactly zero are left out;
- each case is normalised by its own friction velocity from the metadata.

See [`pipeline/S02_preprocessing`](../S02_preprocessing) for the audit numbers.
