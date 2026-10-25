# Data

This folder holds descriptive tables for the UrbanTALES dataset (Nazarian et al., 2025). The simulation fields themselves are not stored here; they are downloaded into `data/raw/` (not tracked) by `pipeline/S01_data_ingestion/download_urbantales.py`.

Tables compiled by Mahir (mahir-m01) from the UrbanTALES website and metadata sheet.

| File | Contents |
| --- | --- |
| `catalog.csv` | One row per simulation case (538 rows). Holds the direct download link and size for each case file, case descriptors (subset, city, country, layout, height mode, density class, wind direction from the case name) and `split_group`, the city or layout family used for grouped data splits. |
| `metadata.csv` | The official UrbanTALES metadata sheet (538 rows): per-case friction velocity `u_tau`, plan and frontal area density, alignedness `gamma` and `gammaH`, building height statistics, domain size, layout type and location. |
| `portal_to_metadata_map.csv` | Links case names in `catalog.csv` (`portal_case`) to case names in `metadata.csv` (`metadata_NameE`). The two sources name 148 cases differently; 535 of 538 cases are matched one-to-one on grid size and geometry. |

## Data notes

- The authoritative wind direction of each case is the forcing vector `dpdxy` in its `_p3d` file. Case names in the CN-BE-V1 series are offset by 90 degrees from it, and ten realistic cases carry no direction in their name.
- Cases CN-BE-V2_d00, CN-CD-V1_d00 and CN-SH-V1_d00 have downloadable building maps that do not match the simulated domain, and are excluded from wind-field analyses.

## Source and licence

UrbanTALES is published by the Climate-Resilient Cities Lab, UNSW Sydney, at https://urbantales.vercel.app under the CC BY-NC 4.0 licence.

> Nazarian, N., Lu, J., Lipson, M., Hart, M. A., Liu, S., Krayenhoff, E. S., Blunn, L., and Martilli, A. (2025). UrbanTALES: A Large-Eddy Simulation Dataset for Urban Canopy Layer Turbulence and Parameterization. *Bulletin of the American Meteorological Society*, 106(12). https://doi.org/10.1175/BAMS-D-25-0061.1
