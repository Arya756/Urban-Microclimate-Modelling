# Data

This folder holds descriptive tables for the UrbanTALES dataset (Nazarian et al., 2025). The simulation fields themselves are not stored here; they are downloaded into `data/raw/` (not tracked) by `pipeline/S01_data_ingestion/download_urbantales.py`.

| File | Contents |
| --- | --- |
| `catalog.csv` | One row per simulation case (538 rows). Holds the direct download link and size for each case file, case descriptors (subset, city, country, layout, height mode, density class, wind direction from the case name) and `split_group`, the group used for grouped data splits: a city for realistic development cases, a layout family (both wind directions and both height spreads) for idealized ones; frozen test groups keep their original city-code names (see `pipeline/S03_splits_evaluation`). |
| `metadata.csv` | The official UrbanTALES metadata sheet (538 rows): per-case friction velocity `u_tau`, plan and frontal area density, alignedness `gamma` and `gammaH`, building height statistics, domain size, layout type and location. |
| `portal_to_metadata_map.csv` | Links case names in `catalog.csv` (`portal_case`) to case names in `metadata.csv` (`metadata_NameE`). The two sources name 148 cases differently; 535 of 538 cases are matched one-to-one on grid size and geometry. |

## Data notes

- The authoritative wind direction of each case is the forcing vector `dpdxy` in its `_p3d` file. Case names in the CN-BE-V1 series are offset by 90 degrees from it, and ten realistic cases carry no direction in their name.
- Cases CN-BE-V2_d00, CN-CD-V1_d00 and CN-SH-V1_d00 have downloadable building maps that do not match the simulated domain, and are excluded from wind-field analyses.
- The `_topo` rasters are written top row first; flipping them vertically aligns them with the `_ped.nc` fields (median open-ground agreement rises from 0.71 to 0.99). Some files store 0 rather than NaN inside buildings, so open ground is taken from the raster (`H == 0`). `urbanwind.data.load_case` applies both rules.
- Each case has its own friction velocity `u_tau` (0.2015 to 0.2106 m/s); its square is linear in the air volume of the 128 m domain. Wind fields are normalised by this per-case value. The `us*` series in the `_ts.nc` files (about 0.06 m/s) is the surface skin friction, not `u_tau`.
- The metadata frontal area density `lf` equals the raster value for idealized cases, but is 1.4 to 2 times the raster value for realistic cases, which suggests a different calculation; the raster value from S05 is used for modelling.
- For the 128 idealized cases with variable heights, the `-SM` and `-CL` name suffixes are swapped relative to the metadata `Config` field. The case preview images agree with the names (CL cases group the tall blocks together).
- JP-Tok-V4_d00 stopped at 4.9 h, so its fields average 0.9 h of stationary flow instead of 8 h. It and five cases whose building map disagrees with the wind field on up to 13% of building cells (UG-UG-U1/U2/U3, NP-NP-U1, AU-Mel-U9) are kept but flagged, and results are also reported without them.

## Source and licence

UrbanTALES is published by the Climate-Resilient Cities Lab, UNSW Sydney, at https://urbantales.vercel.app under the CC BY-NC 4.0 licence.

> Nazarian, N., Lu, J., Lipson, M., Hart, M. A., Liu, S., Krayenhoff, E. S., Blunn, L., and Martilli, A. (2025). UrbanTALES: A Large-Eddy Simulation Dataset for Urban Canopy Layer Turbulence and Parameterization. *Bulletin of the American Meteorological Society*, 106(12). https://doi.org/10.1175/BAMS-D-25-0061.1
