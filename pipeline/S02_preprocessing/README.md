# S02 Profile Preprocessing & Canonical Frame

Standardizes, cleans, validates, and merges the UrbanTALES vertical profiles across all 538 simulation cases into a unified canonical frame.

| | |
| --- | --- |
| Inputs | `data/catalog.csv`, `data/raw/profiles/profile-<case>.csv` (or Google Drive `/content/drive/MyDrive/urbantales_raw/profiles`) |
| Outputs | `data/processed/profiles/canonical_profiles.parquet` (and `.csv.gz`), individual clean CSVs under `idealized/` and `realistic/`, QC inventory, physics validation report, and missing-values summary |
| Gate Requirement | 538/538 cases pass validation, exactly 128 canonical vertical levels per case, physics identities hold within tolerance ($10^{-5}$). |

## Scope & Spatial Resampling Note

Stage S02 processes **1D vertical profiles only** (`canonical_profiles.parquet`).
- There is **no 2D spatial resampling pipeline or 2% resampling tolerance test implemented in S02**.
- Audit verification confirmed that no cases failed a 2% resampling tolerance check because no such check was executed in S02. Unified resolution resampling (e.g. conservative area-averaging of idealized cases from $\Delta x = 0.5\text{ m}$ to $1.0\text{ m}$) is documented solely as a future modeling directive in the S04 EDA synthesis, not an implemented preprocessing stage.

## Contents

| File / Notebook | Description |
| --- | --- |
| `S02_Profile_Preprocessing.ipynb` | Comprehensive Google Colab & local stage notebook executing environment checks, profile cleaning, 538-case batch processing, physics checks, catalog metadata integration, and verified exports |
| `S02_gate_report.md` | Formal Stage S02 Gate Report documenting verification results, shape metrics, and physics pass rates |
| `build_case_targets.py` | Builds `outputs/case_targets.csv`, the case-level regression targets, from the pedestrian-level wind fields |
| `outputs/case_targets.csv` | One row per case: open-ground mean, 95th percentile and standard deviation of $U/u_*$, mean $\text{TKE}/u_*^2$, forcing direction, grid spacing, $u_*$ and an `ok` / `flag` / `exclude` status |

## Case-level targets

`build_case_targets.py` reads every case through `urbanwind.data.load_case`, which:
- flips the `_topo` raster (stored top row first) onto the row order of the `_ped.nc` fields;
- takes open ground from the raster (`H == 0`). Taking it from NaN in the wind file is wrong for files that store 0 inside buildings: for 51 cases the mean wind changes by more than 1%, by up to 11.7% (AU-Mel-U1_d00);
- divides each field by the case's own friction velocity $u_*$ from `data/metadata.csv` (0.2015 to 0.2106 m/s), joined through `data/portal_to_metadata_map.csv`;
- reads the wind direction from the `dpdxy` forcing in `_p3d`.

Statistics are taken over open-ground cells with a finite value. Three cases are excluded and carry no targets: CN-BE-V2_d00, CN-CD-V1_d00 and CN-SH-V1_d00 (the downloadable building map does not match the simulated domain). Six cases are kept and marked `flag`, so that results can be checked without them: JP-Tok-V4_d00 (the run stopped at 4.9 h, so its fields average 0.9 h of stationary flow instead of 8 h) and UG-UG-U1/U2/U3, NP-NP-U1 and AU-Mel-U9 (building map and wind field disagree on up to 13% of building cells).

## Data Cleaning Rules

1. **Index Artifact:** Exclude `Unnamed: 0`, which is an exported integer index artifact ($0$ to $129$).
2. **Ground Row ($z = 0.0\text{ m}$):** Drop row 0. All physical velocities, Reynolds stresses, and fluxes are unmeasured ($\text{NaN}$) at $z=0$, with only subgrid kinetic energy $sgsR = 0$.
3. **Top Boundary Artifact ($z \approx 131.5\text{ m}$):** Drop row 129, the last profile row ($z \approx 131\text{ m}$), which is a boundary artefact of the simulation output. Verified across all 538 cases prior to dropping.
4. **Canonical Vertical Grid:** Cleaning yields exactly 128 vertical levels ($z \in [0.25, 123.57]\text{ m}$) across all 538 cases ($68,864$ total points).
5. **Legitimate Missing Values:** Preserve $\text{NaN}$ in `cdz` (canopy drag profile), which is physically defined only within the urban canopy ($z \le H$) and $\text{NaN}$ above building height.
6. **No Re-normalization:** The source profiles are already normalized by the friction velocity $u_*$ of each case ($u/u_*$, $\text{TKE}/u_*^2$).
7. **Physics Verification & Boundary Stencil Policy:**
   - **Momentum flux balance:** $\text{Mom} \approx \text{Tuw} + \text{Duw}$ (100.0% pass rate at $10^{-5}$ tolerance, evaluated across all 68,864 points with max discrepancy $4.44 \times 10^{-16}$).
   - **Turbulent viscosity:** $Km \approx -\text{Mom} / \text{uGrad}$ (100.0% pass rate at $10^{-4}$ tolerance across all 68,319 valid finite-gradient points, max discrepancy $1.69 \times 10^{-7}$).
   - **Lowest-level boundary stencil limit:** At the lowest vertical level ($z = 0.25\text{ m}$), `uGrad = NaN` and `Km = NaN` across all 538 cases. This is expected physical and numerical behavior: because the ground level ($z = 0.0\text{ m}$) is unmeasured, the one-sided finite-difference stencil for the vertical velocity gradient cannot be evaluated at $z = 0.25\text{ m}$, leaving eddy viscosity undefined. In addition, exactly 7 interior measurement levels exhibit near-zero velocity gradients ($|\text{uGrad}| \le 10^{-6}\text{ s}^{-1}$).
