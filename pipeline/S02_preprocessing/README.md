# S02 Profile Preprocessing & Canonical Frame

Standardizes, cleans, validates, and merges the UrbanTALES vertical profiles across all 538 simulation cases into a unified canonical frame.

| | |
| --- | --- |
| Inputs | `data/catalog.csv`, `data/raw/profiles/profile-<case>.csv` (or Google Drive `/content/drive/MyDrive/urbantales_raw/profiles`) |
| Outputs | `data/processed/profiles/canonical_profiles.parquet` (and `.csv.gz`), individual clean CSVs under `idealized/` and `realistic/`, QC inventory, physics validation report, and missing-values summary |
| Gate Requirement | 538/538 cases pass validation, exactly 128 canonical vertical levels per case, physics identities hold within tolerance ($10^{-5}$). |

## Contents

| File / Notebook | Description |
| --- | --- |
| `S02_Profile_Preprocessing.ipynb` | Comprehensive Google Colab & local stage notebook executing environment checks, profile cleaning, 538-case batch processing, physics checks, catalog metadata integration, and verified exports |
| `S02_gate_report.md` | Formal Stage S02 Gate Report documenting verification results, shape metrics, and physics pass rates |

## Data Cleaning Rules

1. **Index Artifact:** Exclude `Unnamed: 0`, which is an exported integer index artifact ($0$ to $129$).
2. **Ground Row ($z = 0.0\text{ m}$):** Drop row 0. All physical velocities, Reynolds stresses, and fluxes are unmeasured ($\text{NaN}$) at $z=0$, with only subgrid kinetic energy $sgsR = 0$.
3. **Top Boundary Artifact ($z \approx 131.5\text{ m}$):** Drop row 129 per Team Pipeline Handbook Rule 8: *"Drop the last profile row ($z \approx 131\text{ m}$), a boundary artefact"*. Verified across all 538 cases prior to dropping.
4. **Canonical Vertical Grid:** Cleaning yields exactly 128 vertical levels ($z \in [0.25, 123.57]\text{ m}$) across all 538 cases ($68,864$ total points).
5. **Legitimate Missing Values:** Preserve $\text{NaN}$ in `cdz` (canopy drag profile), which is physically defined only within the urban canopy ($z \le H$) and $\text{NaN}$ above building height.
6. **No Re-normalization:** The source profiles are already normalized by friction velocity $u_\tau = u_* = 0.2137\text{ m/s}$ ($u/u_*$, $\text{TKE}/u_*^2$).
7. **Physics Verification:**
   - Momentum flux balance: $\text{Mom} \approx \text{Tuw} + \text{Duw}$ (100.0% pass rate at $10^{-5}$ tolerance, max discrepancy $4.44 \times 10^{-16}$).
   - Turbulent viscosity: $Km \approx -\text{Mom} / \text{uGrad}$ (100.0% pass rate for non-zero velocity gradients).
