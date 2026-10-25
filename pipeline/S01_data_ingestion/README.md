# S01 Data ingestion

Downloads the UrbanTALES case files and checks their integrity.

| | |
| --- | --- |
| Inputs | `data/catalog.csv` |
| Outputs | `data/raw/<case>/` with `<case>_topo`, `<case>_ped.nc`, `<case>_p3d` and `profile-<case>.csv` for all 538 cases (about 4.8 GB, not tracked) |

## Contents

| File | Description |
| --- | --- |
| `download_urbantales.py` | Parallel download of all case files from the direct links in `data/catalog.csv`, with retries, a socket timeout and resume support |
