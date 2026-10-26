# S01 Data ingestion

Downloads the UrbanTALES case files and checks their integrity.

| | |
| --- | --- |
| Inputs | UrbanTALES website data feed (`build_catalog.py`), then `data/catalog.csv` |
| Outputs | `data/raw/<case>/` with `<case>_topo`, `<case>_ped.nc`, `<case>_p3d` and `profile-<case>.csv` for all 538 cases (about 4.8 GB, not tracked) |

## Contents

| File | Description |
| --- | --- |
| `build_catalog.py` | Rebuilds `data/catalog.csv` from the two JSON endpoints behind the UrbanTALES website (one row per case, with direct file links, descriptors and `split_group`) |
| `download_urbantales.py` | Parallel download of all case files from the direct links in `data/catalog.csv`, with retries, a socket timeout and resume support |
