"""Build the case-level target table from the pedestrian-level wind fields.

Stage: S02 Preprocessing

For every case, loads the wind field through urbanwind.data.load_case (topography
flip, open-ground mask from the building map, normalisation by the case's own
friction velocity) and writes open-ground summary statistics of U/u_tau and
TKE/u_tau² to outputs/case_targets.csv. Cases in urbanwind.data.EXCLUDE are listed
with their reason and no targets.

Usage: python pipeline/S02_preprocessing/build_case_targets.py
"""

from __future__ import annotations

import pathlib

import pandas as pd

from urbanwind.data import EXCLUDE, FLAG, case_table, case_targets

OUT = pathlib.Path(__file__).resolve().parent / "outputs" / "case_targets.csv"


def main() -> None:
    table = case_table()
    rows = []
    for case, info in table.iterrows():
        row = {"case": case, "subset": info["subset"], "split_group": info["split_group"],
               "status": info["status"], "note": EXCLUDE.get(case) or FLAG.get(case, "")}
        if case not in EXCLUDE:
            row.update(case_targets(case))
        rows.append(row)
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, float_format="%.6g")
    print(f"Wrote {OUT} ({len(out)} cases, {out['U_mean'].notna().sum()} with targets)")
    print(out.groupby("subset")[["U_mean", "U_p95", "U_std", "TKE_mean"]].describe().T.round(3))


if __name__ == "__main__":
    main()
