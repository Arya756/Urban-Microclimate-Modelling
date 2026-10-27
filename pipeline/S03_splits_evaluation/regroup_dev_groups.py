"""Merge development groups that are not independent, and reassign the five folds.

Version 1 of the splits grouped realistic cases by the case-name prefix, so one city
could sit in several groups and folds (Melbourne, Vancouver, Tokyo, Amsterdam, Lodz,
Singapore, Phoenix), the 0.5 m copies of Melbourne layouts (Val-AU-Mel) formed their
own group, and idealized families that differ only in height spread (-28/-56) were
separate groups. This script merges those development groups, rewrites split_group in
data/catalog.csv for development cases, and writes data/splits.json version 2.

The frozen test groups keep their names and members, so the test set does not change.

    uv run python pipeline/S03_splits_evaluation/regroup_dev_groups.py
"""
from __future__ import annotations

import datetime
import json
import pathlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from urbanwind.splits import _sha256

REPO = pathlib.Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog.csv"
SPLITS = REPO / "data" / "splits.json"
EXCLUDED_GROUPS = {"city-CN-CD", "city-CN-SH"}  # only excluded cases, never in a fold

CITY_MERGES = {  # development groups in the same city as another development group
    "city-VAL-AU-MEL": "city-AU-MEL", "city-AU-PRESTON": "city-AU-MEL",
    "city-CA-SUNSET": "city-CA-VAN", "city-JP-YOYOGI": "city-JP-TOK",
    "city-NL-AMSTERDAM": "city-NL-AMS", "city-PL-NARUTOWICZA": "city-PL-LIPOWA",
    "city-SG-TELOKKURAU06": "city-SG-BOO", "city-US-WESTPHOENIX": "city-US-PHO",
}


def merged_group(group: str) -> str:
    """Development group after merging; idealized -28/-56 height-spread variants share a group."""
    return re.sub(r"^(ideal-V[AS]\w+-[A-Z]+)-(28|56)$", r"\1", CITY_MERGES.get(group, group))


def main() -> None:
    old = json.loads(SPLITS.read_text())
    test_groups = set(old["test_groups"])
    catalog = pd.read_csv(CATALOG, dtype=str, keep_default_na=False)  # rewrite only split_group
    is_dev = ~catalog["split_group"].isin(test_groups | EXCLUDED_GROUPS)
    catalog.loc[is_dev, "split_group"] = catalog.loc[is_dev, "split_group"].map(merged_group)
    assert not set(catalog.loc[is_dev, "split_group"]) & test_groups

    # one row per development case, so the folds balance cases rather than groups
    dev = catalog[is_dev]
    folds = {}
    for k, (_, val) in enumerate(GroupKFold(n_splits=old["n_folds"]).split(np.zeros(len(dev)), groups=dev["split_group"])):
        folds.update({g: k for g in dev["split_group"].iloc[val].unique()})

    payload = {"version": "2", "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
               "n_folds": old["n_folds"], "seed": old["seed"], "test_groups": old["test_groups"], "folds": folds,
               "previous_hash": old["hash"],
               "note": "v2 merges development groups of one city, the Val-AU-Mel copies and -28/-56 idealized "
                       "variants; the test groups are unchanged"}
    payload["hash"] = _sha256(payload)
    catalog.to_csv(CATALOG, index=False, lineterminator="\r\n")  # the file uses CRLF
    SPLITS.write_text(json.dumps(payload, indent=2, sort_keys=True))
    sizes = dev.assign(fold=dev["split_group"].map(folds)).groupby("fold").size().tolist()
    print(f"{len(old['folds'])} -> {len(folds)} development groups; cases per fold {sizes}")


if __name__ == "__main__":
    main()
