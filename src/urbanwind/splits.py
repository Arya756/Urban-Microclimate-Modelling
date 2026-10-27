"""
splits.py – GroupKFold splits and frozen test-set management.

Produced by S03. All downstream stages import from here.

Public API
----------
load_splits(path)          : load and verify data/splits.json
get_fold_cases(splits, k)  : cases in development fold k  (0-indexed)
get_test_cases(splits)     : frozen held-out cases
leakage_test(splits)       : assert no group appears in both test and dev
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

_p2 = pathlib.Path(__file__).parents[2] / "data" / "splits.json"
_p3 = pathlib.Path(__file__).parents[3] / "data" / "splits.json"
SPLITS_PATH = _p2 if _p2.exists() else _p3


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_splits(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Load splits.json and return the parsed dict.

    Parameters
    ----------
    path : optional override; defaults to data/splits.json relative to repo root.

    Returns
    -------
    dict with keys: 'test_groups', 'folds', 'hash', 'n_folds', 'seed'
    """
    p = pathlib.Path(path) if path is not None else SPLITS_PATH
    with p.open() as f:
        data = json.load(f)
    _verify_hash(data, p)
    return data


def _verify_hash(data: dict, path: pathlib.Path) -> None:
    """Re-compute SHA-256 over the content excluding the stored hash field."""
    stored = data.get("hash", "")
    payload = {k: v for k, v in data.items() if k != "hash"}
    recomputed = _sha256(payload)
    if stored != recomputed:
        raise ValueError(
            f"splits.json hash mismatch!\n"
            f"  stored   : {stored}\n"
            f"  computed : {recomputed}\n"
            f"  file     : {path}\n"
            "The file may have been edited. Regenerate with the S03 notebook."
        )


def _sha256(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_test_cases(splits: dict, all_cases: list[str], case_to_group: dict[str, str]) -> list[str]:
    """Return all cases whose split_group is in the frozen test set."""
    test_groups = set(splits["test_groups"])
    return [c for c in all_cases if case_to_group.get(c, "") in test_groups]


def get_dev_cases(splits: dict, all_cases: list[str], case_to_group: dict[str, str]) -> list[str]:
    """Return all cases in the development set (not test)."""
    test_groups = set(splits["test_groups"])
    return [c for c in all_cases if case_to_group.get(c, "") not in test_groups]


def get_fold_cases(
    splits: dict,
    fold_k: int,
    all_cases: list[str],
    case_to_group: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Return (train_cases, val_cases) for development fold k (0-indexed).

    Parameters
    ----------
    fold_k : 0 … n_folds-1
    """
    n_folds = splits["n_folds"]
    if not 0 <= fold_k < n_folds:
        raise ValueError(f"fold_k must be in 0..{n_folds-1}, got {fold_k}")
    folds: dict[str, int] = splits["folds"]          # group → k (0-indexed val fold)
    dev_cases = get_dev_cases(splits, all_cases, case_to_group)
    unknown = sorted({case_to_group.get(c, "") for c in dev_cases} - set(folds))
    if unknown:  # a renamed or unlisted case would otherwise land in every training fold
        raise ValueError(f"development groups without a fold: {unknown[:5]}")

    val_cases = [c for c in dev_cases if folds.get(case_to_group.get(c, ""), -1) == fold_k]
    train_cases = [c for c in dev_cases if c not in set(val_cases)]
    return train_cases, val_cases


# ---------------------------------------------------------------------------
# Leakage test
# ---------------------------------------------------------------------------

def leakage_test(splits: dict) -> None:
    """Assert no group appears in both test_groups and the fold keys.

    Raises AssertionError with a descriptive message on failure.
    """
    test_groups: set[str] = set(splits["test_groups"])
    folds: dict[str, int] = splits["folds"]
    dev_groups: set[str] = set(folds.keys())

    overlap = test_groups & dev_groups
    assert not overlap, (
        f"DATA LEAKAGE: {len(overlap)} group(s) appear in both test and dev folds:\n"
        f"  {sorted(overlap)}"
    )
    print(f"[leakage_test] PASS: no overlap between {len(test_groups)} test groups "
          f"and {len(dev_groups)} dev groups.")
