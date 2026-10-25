"""
evaluate.py – CLI evaluation script.

Usage
-----
    python -m mmnscfd.evaluate <pred_dir> [--exp-id EXP_ID] [--stage STAGE]
                               [--splits PATH] [--out runs.csv]

The prediction directory must contain one of:
  - case_preds.csv  (case-level predictions: columns case, target, y_true, y_pred)
  - {case}.npz      (field predictions; loaded on demand)

Output: one row appended to results/runs.csv with all pooled metrics.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

# Relative import from the same package
from mmnscfd.metrics import case_metrics, pooled_metrics

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parents[3]
DEFAULT_RUNS_CSV = REPO_ROOT / "results" / "runs.csv"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits.json"

RUNS_HEADER = [
    "exp_id", "date", "stage", "owner", "git_sha", "split_hash",
    "pred_dir", "n_cases",
    "pooled_mae", "pooled_rmse", "pooled_r2",
    "median_nrmse", "median_ssim",
    "mean_delta_p95", "mean_delta_max",
    "mean_level_err", "mean_pattern_err",
    "config",
]


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate_case_preds(pred_dir: pathlib.Path) -> list[dict]:
    """Evaluate from case_preds.csv (case-level predictions)."""
    csv_path = pred_dir / "case_preds.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"case_preds.csv not found in {pred_dir}")

    df = pd.read_csv(csv_path)
    required = {"case", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"case_preds.csv must have columns {required}; got {list(df.columns)}")

    per_case = []
    for case, grp in df.groupby("case"):
        m = case_metrics(
            y_true=grp["y_true"].values,
            y_pred=grp["y_pred"].values,
            mask=None,
            case=str(case),
        )
        per_case.append(m)
    return per_case


def evaluate_field_preds(pred_dir: pathlib.Path) -> list[dict]:
    """Evaluate from per-case {case}.npz files in pred_dir."""
    npz_files = sorted(pred_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz field prediction files found in {pred_dir}")

    per_case = []
    for npz_path in npz_files:
        case = npz_path.stem
        data = np.load(npz_path)
        if "y_pred" not in data or "y_true" not in data:
            print(f"  [WARN] {npz_path.name}: missing y_pred or y_true – skipped")
            continue
        mask = data.get("mask", None)
        m = case_metrics(
            y_true=data["y_true"],
            y_pred=data["y_pred"],
            mask=mask,
            case=case,
        )
        per_case.append(m)
    return per_case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _split_hash(splits_path: pathlib.Path) -> str:
    try:
        with splits_path.open() as f:
            data = json.load(f)
        return data.get("hash", "none")
    except Exception:
        return "none"


def _ensure_runs_csv(path: pathlib.Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(RUNS_HEADER)
        print(f"[evaluate] Created {path}")


def _append_row(path: pathlib.Path, row: dict) -> None:
    _ensure_runs_csv(path)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_HEADER, extrasaction="ignore")
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score predictions and append a row to results/runs.csv."
    )
    parser.add_argument("pred_dir", type=pathlib.Path,
                        help="Directory with case_preds.csv or *.npz field predictions")
    parser.add_argument("--exp-id", default=None,
                        help="Experiment identifier (default: pred_dir basename)")
    parser.add_argument("--stage", default="S??",
                        help="Pipeline stage tag, e.g. S06")
    parser.add_argument("--owner", default="",
                        help="Your member ID, e.g. M3")
    parser.add_argument("--splits", type=pathlib.Path, default=DEFAULT_SPLITS,
                        help="Path to splits.json")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_RUNS_CSV,
                        help="Path to runs.csv (default: results/runs.csv)")
    parser.add_argument("--config", default="",
                        help="Free-text config / hyperparameter summary")
    args = parser.parse_args(argv)

    pred_dir = args.pred_dir.expanduser().resolve()
    exp_id = args.exp_id or pred_dir.name

    print(f"[evaluate] Scoring {pred_dir} (exp_id={exp_id})")

    # Try case-level first, then field-level
    if (pred_dir / "case_preds.csv").exists():
        per_case = evaluate_case_preds(pred_dir)
        print(f"  Mode: case-level predictions ({len(per_case)} cases)")
    else:
        per_case = evaluate_field_preds(pred_dir)
        print(f"  Mode: field predictions ({len(per_case)} cases)")

    pooled = pooled_metrics(per_case)

    row = {
        "exp_id": exp_id,
        "date": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "stage": args.stage,
        "owner": args.owner,
        "git_sha": _git_sha(),
        "split_hash": _split_hash(args.splits),
        "pred_dir": str(pred_dir),
        "config": args.config,
        **pooled,
    }

    _append_row(args.out, row)
    print(f"[evaluate] Appended row to {args.out}")
    print(f"  n_cases={pooled.get('n_cases')}  "
          f"pooled_r2={pooled.get('pooled_r2', float('nan')):.4f}  "
          f"median_nrmse={pooled.get('median_nrmse', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
