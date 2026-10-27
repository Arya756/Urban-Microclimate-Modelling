"""
evaluate.py – CLI evaluation script.

Usage
-----
    python -m urbanwind.evaluate <pred_dir> [--exp-id EXP_ID] [--stage STAGE]
                               [--splits PATH] [--out runs.csv]

The prediction directory must contain one of:
  - case_preds.csv  (case-level predictions: columns case, target, y_true, y_pred)
  - {case}.npz      (field predictions; loaded on demand)

Output: one row appended to results/runs.csv. "pooled" metrics pool every predicted
item: cases in scalar mode, open cells of every case in field mode. Per-case R² and
NRMSE, the level/pattern split and the peak errors are field-mode quantities and stay
empty for scalar targets. Cases from frozen test groups are refused unless
--final-test is given (S09, S16 only).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import subprocess
import sys

from typing import Any

import numpy as np
import pandas as pd

# Relative import from the same package
from urbanwind.metrics import case_metrics, pooled_metrics, regression_metrics

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_p2 = pathlib.Path(__file__).parents[2]
_p3 = pathlib.Path(__file__).parents[3]
REPO_ROOT = _p2 if (_p2 / "data").exists() else _p3
DEFAULT_RUNS_CSV = REPO_ROOT / "results" / "runs.csv"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits.json"

RUNS_HEADER = [
    "exp_id", "date", "stage", "git_sha", "split_hash",
    "pred_dir", "mode", "target", "n_cases",
    "pooled_mae", "pooled_rmse", "pooled_r2",
    "mean_case_r2", "median_case_r2",
    "median_nrmse", "median_corr",
    "mean_delta_p95", "mean_delta_max",
    "mean_level_err", "mean_pattern_err",
    "config",
]


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def is_scalar_case_table(df: pd.DataFrame) -> bool:
    """Determine whether a prediction table represents scalar case predictions
    (at most one prediction per case per target) or flattened spatial field data
    (multiple spatial cells per case for the same target).
    """
    if "target" in df.columns and "case" in df.columns:
        return not df.duplicated(subset=["case", "target"]).any()
    elif "case" in df.columns:
        if df["case"].duplicated().any():
            return False
        has_wide_targets = any(c.endswith("_true") for c in df.columns)
        has_long_targets = "y_true" in df.columns and "y_pred" in df.columns
        return has_wide_targets or has_long_targets
    return False


def evaluate_scalar_preds(
    pred_source: pathlib.Path | pd.DataFrame,
    splits_path: pathlib.Path | None = None,
    final_test: bool = False,
) -> dict[str, Any]:
    """Evaluate scalar case-level predictions across independent cases.

    Parameters
    ----------
    pred_source : pathlib.Path or pd.DataFrame
    splits_path : optional path to splits.json for frozen-test validation

    Returns
    -------
    dict with keys:
        mode: "scalar",
        n_cases: int,
        targets: dict of per-target metric results,
        test_exclusion_verified: bool
    """
    if isinstance(pred_source, (str, pathlib.Path)):
        p = pathlib.Path(pred_source)
        if not p.exists():
            raise FileNotFoundError(f"Prediction file not found: {p}")
        df = pd.read_csv(p)
    elif isinstance(pred_source, pd.DataFrame):
        df = pred_source.copy()
    else:
        raise TypeError(f"Expected Path or DataFrame, got {type(pred_source)}")

    if "case" not in df.columns:
        raise ValueError(f"Scalar prediction table must contain 'case' column; got {list(df.columns)}")

    # Detect format: Long format vs Wide format
    wide_pairs = [
        (c[:-5], c, c[:-5] + "_pred")
        for c in df.columns
        if c.endswith("_true") and c != "y_true" and (c[:-5] + "_pred") in df.columns
    ]

    target_results: dict[str, Any] = {}

    if "y_true" in df.columns and "y_pred" in df.columns:
        # Long format: case, [target], y_true, y_pred
        targets = df["target"].unique() if "target" in df.columns else ["default"]

        for tgt in targets:
            sub_df = df[df["target"] == tgt] if "target" in df.columns else df
            if sub_df["case"].duplicated().any():
                dups = sub_df[sub_df["case"].duplicated()]["case"].tolist()
                raise ValueError(f"Duplicate cases found in scalar predictions for target '{tgt}': {dups[:5]}")

            y_t = sub_df["y_true"].values
            y_p = sub_df["y_pred"].values
            if not np.all(np.isfinite(y_t)):
                raise ValueError(f"Non-finite values found in y_true for target '{tgt}'")
            if not np.all(np.isfinite(y_p)):
                raise ValueError(f"Non-finite values found in y_pred for target '{tgt}'")

            overall_m = regression_metrics(y_t, y_p, target=str(tgt))
            subsets_m = {}
            if "subset" in sub_df.columns:
                for sub_val in sub_df["subset"].dropna().unique():
                    sub_mask = sub_df["subset"] == sub_val
                    subsets_m[str(sub_val)] = regression_metrics(
                        sub_df.loc[sub_mask, "y_true"].values,
                        sub_df.loc[sub_mask, "y_pred"].values,
                        target=f"{tgt}_{sub_val}",
                    )

            folds_m = {}
            if "outer_fold" in sub_df.columns and sub_df["outer_fold"].notna().any():
                for f_val in sorted(sub_df["outer_fold"].dropna().unique()):
                    f_mask = sub_df["outer_fold"] == f_val
                    folds_m[int(f_val)] = regression_metrics(
                        sub_df.loc[f_mask, "y_true"].values,
                        sub_df.loc[f_mask, "y_pred"].values,
                        target=f"{tgt}_fold{f_val}",
                    )

            target_results[str(tgt)] = {
                "overall": overall_m,
                "subsets": subsets_m,
                "folds": folds_m,
                "cases": sub_df["case"].tolist(),
            }
        n_unique_cases = len(df["case"].unique())

    elif wide_pairs:
        # Wide format: each row is one case
        if df["case"].duplicated().any():
            dups = df[df["case"].duplicated()]["case"].tolist()
            raise ValueError(f"Duplicate cases found in scalar wide-format predictions: {dups[:5]}")

        for tgt_name, true_col, pred_col in wide_pairs:
            y_t = df[true_col].values
            y_p = df[pred_col].values
            if not np.all(np.isfinite(y_t)):
                raise ValueError(f"Non-finite values found in {true_col} for target '{tgt_name}'")
            if not np.all(np.isfinite(y_p)):
                raise ValueError(f"Non-finite values found in {pred_col} for target '{tgt_name}'")

            overall_m = regression_metrics(y_t, y_p, target=tgt_name)
            subsets_m: dict[str, Any] = {}
            if "subset" in df.columns:
                for sub_val in df["subset"].dropna().unique():
                    sub_mask = df["subset"] == sub_val
                    subsets_m[str(sub_val)] = regression_metrics(
                        df.loc[sub_mask, true_col].values,
                        df.loc[sub_mask, pred_col].values,
                        target=f"{tgt_name}_{sub_val}",
                    )

            folds_m: dict[int, Any] = {}
            if "outer_fold" in df.columns and df["outer_fold"].notna().any():
                for f_val in sorted(df["outer_fold"].dropna().unique()):
                    f_mask = df["outer_fold"] == f_val
                    folds_m[int(f_val)] = regression_metrics(
                        df.loc[f_mask, true_col].values,
                        df.loc[f_mask, pred_col].values,
                        target=f"{tgt_name}_fold{f_val}",
                    )

            target_results[tgt_name] = {
                "overall": overall_m,
                "subsets": subsets_m,
                "folds": folds_m,
                "cases": df["case"].tolist(),
            }
        n_unique_cases = len(df["case"].unique())
    else:
        raise ValueError(
            f"Unrecognized prediction table schema. Expected 'y_true'/'y_pred' or '*_true'/'*_pred' columns; "
            f"got: {list(df.columns)}"
        )

    # Validate frozen test group exclusion if split_group and splits.json are available
    test_verified = False
    s_path = splits_path or DEFAULT_SPLITS
    if "split_group" in df.columns and s_path.exists():
        try:
            with open(s_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            t_groups = set(s_data.get("test_groups", []))
            dev_groups = set(df["split_group"].dropna())
            overlap = dev_groups.intersection(t_groups)
            if overlap and not final_test:
                raise AssertionError(f"Frozen test groups detected in evaluation data: {overlap}")
            test_verified = True
        except AssertionError:
            raise
        except Exception as e:
            print(f"[WARN] Could not verify test exclusion: {e}")

    return {
        "mode": "scalar",
        "n_cases": n_unique_cases,
        "targets": target_results,
        "test_exclusion_verified": test_verified,
    }


def evaluate_case_preds(
    pred_dir: pathlib.Path,
    splits_path: pathlib.Path | None = None,
) -> dict[str, Any] | list[dict]:
    """Evaluate from case_preds.csv (or nested_oof_predictions.csv / oof_preds.csv).

    Automatically identifies whether the CSV represents scalar case predictions
    (evaluated across independent cases per target) or flattened spatial field data
    (evaluated within each case using spatial case_metrics).
    """
    csv_path = None
    for fname in ["case_preds.csv", "nested_oof_predictions.csv", "oof_preds.csv"]:
        if (pred_dir / fname).exists():
            csv_path = pred_dir / fname
            break

    if csv_path is None:
        raise FileNotFoundError(f"No prediction CSV found in {pred_dir}")

    df = pd.read_csv(csv_path)

    if is_scalar_case_table(df):
        return evaluate_scalar_preds(df, splits_path=splits_path)

    # Legacy spatial field table flattened into CSV (e.g. multiple cells per case)
    required = {"case", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"Flattened spatial prediction table must have columns {required}; got {list(df.columns)}")

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
            raise ValueError(f"{npz_path.name}: field predictions need y_true, y_pred and mask")
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

def _check_test_groups(cases, splits_path: pathlib.Path, final_test: bool) -> None:
    """Refuse cases from frozen test groups in every mode, looked up in the catalog."""
    from urbanwind.data import case_table

    test_groups = set(json.loads(pathlib.Path(splits_path).read_text())["test_groups"])
    groups = case_table()["split_group"].reindex(list(cases))
    if groups.isna().any():
        raise ValueError(f"cases not in the catalog: {list(groups[groups.isna()].index[:5])}")
    hits = sorted(set(groups) & test_groups)
    if hits and not final_test:
        raise AssertionError(f"Frozen test groups in the predictions: {hits[:5]}")


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


def _relative(path: pathlib.Path) -> str:
    """Repository-relative path, so runs.csv does not record machine-specific paths."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    parser.add_argument("--splits", type=pathlib.Path, default=DEFAULT_SPLITS,
                        help="Path to splits.json")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_RUNS_CSV,
                        help="Path to runs.csv (default: results/runs.csv)")
    parser.add_argument("--config", default="",
                        help="Free-text config / hyperparameter summary")
    parser.add_argument("--final-test", action="store_true",
                        help="Allow frozen test groups; only for the one-time final scoring (S09, S16)")
    parser.add_argument("--target", default=None,
                        help="Target variable to log for multi-target scalar models (default: U_mean)")
    args = parser.parse_args(argv)

    pred_dir = args.pred_dir.expanduser().resolve()
    exp_id = args.exp_id or pred_dir.name

    print(f"[evaluate] Scoring {pred_dir} (exp_id={exp_id})")

    # Try case-level first, then field-level
    csv_names = ("case_preds.csv", "nested_oof_predictions.csv", "oof_preds.csv")
    if pred_dir.is_file() and pred_dir.suffix == ".csv":
        csv_path = pred_dir
    else:
        csv_path = next((pred_dir / n for n in csv_names if (pred_dir / n).exists()), None)

    if csv_path is not None:
        table = pd.read_csv(csv_path)
        if is_scalar_case_table(table):
            eval_result = evaluate_scalar_preds(table, splits_path=args.splits, final_test=args.final_test)
        else:
            eval_result = evaluate_case_preds(csv_path.parent, splits_path=args.splits)

        if isinstance(eval_result, dict) and eval_result.get("mode") == "scalar":
            targets = eval_result["targets"]
            print(f"  Mode: scalar case-level predictions ({len(targets)} targets, {eval_result['n_cases']} cases)")
            for tgt_name, tgt_data in targets.items():
                m = tgt_data["overall"]
                print(f"  Target: {tgt_name:10s} | R² = {m['r2']:.4f}, MAE = {m['mae']:.4f}, RMSE = {m['rmse']:.4f} (N={m['n']})")
                for sub_name, sub_m in tgt_data.get("subsets", {}).items():
                    print(f"    Subset {sub_name:9s}: R² = {sub_m['r2']:.4f}, MAE = {sub_m['mae']:.4f}, RMSE = {sub_m['rmse']:.4f} (N={sub_m['n']})")

            # Determine primary target to log in runs.csv
            target_to_log = args.target if args.target and args.target in targets else ("U_mean" if "U_mean" in targets else list(targets.keys())[0])
            primary_m = targets[target_to_log]["overall"]
            pooled = {
                "n_cases": primary_m["n"],
                "pooled_mae": primary_m["mae"],
                "pooled_rmse": primary_m["rmse"],
                "pooled_r2": primary_m["r2"],
            }  # per-case and level/pattern metrics do not exist for one number per case
            mode, target, cases = "scalar", target_to_log, table["case"]
        else:
            per_case = eval_result
            print(f"  Mode: flattened spatial field predictions ({len(per_case)} cases)")
            pooled = pooled_metrics(per_case)
            mode, target, cases = "field", "U", [d["case"] for d in per_case]
    else:
        per_case = evaluate_field_preds(pred_dir)
        print(f"  Mode: field predictions ({len(per_case)} cases)")
        pooled = pooled_metrics(per_case)
        mode, target, cases = "field", "U", [d["case"] for d in per_case]

    if not pooled.get("n_cases"):
        raise ValueError(f"no scorable cases in {pred_dir}")
    _check_test_groups(cases, args.splits, args.final_test)

    row = {
        "exp_id": exp_id,
        "date": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(timespec="seconds"),
        "stage": args.stage,
        "git_sha": _git_sha(),
        "split_hash": _split_hash(args.splits),
        "pred_dir": _relative(pred_dir),
        "mode": mode,
        "target": target,
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
