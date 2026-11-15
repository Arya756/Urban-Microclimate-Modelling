"""
S14: uncertainty maps and conformal intervals for the final S13 model (the 200-epoch U-Net).

Uncertainty methods
- ensemble:   the three seeds of the 200-epoch U-Net, read from their saved out-of-fold predictions
              (results/preds/s13_unet_tuned_e200_seed{0,1,2}/). Mean = prediction, standard deviation =
              uncertainty. No network is run, so these are exactly the scored S13 predictions.
- mc_dropout: seed 0 with only its dropout layers active, 30 passes per case, through the project's own
              prediction code (mirror averaging, bilinear return to 0.5 m grids, building mask from the
              cache). Written once with --mc-dropout to results/preds/s14_mc_dropout_seed0/; a GPU helps.

Intervals, each calibrated for the cases of outer fold k on the cells of the other folds only
- constant:             pred +/- q, the same width everywhere (plain split conformal, a baseline);
- normalised:           pred +/- q (sigma + eps), wider where the predictions disagree;
- normalised_by_subset: the same, with idealized and realistic cases calibrated separately (the main one).

Outputs (pipeline/S14_dl_uncertainty/outputs/ and figures/): coverage_summary.csv, reliability.csv,
coverage_by_distance.csv, per_case_coverage_<method>.csv, spearman_selective.csv and four figures; and
results/preds/s14_<method>/<case>.npz with y_true, y_pred, mask, lo and hi (the 90% normalised-by-subset
interval; the evaluator ignores lo and hi). The test set is never read.

Usage:
    uv run python pipeline/S14_dl_uncertainty/run_s14_pipeline.py                # ensemble only, CPU, minutes
    uv run python pipeline/S14_dl_uncertainty/run_s14_pipeline.py --mc-dropout   # also MC dropout
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from urbanwind.dl import uq  # noqa: E402
from urbanwind.dl.cache import load_cached  # noqa: E402
from urbanwind.dl.datasets import predict_arrays  # noqa: E402
from urbanwind.dl.train import (  # noqa: E402
    CHECKPOINTS, PREDS, REPO_ROOT, fold_table, inference, load_best, load_config, native_truth, pick_device,
)

STAGE = pathlib.Path(__file__).resolve().parent
OUT, FIG = STAGE / "outputs", STAGE / "figures"
CONFIG = REPO_ROOT / "pipeline" / "S13_hybrid" / "configs" / "unet_tuned_e200.yaml"
SEEDS = [f"s13_unet_tuned_e200_seed{s}" for s in range(3)]
MC_RUN, MC_PASSES = "s14_mc_dropout_seed0", 30
ALPHA = 0.1  # 90% intervals
LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)  # nominal coverages for the reliability diagram
INTERVALS = ("constant", "normalised", "normalised_by_subset")
MAIN = "normalised_by_subset"
DIST_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf)  # metres to the nearest building
DROPS = (0.0, 0.1, 0.2, 0.3)
SPEARMAN_SAMPLE = 2_000_000  # cells drawn at random for the pooled Spearman correlation
MAP_CASES = ("AU-Mel-U15_d90", "SG-Boo-U1_d30", "AU-Preston")  # the realistic cases S07 and S11 map
CELL_M = {"idealized": 0.5, "realistic": 1.0}


# ---------------------------------------------------------------------------- predictions

def load_method(method: str, case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truth, prediction, uncertainty and validity of one case on its native grid."""
    if method == "ensemble":
        files = [np.load(PREDS / run / f"{case}.npz") for run in SEEDS]
        truth, mask = files[0]["y_true"], files[0]["mask"]
        for f in files[1:]:
            if not (np.array_equal(f["mask"], mask) and np.array_equal(f["y_true"], truth, equal_nan=True)):
                raise ValueError(f"{case}: the seeds disagree on the truth or the mask")
        stack = np.stack([f["y_pred"].astype(np.float64) for f in files])
        return truth.astype(np.float64), stack.mean(axis=0), stack.std(axis=0), ~mask
    with np.load(PREDS / MC_RUN / f"{case}.npz") as f:
        return f["y_true"].astype(np.float64), f["y_pred"].astype(np.float64), f["sigma"].astype(np.float64), ~f["mask"]


def run_mc_dropout(dev: pd.DataFrame) -> None:
    """Seed 0 with dropout on: MC_PASSES predictions per case, saved as their mean and standard deviation."""
    cfg, device = load_config(CONFIG), pick_device()
    out = PREDS / MC_RUN
    out.mkdir(parents=True, exist_ok=True)
    for k in sorted(dev["fold"].unique()):
        model = load_best(cfg, CHECKPOINTS / SEEDS[0] / f"fold{k}.pt", device)
        model.eval()
        for m in model.modules():  # only dropout switches on; GroupNorm keeps no running statistics
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                m.train()
        todo = [c for c in dev.index[dev["fold"] == k] if not (out / f"{c}.npz").exists()]  # resumable
        for case in todo:
            cached = load_cached(case)
            truth = native_truth(case, cached)
            valid = np.isfinite(truth)
            with torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type != "cpu"):
                passes = np.stack([predict_arrays(model, cached, device=device, mirror_tta=cfg["mirror_tta"],
                                                  upsample=cfg["native_upsample"], **inference(cfg))
                                   for _ in range(MC_PASSES)]).astype(np.float64)
            np.savez_compressed(out / f"{case}.npz", y_true=truth.astype(np.float32), mask=~valid,
                                y_pred=np.where(valid, passes.mean(axis=0), np.nan).astype(np.float32),
                                sigma=np.where(valid, passes.std(axis=0), np.nan).astype(np.float32))
        print(f"MC dropout: fold {k} done", flush=True)


# ---------------------------------------------------------------------------- cells

def collect(method: str, dev: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Every open-ground cell of the 482 cases: |error|, sigma, fold, subset, case and wall-distance bin."""
    parts = {k: [] for k in ("abs_err", "sigma", "fold", "subset", "case", "dist_bin")}
    per_case, total = [], 0
    for i, case in enumerate(dev.index):
        truth, pred, sigma, valid = load_method(method, case)
        sub = dev.loc[case, "subset"]
        dist = uq.periodic_wall_distance(~valid, CELL_M[sub])[valid]
        err, sig = np.abs(truth[valid] - pred[valid]), sigma[valid]
        n = err.size
        total += n
        parts["abs_err"].append(err.astype(np.float32))
        parts["sigma"].append(sig.astype(np.float32))
        parts["fold"].append(np.full(n, dev.loc[case, "fold"], np.int8))
        parts["subset"].append(np.full(n, sub == "realistic", np.int8))
        parts["case"].append(np.full(n, i, np.int32))
        parts["dist_bin"].append((np.searchsorted(DIST_EDGES, dist, side="right") - 1).astype(np.int8))
        per_case.append({"case": case, "subset": sub, "fold": int(dev.loc[case, "fold"]), "n_cells": n,
                         "spearman": uq.spearman(sig, err), "mae": float(err.mean())})
    cells = {k: np.concatenate(v) for k, v in parts.items()}
    cells["sample"] = rng.choice(total, size=min(SPEARMAN_SAMPLE, total), replace=False)
    cells["per_case"] = pd.DataFrame(per_case)
    return cells


def quantiles(cells: dict, interval: str, alphas: list[float]) -> dict:
    """Fold-by-fold conformal quantiles of one interval type, each from the other folds' cells."""
    if interval == "constant":
        return uq.fold_quantiles(cells["abs_err"], cells["fold"], alphas)
    scores = cells["abs_err"] / (cells["sigma"] + np.float32(uq.EPS))
    groups = cells["subset"] if interval == "normalised_by_subset" else None
    return uq.fold_quantiles(scores, cells["fold"], alphas, groups)


def half_width(cells: dict, interval: str, q: dict, alpha: float) -> np.ndarray:
    """Per-cell half-width of one interval type at one alpha (float32, one array at a time to save memory)."""
    per_cell = np.empty(cells["abs_err"].shape, np.float32)
    for (k, g), qs in q.items():
        pick = cells["fold"] == k if g is None else (cells["fold"] == k) & (cells["subset"] == g)
        per_cell[pick] = qs[alpha]
    return per_cell if interval == "constant" else per_cell * (cells["sigma"] + np.float32(uq.EPS))


# ---------------------------------------------------------------------------- analysis

def analyse(method: str, cells: dict) -> dict[str, pd.DataFrame]:
    alphas = sorted({round(1 - lv, 4) for lv in LEVELS})
    err, subset, case = cells["abs_err"], cells["subset"], cells["case"]
    n_cases = int(case.max()) + 1
    summary, reliability, by_dist, per_case = [], [], [], cells["per_case"].copy()
    for interval in INTERVALS:
        q = quantiles(cells, interval, alphas)
        for a in alphas:
            hw = half_width(cells, interval, q, a)
            covered = err <= hw
            reliability.append({"method": method, "interval": interval, "nominal": round(1 - a, 4),
                                "coverage": float(covered.mean())})
            if a != ALPHA:
                continue
            row = {"method": method, "interval": interval, "nominal": 1 - ALPHA}
            for name, pick in (("all", slice(None)), ("idealized", subset == 0), ("realistic", subset == 1)):
                row[f"coverage_{name}"] = float(covered[pick].mean())
                row[f"mean_width_{name}"] = float(2 * hw[pick].mean(dtype=np.float64))
            cov_case = np.bincount(case, weights=covered, minlength=n_cases) / np.bincount(case, minlength=n_cases)
            per_case[f"coverage_{interval}"] = cov_case
            row["case_coverage_p10"], row["case_coverage_median"], row["case_coverage_p90"] = np.percentile(
                cov_case, [10, 50, 90])
            row["share_of_cases_85_to_95"] = float(((cov_case >= 0.85) & (cov_case <= 0.95)).mean())
            summary.append(row)
            for b in range(len(DIST_EDGES) - 1):
                pick = cells["dist_bin"] == b
                by_dist.append({"method": method, "interval": interval, "from_m": DIST_EDGES[b],
                                "to_m": DIST_EDGES[b + 1], "n_cells": int(pick.sum()),
                                "coverage": float(covered[pick].mean()) if pick.any() else np.nan,
                                "mean_width": float(2 * hw[pick].mean(dtype=np.float64)) if pick.any() else np.nan})
    s = cells["sample"]
    spear = {"method": method, "spearman_pooled_sample": uq.spearman(cells["sigma"][s], err[s]),
             "spearman_case_median": float(per_case["spearman"].median()),
             "spearman_case_share_above_0.3": float((per_case["spearman"] > 0.3).mean())}
    for d in DROPS:
        spear[f"mae_drop_{int(d * 100)}pct"] = uq.selective_mae(err, cells["sigma"], d)
    return {"summary": pd.DataFrame(summary), "reliability": pd.DataFrame(reliability),
            "by_distance": pd.DataFrame(by_dist), "per_case": per_case, "spearman": pd.DataFrame([spear])}


def write_intervals(method: str, dev: pd.DataFrame, cells: dict) -> None:
    """lo and hi of the main 90% interval next to each prediction, in the evaluator's format."""
    hw = half_width(cells, MAIN, quantiles(cells, MAIN, [ALPHA]), ALPHA)
    out = PREDS / f"s14_{method}"
    out.mkdir(parents=True, exist_ok=True)
    start = 0
    for case in dev.index:
        truth, pred, _, valid = load_method(method, case)
        n = int(valid.sum())
        lo, hi = np.full(valid.shape, np.nan, np.float32), np.full(valid.shape, np.nan, np.float32)
        lo[valid], hi[valid] = pred[valid] - hw[start:start + n], pred[valid] + hw[start:start + n]
        start += n
        np.savez_compressed(out / f"{case}.npz", y_true=truth.astype(np.float32), mask=~valid,
                            y_pred=np.where(valid, pred, np.nan).astype(np.float32), lo=lo, hi=hi)


# ---------------------------------------------------------------------------- figures

def plot_reliability(rel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.45, 1.0], [0.45, 1.0], color="0.6", lw=1, ls="--", label="ideal")
    for (method, interval), g in rel.groupby(["method", "interval"]):
        ax.plot(g["nominal"], g["coverage"], marker="o", lw=1.5, label=f"{method}, {interval}")
    ax.set(xlabel="nominal coverage", ylabel="coverage on held-out folds", xlim=(0.45, 1.0), ylim=(0.45, 1.0),
           title="Reliability (calibrated on the other folds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "reliability_diagram.png", dpi=150)
    plt.close(fig)


def plot_by_distance(by_dist: pd.DataFrame) -> None:
    main = by_dist[by_dist["interval"] == MAIN]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{int(a)}-{int(b)}" if np.isfinite(b) else f">{int(a)}" for a, b in
              main.drop_duplicates(["from_m"])[["from_m", "to_m"]].to_numpy()]
    width = 0.8 / main["method"].nunique()
    for j, (method, g) in enumerate(main.groupby("method")):
        ax.bar(np.arange(len(g)) + j * width, g["coverage"], width=width, label=method)
    ax.axhspan(0.85, 0.95, color="0.9", zorder=0, label="85 to 95%")
    ax.set_xticks(np.arange(len(labels)) + width * (main["method"].nunique() - 1) / 2, labels)
    ax.set(xlabel="distance to the nearest building (m)", ylabel="coverage of the 90% interval", ylim=(0.5, 1.0))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "coverage_by_distance.png", dpi=150)
    plt.close(fig)


def plot_selective(spear: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for _, row in spear.iterrows():
        ax.plot([100 * d for d in DROPS], [row[f"mae_drop_{int(d * 100)}pct"] for d in DROPS], marker="o",
                label=row["method"])
    ax.set(xlabel="most uncertain cells dropped (%)", ylabel="MAE of the remaining cells (u*)",
           title="Selective prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "selective_prediction.png", dpi=150)
    plt.close(fig)


def plot_maps(cells: dict, dev: pd.DataFrame) -> None:
    hw = half_width(cells, MAIN, quantiles(cells, MAIN, [ALPHA]), ALPHA)
    starts = np.concatenate([[0], np.cumsum(cells["per_case"]["n_cells"].to_numpy())])
    index = {c: i for i, c in enumerate(dev.index)}
    fig, axes = plt.subplots(len(MAP_CASES), 4, figsize=(15, 4 * len(MAP_CASES)), squeeze=False)
    for r, case in enumerate(MAP_CASES):
        truth, pred, sigma, valid = load_method("ensemble", case)
        width = np.full(valid.shape, np.nan)
        width[valid] = 2 * hw[starts[index[case]]:starts[index[case] + 1]]
        err = np.where(valid, np.abs(pred - truth), np.nan)
        panels = [(np.where(valid, pred, np.nan), "prediction (u*)", "viridis"), (err, "|error| (u*)", "magma"),
                  (np.where(valid, sigma, np.nan), "ensemble spread (u*)", "magma"),
                  (width, "90% interval width (u*)", "cividis")]
        for c, (a, title, cmap) in enumerate(panels):
            im = axes[r, c].imshow(a, origin="lower", cmap=cmap)
            axes[r, c].set_title(f"{case}: {title}", fontsize=9)
            axes[r, c].axis("off")
            fig.colorbar(im, ax=axes[r, c], fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "uncertainty_maps.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------- main

def markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mc-dropout", action="store_true", help="also run and analyse MC dropout (slow)")
    args = parser.parse_args()
    dev = fold_table("grouped")
    methods = ["ensemble"]
    if args.mc_dropout:
        run_mc_dropout(dev)
        methods.append("mc_dropout")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    results = {k: [] for k in ("summary", "reliability", "by_distance", "spearman")}
    for method in methods:
        cells = collect(method, dev, np.random.default_rng(0))
        res = analyse(method, cells)
        for k in results:
            results[k].append(res[k])
        res["per_case"].to_csv(OUT / f"per_case_coverage_{method}.csv", index=False, float_format="%.4f")
        write_intervals(method, dev, cells)
        if method == "ensemble":
            plot_maps(cells, dev)
        print(f"{method}: done", flush=True)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in results.items()}
    tables["summary"].to_csv(OUT / "coverage_summary.csv", index=False, float_format="%.4f")
    tables["reliability"].to_csv(OUT / "reliability.csv", index=False, float_format="%.4f")
    tables["by_distance"].to_csv(OUT / "coverage_by_distance.csv", index=False, float_format="%.4f")
    tables["spearman"].to_csv(OUT / "spearman_selective.csv", index=False, float_format="%.4f")
    plot_reliability(tables["reliability"])
    plot_by_distance(tables["by_distance"])
    plot_selective(tables["spearman"])
    for name in ("summary", "spearman"):
        print(f"\n{name}\n" + markdown(tables[name]))


if __name__ == "__main__":
    main()
