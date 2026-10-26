"""
S07: pixel-level model of the pedestrian wind field, and the gap it leaves.

A gradient-boosted tree model predicts U/u_tau at every open-ground cell from the
S05 pixel features (local density and height at four radii, distance to the
nearest wall, upwind, downwind and lateral fetch, upwind heights, canyon width
and aspect ratio, wind direction), four sheltering features added here, and six
case-level descriptors. The sheltering features are:
- shelter_ratio: the largest (upwind building height / distance) over the three
  upwind distance bands; a wake reaches a few building heights downstream;
- upwind fetch 30 degrees either side of the wind, for flow that arrives obliquely;
- upwind fetch divided by the mean building height.

Protocol
- Development cases only (frozen test groups and EXCLUDE dropped), the same five
  grouped folds as S06c: every fold holds out whole cities and layout families.
- Training uses N_SAMPLE random open-ground cells per training case, so each case
  weighs the same whatever its size; prediction covers every open cell of each
  held-out case.

Variants scored on the held-out cases
- pixel_gbm_s05      : U/u_tau predicted directly from the S05 features only (ablation).
- pixel_gbm          : U/u_tau predicted directly, with the sheltering features.
- hybrid_gbm         : the field pattern U / U_mean predicted per cell, multiplied by
                       the S06c case-level prediction of U_mean (out-of-fold).
- level_s06c         : the S06c U_mean prediction painted on every cell.
- level_oracle       : the true case mean on every cell (level only, uses the truth).

Usage: python pipeline/S07_pixel_model/s07_pixel_model.py
"""

from __future__ import annotations

import json
import pathlib
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urbanwind.features import assemble_pixel_feature_tensor, compute_directional_fetch_maps
from urbanwind.data import EXCLUDE, FLAG, load_case
from urbanwind.metrics import case_metrics
from urbanwind.splits import load_splits

REPO = pathlib.Path(__file__).resolve().parents[2]
FEATURES_CSV = REPO / "pipeline" / "S05_feature_engineering" / "outputs" / "case_features.csv"
S06C_OOF = REPO / "pipeline" / "S06_case_models" / "outputs" / "s06c" / "oof_predictions.csv"
OUT = REPO / "pipeline" / "S07_pixel_model" / "outputs"
FIG = OUT / "figures"
PREDS = REPO / "results" / "preds"

SEED = 42
N_SAMPLE = 3000
CASE_FEATURES = ["lambda_p", "mean_h", "std_h", "max_h", "lambda_f_dir", "subset_idealized"]
DROP_CHANNELS = ("building_height", "fluid_mask")  # constant on open ground
PARAMS = {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 100,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
          "random_state": SEED, "n_jobs": 10, "verbosity": -1}
EXTRA_FEATURES = ["shelter_ratio", "upwind_fetch_minus30", "upwind_fetch_plus30", "upwind_fetch_over_h"]
BAND_MID_M = {"upwind_h_1_10m": 5.5, "upwind_h_11_30m": 20.5, "upwind_h_31_60m": 45.5}
VARIANTS = ["pixel_gbm_s05", "pixel_gbm", "hybrid_gbm", "level_s06c", "level_oracle"]
SAVED = VARIANTS[1:]  # the S05-only ablation is scored but its fields are not written


def dev_cases() -> pd.DataFrame:
    splits = load_splits()
    feats = pd.read_csv(FEATURES_CSV)
    feats = feats[~feats["case"].isin(EXCLUDE) & ~feats["split_group"].isin(splits["test_groups"])].copy()
    feats["outer_fold"] = feats["split_group"].map(splits["folds"]).astype(int)
    feats["subset_idealized"] = (feats["subset"] == "idealized").astype(float)
    oof = pd.read_csv(S06C_OOF).query("features == 'full' and target == 'U_mean'")
    r2 = {m: 1 - ((g.y_true - g.y_pred) ** 2).sum() / ((g.y_true - g.y_true.mean()) ** 2).sum()
          for m, g in oof.groupby("model")}
    best = max(r2, key=r2.get)
    level = oof[oof["model"] == best].set_index("case")["y_pred"].rename("U_mean_s06c")
    print(f"level model from S06c: {best}")
    return feats.merge(level, left_on="case", right_index=True, how="left").reset_index(drop=True), best


def case_pixels(case: str, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Feature matrix (cells x features) on valid open ground, target, validity mask and channel names."""
    c = load_case(case)
    X, names = assemble_pixel_feature_tensor(c["H"], dx=c["dx"], forcing_dir_deg=c["wind_dir_deg"])
    keep = [i for i, n in enumerate(names) if n not in DROP_CHANNELS]
    valid = c["fluid"] & np.isfinite(c["U"])
    channel = dict(zip(names, X, strict=True))
    shelter = np.maximum.reduce([channel[band] / mid for band, mid in BAND_MID_M.items()])
    oblique = [compute_directional_fetch_maps(c["H"], dx=c["dx"], forcing_dir_deg=c["wind_dir_deg"] + off)[0]["upwind_fetch"]
               for off in (-30.0, 30.0)]
    extra = [shelter, *oblique, channel["upwind_fetch"] / max(row["mean_h"], 1.0)]
    local = np.vstack([X[keep][:, valid], np.stack(extra)[:, valid]]).T
    context = np.repeat(row[CASE_FEATURES].to_numpy(float)[None, :], local.shape[0], axis=0)
    names = [names[i] for i in keep] + EXTRA_FEATURES
    return np.hstack([local, context]).astype(np.float32), c["U"][valid], valid, names, c


def s05_columns(n_columns: int) -> np.ndarray:
    """Column indices without the sheltering features (S05 channels and case descriptors only)."""
    n_s05 = n_columns - len(EXTRA_FEATURES) - len(CASE_FEATURES)
    return np.r_[0:n_s05, n_s05 + len(EXTRA_FEATURES):n_columns]


def build_training_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(SEED)
    Xs, ys, ratio, case_idx = [], [], [], []
    for i, row in df.iterrows():
        X, y, _, names, _ = case_pixels(row["case"], row)
        pick = rng.choice(len(y), size=min(N_SAMPLE, len(y)), replace=False)
        Xs.append(X[pick])
        ys.append(y[pick])
        ratio.append(y[pick] / y.mean())
        case_idx.append(np.full(len(pick), i))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ratio), np.concatenate(case_idx), names + CASE_FEATURES


def save_field(variant: str, case: str, valid: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    true_map = np.full(valid.shape, np.nan, np.float32)
    pred_map = np.full(valid.shape, np.nan, np.float32)
    true_map[valid], pred_map[valid] = y_true, y_pred
    path = PREDS / f"s07_{variant}"
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"{case}.npz", y_true=true_map, y_pred=pred_map, mask=~valid)


def run_folds(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    t0 = time.time()
    X, y, ratio, case_idx, feature_names = build_training_table(df)
    print(f"training table: {X.shape[0]:,} cells x {X.shape[1]} features [{time.time() - t0:.0f}s]")
    case_fold = df["outer_fold"].to_numpy()[case_idx]
    rows, pooled, importances = [], {v: {} for v in VARIANTS}, []
    for k in sorted(df["outer_fold"].unique()):
        tr = case_fold != k
        base_cols = s05_columns(X.shape[1])
        direct_s05 = lgb.LGBMRegressor(**PARAMS).fit(X[tr][:, base_cols], y[tr])
        direct = lgb.LGBMRegressor(**PARAMS).fit(X[tr], y[tr])
        pattern = lgb.LGBMRegressor(**PARAMS).fit(X[tr], ratio[tr])
        importances.append(pd.Series(direct.booster_.feature_importance("gain"), index=feature_names))
        for _, row in df[df["outer_fold"] == k].iterrows():
            Xc, yc, valid, _, c = case_pixels(row["case"], row)
            preds = {"pixel_gbm_s05": direct_s05.predict(Xc[:, base_cols]),
                     "pixel_gbm": direct.predict(Xc),
                     "hybrid_gbm": pattern.predict(Xc) * row["U_mean_s06c"],
                     "level_s06c": np.full(len(yc), row["U_mean_s06c"]),
                     "level_oracle": np.full(len(yc), yc.mean())}
            for variant, pred in preds.items():
                if variant in SAVED:
                    save_field(variant, row["case"], valid, yc, pred)
                m = case_metrics(yc, pred, case=row["case"])
                rows.append({"variant": variant, "subset": row["subset"], "outer_fold": k,
                             "flagged": row["case"] in FLAG, **m})
                acc = pooled[variant].setdefault(row["subset"], np.zeros(4))
                acc += [len(yc), yc.sum(), (yc ** 2).sum(), ((yc - pred) ** 2).sum()]
        print(f"fold {k} done [{time.time() - t0:.0f}s]")
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    imp = (imp / imp.sum()).rename("gain_share").reset_index().rename(columns={"index": "feature"})
    return pd.DataFrame(rows), pooled, imp


def pooled_r2(acc: np.ndarray) -> float:
    n, s, ss, sse = acc
    return float(1 - sse / (ss - s * s / n))


def summary_table(per_case: pd.DataFrame, pooled: dict) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for subset in ("all", "idealized", "realistic"):
            sub = per_case[per_case["variant"] == variant]
            if subset != "all":
                sub = sub[sub["subset"] == subset]
            acc = sum(pooled[variant].values()) if subset == "all" else pooled[variant][subset]
            rows.append({"variant": variant, "subset": subset, "n_cases": len(sub),
                         "pooled_r2": pooled_r2(acc),
                         "median_within_case_r2": sub["r2"].median(),
                         "mean_mae": sub["mae"].mean(), "median_nrmse": sub["nrmse"].median(),
                         "median_corr": sub["corr"].median(),
                         "mean_delta_p95": sub["delta_p95"].mean(), "mean_delta_max": sub["delta_max"].mean(),
                         "mean_level_err": sub["level_err"].mean(), "mean_pattern_err": sub["pattern_err"].mean()})
    return pd.DataFrame(rows)


def plot_error_maps(per_case: pd.DataFrame) -> list[str]:
    sub = per_case[(per_case["variant"] == "pixel_gbm") & (per_case["subset"] == "realistic")].sort_values("r2")
    picks = {"worst": sub.iloc[0]["case"], "median": sub.iloc[len(sub) // 2]["case"], "best": sub.iloc[-1]["case"]}
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    for r, (label, case) in enumerate(picks.items()):
        pix = np.load(PREDS / "s07_pixel_gbm" / f"{case}.npz")
        hyb = np.load(PREDS / "s07_hybrid_gbm" / f"{case}.npz")
        true, pred = pix["y_true"], pix["y_pred"]
        vmax = np.nanpercentile(true, 99)
        panels = [(true, "LES (U/u*)", "viridis", 0, vmax), (pred, "pixel GBM", "viridis", 0, vmax),
                  (pred - true, "pixel GBM error", "RdBu_r", -2, 2),
                  (hyb["y_pred"] - true, "hybrid error", "RdBu_r", -2, 2)]
        for c, (field, title, cmap, lo, hi) in enumerate(panels):
            ax = axes[r, c]
            ax.set_facecolor("0.6")
            im = ax.imshow(field, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{case} ({label}): {title}" if c == 0 else title, fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Held-out realistic cases (grey = buildings); errors in U/u*", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "error_maps_realistic.png", dpi=110)
    plt.close(fig)
    return list(picks.values())


def plot_level_pattern(summary: pd.DataFrame) -> None:
    sub = summary[summary["subset"] != "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, subset in zip(axes, ("idealized", "realistic"), strict=True):
        s = sub[sub["subset"] == subset].set_index("variant").loc[VARIANTS]
        ax.bar(s.index, s["mean_level_err"], label="level error (mean bias²)", color="#c44e52")
        ax.bar(s.index, s["mean_pattern_err"], bottom=s["mean_level_err"], label="pattern error", color="#4c72b0")
        ax.set_title(f"{subset}: per-case MSE split")
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("mean squared error, (U/u*)²")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "level_pattern_error.png", dpi=150)
    plt.close(fig)


def plot_error_vs_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute error binned by distance to the nearest building, realistic held-out cases."""
    bins = np.array([0, 1, 2, 4, 8, 16, 32, 1e9])
    sums = {v: np.zeros(len(bins) - 1) for v in ("pixel_gbm", "hybrid_gbm", "level_s06c")}
    counts = np.zeros(len(bins) - 1)
    for _, row in df[df["subset"] == "realistic"].iterrows():
        Xc, yc, valid, names, _ = case_pixels(row["case"], row)
        which = np.digitize(Xc[:, names.index("dist_to_building")], bins) - 1
        counts += np.bincount(which, minlength=len(bins) - 1)
        for v in sums:
            err = np.abs(np.load(PREDS / f"s07_{v}" / f"{row['case']}.npz")["y_pred"][valid] - yc)
            sums[v] += np.bincount(which, weights=err, minlength=len(bins) - 1)
    labels = ["0-1", "1-2", "2-4", "4-8", "8-16", "16-32", ">32"]
    table = pd.DataFrame({v: s / counts for v, s in sums.items()}, index=labels)
    table["cells"] = counts.astype(int)
    fig, ax = plt.subplots(figsize=(7, 4))
    for v in sums:
        ax.plot(labels, table[v], marker="o", label=v)
    ax.set_xlabel("distance to nearest building (m)")
    ax.set_ylabel("mean |error| (U/u*)")
    ax.set_title("Where the error sits (realistic held-out cases)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_wall_distance.png", dpi=150)
    plt.close(fig)
    return table


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df, level_model = dev_cases()
    print(f"dev cases: {len(df)} ({(df.subset == 'idealized').sum()} idealized, {(df.subset == 'realistic').sum()} "
          f"realistic)")
    per_case, pooled, importance = run_folds(df)
    per_case.to_csv(OUT / "per_case_metrics.csv", index=False, float_format="%.5g")
    importance.to_csv(OUT / "feature_importance.csv", index=False, float_format="%.4f")
    summary = summary_table(per_case, pooled)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    no_flag = summary_table(per_case[~per_case["flagged"]], pooled)  # pooled R² columns not re-weighted
    no_flag[["variant", "subset", "n_cases", "median_within_case_r2", "mean_mae"]].to_csv(
        OUT / "summary_without_flagged.csv", index=False, float_format="%.4f")
    picks = plot_error_maps(per_case)
    plot_level_pattern(summary)
    dist = plot_error_vs_distance(df)
    dist.to_csv(OUT / "error_vs_wall_distance.csv", float_format="%.4f")
    (OUT / "run_info.json").write_text(json.dumps({"level_model": level_model, "n_sample_per_case": N_SAMPLE,
                                                    "lightgbm_params": PARAMS, "error_map_cases": picks}, indent=1))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
