"""
S06c: case-level model comparison with nested grouped cross-validation.

Predicts open-ground summary statistics of pedestrian-level wind for each case
(U_mean, U_p95, U_std and TKE_mean, normalised by the case's own u_tau) from the
S05 morphology features, and compares eight regressors on identical folds.

Protocol
- Development cases only: the frozen test groups of data/splits.json and the
  cases in urbanwind.data.EXCLUDE are dropped on load.
- Outer loop: the five fixed folds of data/splits.json; every fold holds out
  whole cities and whole idealized layout families.
- Inner loop: 5-fold GroupKFold on the outer-training cases only, used to tune
  hyperparameters with Optuna (TPE sampler). The outer-validation cases never
  influence tuning.
- Scores come from the out-of-fold predictions of the outer loop.

Experiments
1. Model ladder: mean, ridge, kNN, SVR, Gaussian process, random forest, XGBoost
   and LightGBM on the full feature set, for all four targets; the three tree
   ensembles also with fixed default settings, to show what tuning adds.
2. Feature ablation (basic, morphology, full) for the three tree ensembles.
3. Literature-style baseline B1: untuned random forest on the basic features.
4. Leakage check: random KFold against grouped folds for the same models.
5. Transfer between the idealized and realistic subsets.
6. Group bootstrap confidence intervals for differences in R².
7. TreeSHAP attributions of the best tree model on held-out folds.

Usage: python pipeline/S06_case_models/s06c_model_comparison.py [--trials 30]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from urbanwind.data import EXCLUDE, FLAG
from urbanwind.metrics import regression_metrics
from urbanwind.splits import load_splits

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO = pathlib.Path(__file__).resolve().parents[2]
FEATURES_CSV = REPO / "pipeline" / "S05_feature_engineering" / "outputs" / "case_features.csv"
TARGETS_CSV = REPO / "pipeline" / "S02_preprocessing" / "outputs" / "case_targets.csv"
OUT = REPO / "pipeline" / "S06_case_models" / "outputs" / "s06c"
FIG = OUT / "figures"
PREDS = REPO / "results" / "preds"

SEED = 42
TARGETS = ["U_mean", "U_p95", "U_std", "TKE_mean"]
BASIC = ["lambda_p", "mean_h", "std_h", "max_h", "forcing_dir_deg", "lambda_f_dir", "wind_cos", "wind_sin",
         "subset_idealized"]
MORPH = BASIC + ["lambda_w", "skew_h", "cv_h", "max_to_mean_h", "volume_per_area_m", "components_per_hectare",
                 "mean_footprint_area_m2", "gamma_h"]
FULL = MORPH + ["lambda_f_x", "lambda_f_y", "d_over_h_macdonald", "z0_over_h_macdonald", "d_over_h_kanda",
                "z0_over_h_kanda", "upwind_fetch_mean_m", "upwind_fetch_median_m", "downwind_fetch_mean_m",
                "lateral_fetch_mean_m", "unobstructed_fraction_gt50m", "dist_mean_m", "dist_median_m", "dist_p90_m",
                "facet_orientation_entropy", "windward_wall_share", "relative_wind_facet_angle_deg"]
FEATURE_SETS = {"basic": BASIC, "morphology": MORPH, "full": FULL}
TREE_MODELS = ["random_forest", "xgboost", "lightgbm"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dev() -> pd.DataFrame:
    """Development cases with features, targets and their outer fold."""
    splits = load_splits()
    feats = pd.read_csv(FEATURES_CSV)
    targets = pd.read_csv(TARGETS_CSV)[["case", *TARGETS]]
    df = feats.merge(targets, on="case", how="inner")
    df = df[~df["case"].isin(EXCLUDE) & ~df["split_group"].isin(splits["test_groups"])].copy()
    df["outer_fold"] = df["split_group"].map(splits["folds"]).astype(int)
    df["subset_idealized"] = (df["subset"] == "idealized").astype(float)
    df["flagged"] = df["case"].isin(FLAG)
    assert df[FULL + TARGETS].notna().all().all(), "missing features or targets"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Models: builder(params) -> estimator, and an Optuna search space (None = no tuning)
# ---------------------------------------------------------------------------

def _scaled_target(estimator):
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), estimator), transformer=StandardScaler())


def _boosting_space(t) -> dict:
    """Search space shared by XGBoost and LightGBM, so the two are tuned on equal terms."""
    return {"n_estimators": t.suggest_int("n_estimators", 200, 1500),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def _gp(_params):
    kernel = ConstantKernel() * Matern(length_scale=np.ones(1), nu=2.5) + WhiteKernel()
    return _scaled_target(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=SEED))


MODELS = {
    "mean": (lambda p: DummyRegressor(), None),
    "ridge": (lambda p: make_pipeline(StandardScaler(), Ridge(**p)),
              lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True)}),
    "knn": (lambda p: make_pipeline(StandardScaler(), KNeighborsRegressor(**p)),
            lambda t: {"n_neighbors": t.suggest_int("n_neighbors", 2, 30),
                       "weights": t.suggest_categorical("weights", ["uniform", "distance"])}),
    "svr": (lambda p: _scaled_target(SVR(**p)),
            lambda t: {"C": t.suggest_float("C", 0.1, 300, log=True),
                       "gamma": t.suggest_float("gamma", 1e-3, 1.0, log=True),
                       "epsilon": t.suggest_float("epsilon", 0.01, 0.5, log=True)}),
    "gaussian_process": (_gp, None),
    "random_forest": (lambda p: RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1, **p),
                      lambda t: {"max_features": t.suggest_float("max_features", 0.2, 1.0),
                                 "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
                                 "max_depth": t.suggest_categorical("max_depth", [None, 8, 16])}),
    "xgboost": (lambda p: XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0, **p),
                lambda t: {**_boosting_space(t),
                           "max_depth": t.suggest_int("max_depth", 2, 8),
                           "min_child_weight": t.suggest_int("min_leaf", 1, 40, log=True)}),
    "lightgbm": (lambda p: lgb.LGBMRegressor(random_state=SEED, n_jobs=1, verbosity=-1, subsample_freq=1, **p),
                 lambda t: {**_boosting_space(t),
                            "max_depth": t.suggest_int("max_depth", 2, 8),
                            "num_leaves": t.suggest_int("num_leaves", 4, 64, log=True),
                            "min_child_samples": t.suggest_int("min_leaf", 1, 40, log=True)}),
}

# Untuned settings (library defaults with 500-600 trees), used for the "default" rows, B1 and the
# leakage and transfer checks
DEFAULTS = {"ridge": {"alpha": 1.0}, "knn": {"n_neighbors": 10, "weights": "uniform"}, "random_forest": {},
            "xgboost": {"n_estimators": 600, "learning_rate": 0.05},
            "lightgbm": {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31}}
INNER_FOLDS = 5


def build(model: str, params: dict, n_features: int):
    estimator = MODELS[model][0](params)
    if model == "gaussian_process":
        kernel = estimator.regressor.steps[-1][1].kernel
        kernel.k1.k2.length_scale = np.ones(n_features)  # one length scale per feature (ARD)
    return estimator


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------

def tune(model: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_trials: int) -> dict:
    """Optuna search on the outer-training cases, scored by grouped inner-fold R²."""
    space = MODELS[model][1]
    if space is None:
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # runs inside joblib workers
    inner = list(GroupKFold(n_splits=INNER_FOLDS).split(X, y, groups))

    def objective(trial):
        params = space(trial)
        scores = []
        for tr, va in inner:
            pred = build(model, params, X.shape[1]).fit(X[tr], y[tr]).predict(X[va])
            scores.append(1 - np.sum((y[va] - pred) ** 2) / np.sum((y[va] - y[va].mean()) ** 2))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    params = dict(study.best_params)
    if "min_leaf" in params:  # shared name in the search space, library-specific name in the model
        params["min_child_weight" if model == "xgboost" else "min_child_samples"] = params.pop("min_leaf")
    return params


def _outer_fold(model, X, y, groups, folds, k, n_trials):
    tr, va = folds != k, folds == k
    params = tune(model, X[tr], y[tr], groups[tr], n_trials)
    fitted = build(model, params, X.shape[1]).fit(X[tr], y[tr])
    return k, params, fitted, fitted.predict(X[va])


def nested_cv(df: pd.DataFrame, features: list[str], target: str, model: str, n_trials: int) -> dict:
    """Out-of-fold predictions, tuned parameters and fitted models for each outer fold."""
    X, y = df[features].to_numpy(float), df[target].to_numpy(float)
    groups, folds = df["split_group"].to_numpy(), df["outer_fold"].to_numpy()
    results = Parallel(n_jobs=5)(delayed(_outer_fold)(model, X, y, groups, folds, k, n_trials)
                                 for k in sorted(set(folds)))
    oof = np.full(len(y), np.nan)
    params, fitted = {}, {}
    for k, p, m, pred in results:
        oof[folds == k] = pred
        params[int(k)], fitted[int(k)] = p, m
    return {"oof": oof, "params": params, "models": fitted}


def fixed_cv(df, features, target, model, splitter) -> np.ndarray:
    """Out-of-fold predictions of an untuned model for an arbitrary splitter."""
    X, y = df[features].to_numpy(float), df[target].to_numpy(float)
    oof = np.full(len(y), np.nan)
    for tr, va in splitter:
        oof[va] = build(model, DEFAULTS.get(model, {}), X.shape[1]).fit(X[tr], y[tr]).predict(X[va])
    return oof


def grouped_splits(df):
    folds = df["outer_fold"].to_numpy()
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in sorted(set(folds))]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def r2(y, p) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def summarise(df: pd.DataFrame, target: str, model: str, oof: np.ndarray, label: str = "full") -> dict:
    y = df[target].to_numpy(float)
    overall = regression_metrics(y, oof, target)
    row = {"model": model, "features": label, "target": target, "n": overall["n"], "r2": overall["r2"],
           "mae": overall["mae"], "rmse": overall["rmse"]}
    for subset in ("idealized", "realistic"):
        m = (df["subset"] == subset).to_numpy()
        row[f"r2_{subset}"] = r2(y[m], oof[m])
        row[f"mae_{subset}"] = float(np.mean(np.abs(y[m] - oof[m])))
    keep = ~df["flagged"].to_numpy()
    row["r2_without_flagged"] = r2(y[keep], oof[keep])
    fold_r2 = [r2(y[df["outer_fold"] == k], oof[df["outer_fold"] == k]) for k in sorted(df["outer_fold"].unique())]
    row["fold_r2_mean"], row["fold_r2_std"] = float(np.mean(fold_r2)), float(np.std(fold_r2))
    return row


def group_bootstrap(df, target, pred_a, pred_b, n_boot=2000) -> dict:
    """R²(a) - R²(b) with a 95% interval from resampling whole split groups."""
    rng = np.random.default_rng(SEED)
    y = df[target].to_numpy(float)
    idx_by_group = [np.where(df["split_group"].to_numpy() == g)[0] for g in df["split_group"].unique()]
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_group[i] for i in rng.integers(0, len(idx_by_group), len(idx_by_group))])
        diffs.append(r2(y[pick], pred_a[pick]) - r2(y[pick], pred_b[pick]))
    diffs = np.array(diffs)
    return {"delta_r2": r2(y, pred_a) - r2(y, pred_b), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_le_0": float(np.mean(diffs <= 0))}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_ladder(table: pd.DataFrame) -> None:
    sub = table[(table["features"] == "full") & (table["target"] == "U_mean")].sort_values("r2")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(sub))
    ax.barh(y - 0.27, sub["r2"], 0.27, label="all cases", color="#4c72b0")
    ax.barh(y, sub["r2_idealized"], 0.27, label="idealized", color="#55a868")
    ax.barh(y + 0.27, sub["r2_realistic"], 0.27, label="realistic", color="#c44e52")
    ax.set_yticks(y, sub["model"])
    ax.set_xlim(min(0, sub[["r2", "r2_realistic"]].min().min()) - 0.05, 1)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("out-of-fold R² (grouped, nested CV)")
    ax.set_title("Case-level U_mean: model comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "model_ladder_u_mean.png", dpi=150)
    plt.close(fig)


def plot_targets(table: pd.DataFrame) -> None:
    sub = table[(table["features"] == "full") & table["model"].isin(["ridge", "svr", "gaussian_process", *TREE_MODELS])]
    pivot = sub.pivot(index="model", columns="target", values="r2")[TARGETS]
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("out-of-fold R²")
    ax.set_ylim(0, 1)
    ax.set_title("All targets, full feature set")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "model_ladder_all_targets.png", dpi=150)
    plt.close(fig)


def plot_parity(df, oof, model, target="U_mean") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for subset, color in (("idealized", "#55a868"), ("realistic", "#c44e52")):
        m = (df["subset"] == subset).to_numpy()
        ax.scatter(df.loc[m, target], oof[m], s=10, alpha=0.6, color=color, label=subset)
    lims = [df[target].min() * 0.95, df[target].max() * 1.05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(f"true {target} (U/u*)")
    ax.set_ylabel(f"predicted {target} (U/u*)")
    ax.set_title(f"{model}, held-out folds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"parity_{target}.png", dpi=150)
    plt.close(fig)


def plot_leakage(leak: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(leak))
    ax.bar(x - 0.2, leak["r2_random_kfold"], 0.4, label="random KFold", color="#8c8c8c")
    ax.bar(x + 0.2, leak["r2_grouped"], 0.4, label="grouped by city / layout family", color="#4c72b0")
    for xi, (a, b) in enumerate(zip(leak["r2_random_kfold"], leak["r2_grouped"], strict=True)):
        ax.text(xi, max(a, b) + 0.02, f"-{a - b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, leak["model"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("U_mean R² (realistic cases)")
    ax.set_ylim(min(0.0, leak[["r2_random_kfold", "r2_grouped"]].min().min()) - 0.1, 1.1)
    ax.set_title("Optimism of random splits on unseen cities")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "leakage_random_vs_grouped.png", dpi=150)
    plt.close(fig)


def plot_ablation(abl: pd.DataFrame) -> None:
    pivot = abl.pivot(index="features", columns="model", values="r2_realistic").loc[list(FEATURE_SETS)]
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("U_mean R², realistic cases")
    ax.set_title("Feature ablation (nested CV)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG / "feature_ablation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_preds(df, model, oofs: dict[str, np.ndarray]) -> pathlib.Path:
    rows = [pd.DataFrame({"case": df["case"], "target": t, "y_true": df[t], "y_pred": oofs[t], "subset": df["subset"],
                          "split_group": df["split_group"], "outer_fold": df["outer_fold"]}) for t in TARGETS]
    pred_dir = PREDS / f"s06c_{model}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(pred_dir / "case_preds.csv", index=False)
    return pred_dir


def shap_on_held_out(df, features, fitted: dict) -> pd.DataFrame:
    """TreeSHAP of each outer-fold model on its own held-out cases."""
    parts, rows = [], []
    for k, model in fitted.items():
        m = (df["outer_fold"] == k).to_numpy()
        X = df.loc[m, features]
        parts.append(pd.DataFrame(shap.TreeExplainer(model).shap_values(X), columns=features, index=X.index))
    values = pd.concat(parts).loc[df.index]
    shap.summary_plot(values.to_numpy(), df[features], max_display=15, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.title("TreeSHAP on held-out folds (U_mean)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_u_mean.png", dpi=150)
    plt.close("all")
    for f in features:
        rows.append({"feature": f, "mean_abs_shap": float(values[f].abs().mean())})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def main(n_trials: int) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_dev()
    print(f"dev cases: {len(df)} ({(df.subset == 'idealized').sum()} idealized, {(df.subset == 'realistic').sum()} "
          f"realistic), {df.split_group.nunique()} groups, {df.flagged.sum()} flagged")
    rows, params, oof = [], {}, {}

    # 1. Model ladder on the full feature set
    for model in MODELS:
        t0 = time.time()
        for target in TARGETS:
            res = nested_cv(df, FULL, target, model, n_trials)
            oof[(model, "full", target)] = res["oof"]
            params[f"{model}/{target}"] = res["params"]
            rows.append(summarise(df, target, model, res["oof"]))
            if model == "lightgbm" and target == "U_mean":
                lgbm_models = res["models"]
            if model == "xgboost" and target == "U_mean":
                xgb_models = res["models"]
        print(f"{model:17s} " + "  ".join(f"{r['target']} {r['r2']:.3f}" for r in rows[-4:])
              + f"  [{time.time() - t0:.0f}s]")
        save_preds(df, model, {t: oof[(model, "full", t)] for t in TARGETS})

    # 2. Feature ablation for the tree ensembles (U_mean)
    for model in TREE_MODELS:
        for label in ("basic", "morphology"):
            res = nested_cv(df, FEATURE_SETS[label], "U_mean", model, n_trials)
            oof[(model, label, "U_mean")] = res["oof"]
            rows.append(summarise(df, "U_mean", model, res["oof"], label))

    # 2b. The tree ensembles with fixed default settings, to show what tuning adds
    for model in TREE_MODELS:
        for target in TARGETS:
            p = fixed_cv(df, FULL, target, model, grouped_splits(df))
            oof[(f"{model}_default", "full", target)] = p
            rows.append(summarise(df, target, f"{model}_default", p))

    # 3. B1: untuned random forest on the basic features
    for target in TARGETS:
        p = fixed_cv(df, BASIC, target, "random_forest", grouped_splits(df))
        oof[("B1", "basic", target)] = p
        rows.append(summarise(df, target, "B1_rf_basic_untuned", p, "basic"))

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "model_comparison.csv", index=False, float_format="%.4f")
    (OUT / "best_params.json").write_text(json.dumps(params, indent=1, default=str))
    oof_long = pd.concat([pd.DataFrame({"case": df["case"], "subset": df["subset"], "split_group": df["split_group"],
                                        "outer_fold": df["outer_fold"], "model": m, "features": f, "target": t,
                                        "y_true": df[t], "y_pred": p}) for (m, f, t), p in oof.items()])
    oof_long.to_csv(OUT / "oof_predictions.csv", index=False, float_format="%.5g")

    # 4. Leakage: random KFold against grouped folds, untuned models, scored on realistic cases
    leak_rows = []
    real = (df["subset"] == "realistic").to_numpy()
    for model in ("ridge", "knn", "random_forest", "lightgbm"):
        grouped = fixed_cv(df, FULL, "U_mean", model, grouped_splits(df))
        rand = np.mean([r2(df["U_mean"][real], fixed_cv(df, FULL, "U_mean", model,
                                                          KFold(5, shuffle=True, random_state=s).split(df))[real])
                        for s in range(5)])
        leak_rows.append({"model": model, "r2_random_kfold": rand, "r2_grouped": r2(df["U_mean"][real], grouped[real])})
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(OUT / "leakage_random_vs_grouped.csv", index=False, float_format="%.4f")

    # 5. Transfer between subsets (untuned LightGBM, U_mean)
    trans_rows = []
    for train, test in (("idealized", "realistic"), ("realistic", "idealized")):
        tr, te = (df["subset"] == train).to_numpy(), (df["subset"] == test).to_numpy()
        model = build("lightgbm", DEFAULTS["lightgbm"], len(FULL)).fit(df.loc[tr, FULL], df.loc[tr, "U_mean"])
        trans_rows.append({"train": train, "test": test,
                           "r2_transfer": r2(df.loc[te, "U_mean"], model.predict(df.loc[te, FULL])),
                           "r2_within_subset_cv": table.query("model == 'lightgbm' and features == 'full' and "
                                                              "target == 'U_mean'")[f"r2_{test}"].item()})
    pd.DataFrame(trans_rows).to_csv(OUT / "transfer.csv", index=False, float_format="%.4f")

    # 6. Group bootstrap of R² differences (U_mean and U_p95)
    boot_rows = []
    for target in ("U_mean", "U_p95"):
        pairs = [("xgboost", "B1"), ("lightgbm", "B1"), ("random_forest", "B1"), ("gaussian_process", "B1"),
                 ("xgboost", "lightgbm"), ("xgboost", "random_forest"), ("lightgbm", "random_forest"),
                 ("xgboost", "gaussian_process"), ("xgboost_default", "lightgbm_default")]
        for a, b in pairs:
            pa = oof[(a, "full", target)]
            pb = oof[("B1", "basic", target)] if b == "B1" else oof[(b, "full", target)]
            boot_rows.append({"target": target, "model_a": a, "model_b": b, **group_bootstrap(df, target, pa, pb)})
    pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap_r2_differences.csv", index=False, float_format="%.4f")

    # 7. SHAP for the better of XGBoost and LightGBM on U_mean
    best = max(("xgboost", "lightgbm"), key=lambda m: r2(df["U_mean"], oof[(m, "full", "U_mean")]))
    importance = shap_on_held_out(df, FULL, xgb_models if best == "xgboost" else lgbm_models)
    importance.insert(0, "model", best)
    importance.to_csv(OUT / "shap_importance_u_mean.csv", index=False, float_format="%.4f")

    plot_ladder(table)
    plot_targets(table)
    plot_parity(df, oof[(best, "full", "U_mean")], best)
    plot_parity(df, oof[(best, "full", "U_p95")], best, "U_p95")
    plot_leakage(leak)
    plot_ablation(table[table["model"].isin(TREE_MODELS) & (table["target"] == "U_mean")])
    print(table.query("features == 'full' and target == 'U_mean'")[["model", "r2", "r2_idealized", "r2_realistic"]]
          .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per model, outer fold and target")
    main(parser.parse_args().trials)
