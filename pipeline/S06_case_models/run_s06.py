"""
S06 Case-Level Models -- Urban Microclimate Surrogate
Stage: S06

Runs:
  1. B1 baseline reproduction (RF + basic features)
  2. Full model ladder (Ridge, Ridge-deg2, kNN, RF, LightGBM)
  3. Feature-set ablation (Basic / Morph / Physics)
  4. Multi-target training (U_mean, U_p95, U_std, TKE_mean)
  5. Optuna hyperparameter tuning (60 trials, inner 3-fold)
  6. Monotonic-constraint experiment
  7. Transfer tests (idealized <-> realistic)
  8. Per-subset reporting
  9. Wilcoxon significance test vs B1
  10. Feature importance
  11. 5 figures
  12. Contract-format predictions -> results/preds/s06_lgbm_tuned_physics_v1/case_preds.csv
  13. Scoring via urbanwind.evaluate -> results/runs.csv
  14. Gate report -> pipeline/S06_case_models/outputs/S06_gate_report.md

Usage:
    python3 pipeline/S06_case_models/run_s06.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "src"))

FEATURES_PATH = REPO / "pipeline" / "S05_feature_engineering" / "outputs" / "case_features.csv"
TARGET_TABLE  = REPO / "pipeline" / "S04_eda" / "S04g_combined_report" / "master_case_table.csv"
SPLITS_PATH   = REPO / "data" / "splits.json"
OUT_DIR       = REPO / "pipeline" / "S06_case_models" / "outputs"
FIG_DIR       = OUT_DIR / "figures"
PRED_BASE     = REPO / "results" / "preds"

for d in [OUT_DIR, FIG_DIR, PRED_BASE]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("S06 Case-Level Models")
print("=" * 60)

with open(SPLITS_PATH) as f:
    splits = json.load(f)

fold_map    = splits["folds"]
test_groups = set(splits["test_groups"])
N_FOLDS     = int(splits["n_folds"])

feat_df = pd.read_csv(FEATURES_PATH)
tgt_df  = pd.read_csv(TARGET_TABLE)
tgt_df  = tgt_df[tgt_df["is_model_ready_spatial"].eq(True)].copy()

feat_df["fold"]    = feat_df["split_group"].map({k: int(v) for k, v in fold_map.items()})
feat_df["is_test"] = feat_df["split_group"].isin(test_groups)

dev_df = feat_df[~feat_df["is_test"]].copy()

RENAME = {
    "u_ped_mean": "U_mean",
    "u_p95":      "U_p95",
    "u_ped_std":  "U_std",
    "tke_mean":   "TKE_mean",
}
full_df = dev_df.merge(
    tgt_df[["case"] + list(RENAME.keys())].rename(columns=RENAME),
    on="case", how="inner"
)
full_df["subset_idealized"] = (full_df["subset"] == "idealized").astype(float)

print(f"Dev cases: {len(full_df)}")
print(f"  idealized: {(full_df['subset']=='idealized').sum()}")
print(f"  realistic: {(full_df['subset']=='realistic').sum()}")
print(f"Folds: {sorted(full_df['fold'].unique())}")
print(f"Frozen test groups: {len(test_groups)} (NOT TOUCHED)")

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
BASIC = [
    "lambda_p", "mean_h", "std_h", "max_h",
    "forcing_dir_deg", "lambda_f_dir", "wind_cos", "wind_sin",
    "subset_idealized",
]
MORPH = BASIC + [
    "lambda_w", "skew_h", "cv_h", "max_to_mean_h",
    "volume_per_area_m", "components_per_hectare",
    "mean_footprint_area_m2", "gamma_h",
]
PHYSICS = MORPH + [
    "lambda_f_x", "lambda_f_y",
    "d_over_h_macdonald", "z0_over_h_macdonald",
    "d_over_h_kanda", "z0_over_h_kanda",
    "upwind_fetch_mean_m", "upwind_fetch_median_m",
    "downwind_fetch_mean_m", "lateral_fetch_mean_m",
    "unobstructed_fraction_gt50m",
    "dist_mean_m", "dist_median_m", "dist_p90_m",
    "facet_orientation_entropy", "windward_wall_share",
    "relative_wind_facet_angle_deg",
]

def valid(fset):
    seen = set()
    out = []
    for f in fset:
        if f not in seen and f in full_df.columns:
            seen.add(f)
            out.append(f)
    return out

BASIC   = valid(BASIC)
MORPH   = valid(MORPH)
PHYSICS = valid(PHYSICS)
print(f"\nFeatures -- Basic: {len(BASIC)}, Morph: {len(MORPH)}, Physics: {len(PHYSICS)}")

# ---------------------------------------------------------------------------
# CV harness
# ---------------------------------------------------------------------------
def grouped_cv(model_fn, X, y, fold_col, log_target=False):
    oof  = np.full(len(y), np.nan)
    r2s  = []
    for k in range(N_FOLDS):
        vm = fold_col == k
        tm = ~vm
        y_fit = np.log1p(y[tm]) if log_target else y[tm]
        m = model_fn()
        m.fit(X[tm], y_fit)
        p = m.predict(X[vm])
        if log_target:
            p = np.expm1(p)
        oof[vm] = p
        r2s.append(r2_score(y[vm], p))
    return r2s, oof

folds   = full_df["fold"].values.astype(int)
TARGETS = ["U_mean", "U_p95", "U_std", "TKE_mean"]

# ---------------------------------------------------------------------------
# STEP 1: B1 baseline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1 -- B1 Baseline: RF + Basic Features")
print("=" * 60)

X_basic = full_df[BASIC].fillna(0).values
y_umean = full_df["U_mean"].values

b1_r2s, b1_oof = grouped_cv(
    lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    X_basic, y_umean, folds
)
b1_mean = float(np.mean(b1_r2s))
print(f"B1 fold R2: {[round(r, 4) for r in b1_r2s]}")
print(f"B1 mean R2: {b1_mean:.4f}  (reference ~0.671)")

# ---------------------------------------------------------------------------
# STEP 2: Model ladder
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 -- Model Ladder (Physics Features)")
print("=" * 60)

import lightgbm as lgb

X_phys = full_df[PHYSICS].fillna(0).values

MODELS = {
    "Ridge":      lambda: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Ridge_deg2": lambda: Pipeline([
        ("sc", StandardScaler()),
        ("pf", PolynomialFeatures(2, include_bias=False)),
        ("m",  Ridge(alpha=1.0)),
    ]),
    "kNN":       lambda: Pipeline([("sc", StandardScaler()),
                                    ("m", KNeighborsRegressor(n_neighbors=10))]),
    "RF":        lambda: RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    "LightGBM":  lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                             num_leaves=63, random_state=42,
                                             n_jobs=-1, verbosity=-1),
}

ladder_results = {}
for name, fn in MODELS.items():
    r2s, oof = grouped_cv(fn, X_phys, y_umean, folds)
    ladder_results[name] = {"r2s": r2s, "oof": oof}
    print(f"  {name:12s}: mean R2={np.mean(r2s):.4f}  {[round(r,4) for r in r2s]}")

# ---------------------------------------------------------------------------
# STEP 3: Feature-set ablation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 -- Feature-Set Ablation (LightGBM)")
print("=" * 60)

ablation_rows = []
for fname, fset in [("Basic", BASIC), ("Morph", MORPH), ("Physics", PHYSICS)]:
    X_ab = full_df[fset].fillna(0).values
    r2s, _ = grouped_cv(
        lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                   random_state=42, n_jobs=-1, verbosity=-1),
        X_ab, y_umean, folds
    )
    ablation_rows.append({"feature_set": fname, "n_features": len(fset),
                           "mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s))})
    print(f"  {fname:8s}: mean R2={np.mean(r2s):.4f}")
    for subset in ["idealized", "realistic"]:
        mask = (full_df["subset"] == subset).values
        if mask.sum() < 10:
            continue
        r2s_sub, _ = grouped_cv(
            lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                       random_state=42, n_jobs=-1, verbosity=-1),
            full_df.loc[mask, fset].fillna(0).values,
            full_df.loc[mask, "U_mean"].values,
            folds[mask]
        )
        print(f"    {fname} on {subset:12s}: mean R2={np.mean(r2s_sub):.4f}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(OUT_DIR / "ablation_table.csv", index=False)
print(f"Saved: {OUT_DIR / 'ablation_table.csv'}")

# ---------------------------------------------------------------------------
# STEP 4: Multi-target training
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 -- Multi-Target LightGBM")
print("=" * 60)

lgbm_results = {}
fold_r2_rows = []
for tgt in TARGETS:
    y_t = full_df[tgt].values
    r2s, oof = grouped_cv(
        lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                   num_leaves=63, random_state=42,
                                   n_jobs=-1, verbosity=-1),
        X_phys, y_t, folds
    )
    lgbm_results[tgt] = {"r2s": r2s, "oof": oof}
    print(f"  {tgt:10s}: mean R2={np.mean(r2s):.4f}  {[round(r,4) for r in r2s]}")
    for k, r2 in enumerate(r2s):
        fold_r2_rows.append({"target": tgt, "model": "LightGBM_physics", "fold": k, "r2": r2})

fold_r2_df = pd.DataFrame(fold_r2_rows)
fold_r2_df.to_csv(OUT_DIR / "fold_r2_summary.csv", index=False)
print(f"Saved: {OUT_DIR / 'fold_r2_summary.csv'}")

# ---------------------------------------------------------------------------
# STEP 5: Optuna tuning
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 -- Optuna Tuning (60 trials, U_mean)")
print("=" * 60)

import optuna
from sklearn.model_selection import GroupKFold, cross_val_score

def tune_lgbm(X, y, fold_arr, n_trials=60):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbosity": -1,
        }
        gkf = GroupKFold(n_splits=3)
        scores = cross_val_score(
            lgb.LGBMRegressor(**params), X, y,
            cv=gkf, groups=fold_arr, scoring="r2", n_jobs=1
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

best_params, best_inner_r2 = tune_lgbm(X_phys, y_umean, folds, n_trials=60)
print(f"Best inner CV R2: {best_inner_r2:.4f}")
print(f"Best params: {best_params}")

# Evaluate tuned on outer folds
tuned_r2s, tuned_oof = grouped_cv(
    lambda: lgb.LGBMRegressor(**{**best_params, "random_state": 42, "n_jobs": -1, "verbosity": -1}),
    X_phys, y_umean, folds
)
tuned_up95_r2s, tuned_up95_oof = grouped_cv(
    lambda: lgb.LGBMRegressor(**{**best_params, "random_state": 42, "n_jobs": -1, "verbosity": -1}),
    X_phys, full_df["U_p95"].values, folds
)
print(f"\nTuned LightGBM U_mean fold R2: {[round(r,4) for r in tuned_r2s]}")
print(f"Tuned LightGBM U_mean mean R2: {np.mean(tuned_r2s):.4f}")
print(f"Tuned LightGBM U_p95  mean R2: {np.mean(tuned_up95_r2s):.4f}")

# ---------------------------------------------------------------------------
# STEP 6: Monotonic constraints
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6 -- Monotonic Constraints Experiment")
print("=" * 60)

lp_idx = PHYSICS.index("lambda_p") if "lambda_p" in PHYSICS else -1
if lp_idx >= 0:
    mono = [0] * len(PHYSICS)
    mono[lp_idx] = -1
    mono_params = {**best_params, "monotone_constraints": mono,
                   "random_state": 42, "n_jobs": -1, "verbosity": -1}
    mono_r2s, _ = grouped_cv(
        lambda: lgb.LGBMRegressor(**mono_params),
        X_phys, y_umean, folds
    )
    print(f"  Monotonic R2: {np.mean(mono_r2s):.4f}  {[round(r,4) for r in mono_r2s]}")
else:
    mono_r2s = tuned_r2s
    print("  lambda_p not in PHYSICS list, skipping")

# ---------------------------------------------------------------------------
# STEP 7: Transfer tests
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 -- Transfer Tests")
print("=" * 60)

ideal_mask = (full_df["subset"] == "idealized").values
real_mask  = (full_df["subset"] == "realistic").values

m_i = lgb.LGBMRegressor(**{**best_params, "random_state": 42, "n_jobs": -1, "verbosity": -1})
m_i.fit(X_phys[ideal_mask], y_umean[ideal_mask])
r2_i2r = float(r2_score(y_umean[real_mask], m_i.predict(X_phys[real_mask])))
print(f"  idealized -> realistic: R2={r2_i2r:.4f}")

m_r = lgb.LGBMRegressor(**{**best_params, "random_state": 42, "n_jobs": -1, "verbosity": -1})
m_r.fit(X_phys[real_mask], y_umean[real_mask])
r2_r2i = float(r2_score(y_umean[ideal_mask], m_r.predict(X_phys[ideal_mask])))
print(f"  realistic -> idealized: R2={r2_r2i:.4f}")

# ---------------------------------------------------------------------------
# STEP 8: Per-subset reporting
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8 -- Per-Subset R2 (LightGBM default, OOF)")
print("=" * 60)

for tgt in TARGETS:
    y_t  = full_df[tgt].values
    oof  = lgbm_results[tgt]["oof"]
    r2_i = float(r2_score(y_t[ideal_mask], oof[ideal_mask]))
    r2_r = float(r2_score(y_t[real_mask],  oof[real_mask]))
    print(f"  {tgt:10s}  idealized={r2_i:.4f}  realistic={r2_r:.4f}")

# ---------------------------------------------------------------------------
# STEP 9: Wilcoxon test
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 9 -- Wilcoxon Test (tuned vs B1)")
print("=" * 60)

stat, pval = wilcoxon(tuned_r2s, b1_r2s, alternative="greater")
sig = bool(pval < 0.05)
print(f"  stat={stat:.2f}  p={pval:.4f}  significant={sig}")

# ---------------------------------------------------------------------------
# STEP 10: Feature importance
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 10 -- Feature Importance")
print("=" * 60)

final_model = lgb.LGBMRegressor(**{**best_params, "random_state": 42, "n_jobs": -1, "verbosity": -1})
final_model.fit(X_phys, y_umean)
importances = pd.Series(final_model.feature_importances_, index=PHYSICS).sort_values(ascending=False)
top15 = importances.head(15)
print(top15.to_string())

# ---------------------------------------------------------------------------
# STEP 11: Figures
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 11 -- Figures")
print("=" * 60)

NORD0  = "#2e3440"
NORD4  = "#d8dee9"
NORD9  = "#81a1c1"
NORD11 = "#bf616a"
NORD14 = "#a3be8c"
NORD13 = "#ebcb8b"

plt.rcParams.update({
    "figure.facecolor": NORD0, "axes.facecolor": "#3b4252",
    "axes.edgecolor": NORD4, "axes.labelcolor": NORD4,
    "xtick.color": NORD4, "ytick.color": NORD4,
    "text.color": NORD4, "grid.color": "#434c5e",
    "grid.linestyle": "--", "grid.alpha": 0.4,
})

# Fig 1: Model ladder
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(ladder_results.keys()) + ["Tuned LightGBM"]
means = [np.mean(ladder_results[n]["r2s"]) for n in ladder_results] + [np.mean(tuned_r2s)]
stds  = [np.std(ladder_results[n]["r2s"])  for n in ladder_results] + [np.std(tuned_r2s)]
colors = [NORD9] * len(ladder_results) + [NORD11]
ax.bar(model_names, means, yerr=stds, color=colors, capsize=5, alpha=0.88, edgecolor=NORD0)
ax.axhline(b1_mean, color=NORD13, linestyle="--", linewidth=1.5, label=f"B1 baseline ({b1_mean:.3f})")
ax.axhline(0.867,   color=NORD14, linestyle="--", linewidth=1.5, label="B2 target (0.867)")
ax.set_ylabel("Grouped 5-fold CV R2  (U_mean)")
ax.set_title("S06 -- Model Ladder: U_mean Prediction")
ax.legend(fontsize=9); ax.set_ylim(0, 1); ax.grid(axis="y")
plt.xticks(rotation=18, ha="right")
plt.tight_layout()
fig.savefig(FIG_DIR / "model_ladder.png", dpi=150)
plt.close()
print(f"  {FIG_DIR / 'model_ladder.png'}")

# Fig 2: Feature ablation
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(ablation_df["feature_set"], ablation_df["mean_r2"],
       yerr=ablation_df["std_r2"],
       color=[NORD9, "#81a1c1", NORD11], capsize=5, alpha=0.88, edgecolor=NORD0)
ax.axhline(b1_mean, color=NORD13, linestyle="--", linewidth=1.5, label=f"B1 ({b1_mean:.3f})")
ax.set_ylabel("Grouped 5-fold CV R2"); ax.set_title("S06 -- Feature-Set Ablation (U_mean)")
ax.legend(fontsize=9); ax.set_ylim(0, 1); ax.grid(axis="y")
plt.tight_layout()
fig.savefig(FIG_DIR / "ablation_features.png", dpi=150)
plt.close()
print(f"  {FIG_DIR / 'ablation_features.png'}")

# Fig 3: True vs predicted
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, subset, color in zip(axes, ["idealized","realistic"], [NORD9, NORD11]):
    mask = (full_df["subset"] == subset).values
    yt = y_umean[mask]; yp = tuned_oof[mask]
    ax.scatter(yt, yp, alpha=0.45, s=18, color=color, edgecolors="none")
    lo = min(yt.min(), yp.min()) - 0.05
    hi = max(yt.max(), yp.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=NORD4, linewidth=1.2, linestyle="--")
    ax.set_xlabel("True U_mean (U/u*)"); ax.set_ylabel("Predicted")
    ax.set_title(f"{subset}  R2={r2_score(yt, yp):.4f}")
    ax.grid(True)
plt.suptitle("S06 -- True vs Predicted U_mean (Tuned LightGBM, OOF)")
plt.tight_layout()
fig.savefig(FIG_DIR / "true_vs_pred_umean.png", dpi=150)
plt.close()
print(f"  {FIG_DIR / 'true_vs_pred_umean.png'}")

# Fig 4: Feature importance
fig, ax = plt.subplots(figsize=(8, 5))
top15.plot.barh(ax=ax, color=NORD9, alpha=0.88, edgecolor=NORD0)
ax.invert_yaxis()
ax.set_xlabel("LightGBM Feature Importance (split)")
ax.set_title("S06 -- Top-15 Feature Importances (U_mean)")
ax.grid(axis="x")
plt.tight_layout()
fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
plt.close()
print(f"  {FIG_DIR / 'feature_importance.png'}")

# Fig 5: Multi-target fold R2
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(N_FOLDS)
width = 0.2
palette = [NORD9, NORD11, NORD14, NORD13]
for i, tgt in enumerate(TARGETS):
    ax.bar(x + i * width, lgbm_results[tgt]["r2s"], width,
           label=tgt, color=palette[i], alpha=0.85, edgecolor=NORD0)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"Fold {k}" for k in range(N_FOLDS)])
ax.set_ylabel("R2"); ax.set_title("S06 -- Multi-Target Fold R2 (LightGBM, Physics)")
ax.legend(fontsize=9); ax.set_ylim(0, 1); ax.grid(axis="y")
plt.tight_layout()
fig.savefig(FIG_DIR / "multitarget_fold_r2.png", dpi=150)
plt.close()
print(f"  {FIG_DIR / 'multitarget_fold_r2.png'}")

# ---------------------------------------------------------------------------
# STEP 12: Save contract predictions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 12 -- Contract-Format Predictions")
print("=" * 60)

EXP_ID   = "s06_lgbm_tuned_physics_v1"
PRED_DIR = PRED_BASE / EXP_ID
PRED_DIR.mkdir(parents=True, exist_ok=True)

records = []
for tgt in TARGETS:
    oof = lgbm_results[tgt]["oof"]
    y_t = full_df[tgt].values
    for i, case in enumerate(full_df["case"].values):
        records.append({
            "case":   case,
            "target": tgt,
            "y_true": float(y_t[i]),
            "y_pred": float(oof[i]),
        })

pred_csv = PRED_DIR / "case_preds.csv"
pd.DataFrame(records).to_csv(pred_csv, index=False)
print(f"Saved: {pred_csv}  ({len(records)} rows)")

# ---------------------------------------------------------------------------
# STEP 13: Score via evaluate.py
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 13 -- urbanwind.evaluate")
print("=" * 60)

config_str = (
    f"LightGBM+physics({len(PHYSICS)}feats)"
    f"+optuna60+n_est={best_params.get('n_estimators','?')}"
    f"+lr={best_params.get('learning_rate', 0.0):.4f}"
    f"+leaves={best_params.get('num_leaves','?')}"
)
res = subprocess.run(
    [sys.executable, "-m", "urbanwind.evaluate",
     str(PRED_DIR),
     "--exp-id", EXP_ID,
     "--stage",  "S06",
     "--config", config_str],
    capture_output=True, text=True, cwd=str(REPO)
)
print(res.stdout)
if res.returncode != 0:
    print("STDERR:", res.stderr[:600])

# ---------------------------------------------------------------------------
# STEP 14: Gate check & write report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 14 -- Gate Check & Report")
print("=" * 60)

B1_UMEAN = 0.671
B1_UP95  = 0.664
B2_UMEAN = 0.867
B2_UP95  = 0.791
B2_USTD  = 0.769
B2_TKE   = 0.941

umean_r2 = float(np.mean(tuned_r2s))
up95_r2  = float(np.mean(tuned_up95_r2s))
ustd_r2  = float(np.mean(lgbm_results["U_std"]["r2s"]))
tke_r2   = float(np.mean(lgbm_results["TKE_mean"]["r2s"]))

gate_umean  = umean_r2 > B1_UMEAN
gate_up95   = up95_r2  > B1_UP95
gate_wilcox = sig
gate_pass   = gate_umean and gate_up95 and gate_wilcox

def tick(b): return "PASS" if b else "FAIL"

print(f"  U_mean R2 = {umean_r2:.4f}  > {B1_UMEAN}  -> {tick(gate_umean)}")
print(f"  U_p95  R2 = {up95_r2:.4f}  > {B1_UP95}  -> {tick(gate_up95)}")
print(f"  Wilcoxon p = {pval:.4f}  < 0.05  -> {tick(gate_wilcox)}")
print(f"\n  OVERALL GATE: {'PASSED' if gate_pass else 'FAILED'}")

# Per-subset table strings
subset_rows = "\n".join(
    f"| {t} | "
    f"{r2_score(full_df[t].values[ideal_mask], lgbm_results[t]['oof'][ideal_mask]):.4f} | "
    f"{r2_score(full_df[t].values[real_mask],  lgbm_results[t]['oof'][real_mask]):.4f} |"
    for t in TARGETS
)

ladder_rows = "\n".join(
    f"| {n} | {np.mean(ladder_results[n]['r2s']):.4f} | "
    f"{[round(r,4) for r in ladder_results[n]['r2s']]} |"
    for n in ladder_results
)

abl_md = ablation_df.to_markdown(index=False)

top_imp_rows = "\n".join(
    f"| {feat} | {imp} |"
    for feat, imp in top15.items()
)

gate_md = f"""# Stage S06 Gate Report -- Case-Level Models

- **Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
- **Exp ID:** `{EXP_ID}`
- **Dev Cases:** {len(full_df)} (idealized: {int(ideal_mask.sum())}, realistic: {int(real_mask.sum())})
- **Frozen Test Groups:** {len(test_groups)} -- NOT OPENED

## Best Model: Tuned LightGBM + Physics Features ({len(PHYSICS)} features)

### Best Optuna Params (60 trials, inner 3-fold CV)

| Param | Value |
|---|---|
| n_estimators | {best_params.get('n_estimators')} |
| learning_rate | {best_params.get('learning_rate', 0):.6f} |
| num_leaves | {best_params.get('num_leaves')} |
| min_child_samples | {best_params.get('min_child_samples')} |
| subsample | {best_params.get('subsample', 0):.4f} |
| colsample_bytree | {best_params.get('colsample_bytree', 0):.4f} |
| reg_alpha | {best_params.get('reg_alpha', 0):.6f} |
| reg_lambda | {best_params.get('reg_lambda', 0):.6f} |

### Gate Metrics

| Metric | Value | Threshold | Result |
|---|---|---|---|
| U_mean grouped CV R2 | {umean_r2:.4f} | > {B1_UMEAN} (B1) | {tick(gate_umean)} |
| U_p95  grouped CV R2 | {up95_r2:.4f}  | > {B1_UP95}  (B1) | {tick(gate_up95)} |
| Wilcoxon p-value     | {pval:.4f}    | < 0.05            | {tick(gate_wilcox)} |

### B2 Target Progress

| Target | Tuned LightGBM R2 | B2 Target | Status |
|---|---|---|---|
| U_mean   | {umean_r2:.4f} | {B2_UMEAN} | {tick(umean_r2 >= B2_UMEAN)} |
| U_p95    | {up95_r2:.4f}  | {B2_UP95}  | {tick(up95_r2  >= B2_UP95)} |
| U_std    | {ustd_r2:.4f}  | {B2_USTD}  | {tick(ustd_r2  >= B2_USTD)} |
| TKE_mean | {tke_r2:.4f}   | {B2_TKE}   | {tick(tke_r2   >= B2_TKE)} |

### Per-Subset R2 (LightGBM default, OOF)

| Target | Idealized | Realistic |
|---|---|---|
{subset_rows}

### Model Ladder (U_mean, Physics Features)

| Model | Mean R2 | Fold R2s |
|---|---|---|
| B1-RF+Basic (reference) | {b1_mean:.4f} | {[round(r,4) for r in b1_r2s]} |
{ladder_rows}
| **Tuned LightGBM** | **{umean_r2:.4f}** | {[round(r,4) for r in tuned_r2s]} |

### Feature-Set Ablation

{abl_md}

### Transfer Tests

| Direction | U_mean R2 |
|---|---|
| Idealized trained, tested on realistic | {r2_i2r:.4f} |
| Realistic trained, tested on idealized | {r2_r2i:.4f} |

### Monotonic Constraint (lambda_p: U falls as density rises)

Mean R2 with constraint: {np.mean(mono_r2s):.4f}  (tuned without: {umean_r2:.4f})

### Top-15 Feature Importances (U_mean)

| Feature | Importance |
|---|---|
{top_imp_rows}

## Stage Gate Result: {'PASSED' if gate_pass else 'FAILED'}
"""

gate_path = OUT_DIR / "S06_gate_report.md"
gate_path.write_text(gate_md, encoding="utf-8")
print(f"\nGate report -> {gate_path}")

print("\n" + "=" * 60)
print("S06 COMPLETE")
print(f"  Gate:       {'PASSED' if gate_pass else 'FAILED'}")
print(f"  U_mean R2 = {umean_r2:.4f}  |  U_p95 R2 = {up95_r2:.4f}")
print(f"  Outputs:    {OUT_DIR}")
print(f"  Preds:      {PRED_DIR}")
print("=" * 60)
