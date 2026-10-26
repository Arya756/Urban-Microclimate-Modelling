"""
S09: one-time scoring of the frozen test set with the final Phase 1 models.

Every modelling choice was fixed on the development folds (S06c, S07) before this
script was written. It refits each final model on all 482 development cases and
scores the 53 cases of the 24 frozen test groups once:
- case level: the S06c models, each re-tuned on the development cases only with
  the same nested-search settings (Optuna, grouped inner folds);
- field level: the S07 pixel model, the hybrid (pattern x S09 XGBoost level) and the
  level-only references.

Two test cases come from cities that also have development neighbourhoods
(AU-SURREYHILLS in Melbourne, FI-MS in Helsinki), so every score is also reported
without them.

The script refuses to run again once outputs/test_scored.json exists, so the test
set cannot be used to make choices.

Usage: python pipeline/S09_phase1_evaluation/score_test_set.py   (about 25 minutes)
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.S06_case_models import s06c_model_comparison as s06c  # noqa: E402
from pipeline.S07_pixel_model import s07_pixel_model as s07  # noqa: E402
from urbanwind.data import EXCLUDE  # noqa: E402
from urbanwind.metrics import case_metrics, regression_metrics  # noqa: E402
from urbanwind.splits import load_splits  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
GUARD = OUT / "test_scored.json"
SHARED_CITY = {"city-AU-SURREYHILLS", "city-FI-MS"}
CASE_MODELS = ["mean", "ridge", "knn", "svr", "gaussian_process", "random_forest", "xgboost", "lightgbm"]


def load_test() -> pd.DataFrame:
    splits = load_splits()
    feats = pd.read_csv(s06c.FEATURES_CSV)
    targets = pd.read_csv(s06c.TARGETS_CSV)[["case", *s06c.TARGETS]]
    df = feats.merge(targets, on="case", how="inner")
    df = df[~df["case"].isin(EXCLUDE) & df["split_group"].isin(splits["test_groups"])].copy()
    df["subset_idealized"] = (df["subset"] == "idealized").astype(float)
    df["shared_city"] = df["split_group"].isin(SHARED_CITY)
    return df.reset_index(drop=True)


def fit_predict(model: str, target: str, dev: pd.DataFrame, test: pd.DataFrame, n_trials: int) -> tuple:
    features = s06c.BASIC if model == "B1" else s06c.FULL
    X, y = dev[features].to_numpy(float), dev[target].to_numpy(float)
    if model == "B1":
        est = s06c.build("random_forest", {}, X.shape[1])
        params = {}
    else:
        params = s06c.tune(model, X, y, dev["split_group"].to_numpy(), n_trials)
        est = s06c.build(model, params, X.shape[1])
    pred = est.fit(X, y).predict(test[features].to_numpy(float))
    return model, target, params, pred


def score_rows(test: pd.DataFrame, target: str, model: str, pred: np.ndarray) -> list[dict]:
    rows = []
    y = test[target].to_numpy(float)
    for label, mask in (("all", np.ones(len(test), bool)), ("idealized", (test["subset"] == "idealized").to_numpy()),
                        ("realistic", (test["subset"] == "realistic").to_numpy()),
                        ("all without shared-city cases", ~test["shared_city"].to_numpy())):
        m = regression_metrics(y[mask], pred[mask], target)
        rows.append({"model": model, "target": target, "cases": label, "n": m["n"], "r2": m["r2"],
                     "mae": m["mae"], "rmse": m["rmse"]})
    return rows


def case_level(dev: pd.DataFrame, test: pd.DataFrame, n_trials: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    jobs = [(m, t) for m in [*CASE_MODELS, "B1"] for t in s06c.TARGETS]
    results = Parallel(n_jobs=8)(delayed(fit_predict)(m, t, dev, test, n_trials) for m, t in jobs)
    rows, params, preds = [], {}, []
    for model, target, p, pred in results:
        rows += score_rows(test, target, model, pred)
        params[f"{model}/{target}"] = p
        preds.append(pd.DataFrame({"case": test["case"], "target": target, "model": model, "y_true": test[target],
                                   "y_pred": pred, "subset": test["subset"], "split_group": test["split_group"]}))
    return pd.DataFrame(rows), params, pd.concat(preds)


def field_level(dev: pd.DataFrame, test: pd.DataFrame, level: pd.Series) -> pd.DataFrame:
    X, y, ratio, _, _ = s07.build_training_table(dev)
    base = s07.s05_columns(X.shape[1])
    direct_s05 = lgb.LGBMRegressor(**s07.PARAMS).fit(X[:, base], y)
    direct = lgb.LGBMRegressor(**s07.PARAMS).fit(X, y)
    pattern = lgb.LGBMRegressor(**s07.PARAMS).fit(X, ratio)
    rows = []
    for _, row in test.iterrows():
        Xc, yc, valid, _, _ = s07.case_pixels(row["case"], row)
        preds = {"pixel_gbm_s05": direct_s05.predict(Xc[:, base]), "pixel_gbm": direct.predict(Xc),
                 "hybrid_gbm": pattern.predict(Xc) * level[row["case"]],
                 "level_s09": np.full(len(yc), level[row["case"]]), "level_oracle": np.full(len(yc), yc.mean())}
        for variant, pred in preds.items():
            out = s07.PREDS / f"s09_{variant}"
            out.mkdir(parents=True, exist_ok=True)
            true_map = np.full(valid.shape, np.nan, np.float32)
            pred_map = np.full(valid.shape, np.nan, np.float32)
            true_map[valid], pred_map[valid] = yc, pred
            np.savez_compressed(out / f"{row['case']}.npz", y_true=true_map, y_pred=pred_map, mask=~valid)
            rows.append({"variant": variant, "case": row["case"], "subset": row["subset"],
                         "shared_city": row["shared_city"], "n_cells": len(yc), "sum_y": yc.sum(),
                         "sum_y2": (yc ** 2).sum(), "sse": ((yc - pred) ** 2).sum(), **case_metrics(yc, pred)})
    return pd.DataFrame(rows)


def field_summary(per_case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in per_case.groupby("variant", sort=False):
        for label, sub in (("all", g), ("idealized", g[g["subset"] == "idealized"]),
                           ("realistic", g[g["subset"] == "realistic"]), ("all without shared-city cases", g[~g["shared_city"]])):
            n, s, ss, sse = sub[["n_cells", "sum_y", "sum_y2", "sse"]].sum()
            rows.append({"variant": variant, "cases": label, "n_cases": len(sub), "pooled_r2": 1 - sse / (ss - s * s / n),
                         "median_within_case_r2": sub["r2"].median(), "mean_mae": sub["mae"].mean(),
                         "mean_delta_p95": sub["delta_p95"].mean(), "mean_level_err": sub["level_err"].mean(),
                         "mean_pattern_err": sub["pattern_err"].mean()})
    return pd.DataFrame(rows)


def main(n_trials: int = 30) -> None:
    if GUARD.exists():
        sys.exit(f"The test set was already scored ({json.loads(GUARD.read_text())['date']}); refusing to score it again.")
    OUT.mkdir(parents=True, exist_ok=True)
    dev, test = s06c.load_dev(), load_test()
    print(f"development cases: {len(dev)}, test cases: {len(test)} ({test['subset'].value_counts().to_dict()})")

    case_table, params, case_preds = case_level(dev, test, n_trials)
    case_table.to_csv(OUT / "test_case_level.csv", index=False, float_format="%.4f")
    (OUT / "test_case_level_params.json").write_text(json.dumps(params, indent=1, default=str))
    for model, g in case_preds.groupby("model"):
        pred_dir = s07.PREDS / f"s09_{model}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        g.drop(columns="model").to_csv(pred_dir / "case_preds.csv", index=False)

    test_px = test.copy()
    level = case_preds.query("model == 'xgboost' and target == 'U_mean'").set_index("case")["y_pred"]
    dev_px = dev.copy()
    per_case = field_level(dev_px, test_px, level)
    per_case.drop(columns=["n_cells", "sum_y", "sum_y2", "sse"]).to_csv(OUT / "test_field_per_case.csv", index=False,
                                                                       float_format="%.5g")
    field_summary(per_case).to_csv(OUT / "test_field_level.csv", index=False, float_format="%.4f")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    GUARD.write_text(json.dumps({"date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                                 "git_sha": sha, "n_test_cases": len(test), "n_trials": n_trials}, indent=1))
    print(case_table.query("target == 'U_mean' and cases == 'all'").to_string(index=False))
    print(field_summary(per_case).query("cases == 'all'").to_string(index=False))


if __name__ == "__main__":
    main()
