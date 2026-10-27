"""
S09: one-time scoring of the frozen test set with the final Phase 1 models.

Every modelling choice was fixed on the development folds (S06c, S07) before this
script was written. It refits each final model on all 482 development cases and
scores the 53 cases of the 24 frozen test groups once:
- case level: the S06c models, each re-tuned on the development cases only with
  the same nested-search settings (Optuna, grouped inner folds);
- field level: the S07 pixel model, the hybrid (pattern x the S09 prediction of the level
  model chosen in S07) and the level-only references.

Two test cases come from cities that also have development neighbourhoods
(AU-SURREYHILLS in Melbourne, FI-MS in Helsinki), so every score is also reported
without them.

The script refuses to run again once outputs/test_scored.json exists, so the test
set cannot be used to make choices. The guard is written before anything is scored. A
second scoring is possible only with --rescore and a reason (for example a bug fix that
changes no modelling choice); the guard then keeps the full scoring history.

Test-case targets are computed here, through the shared loader: S02 leaves the frozen
test cases out of its target table.

Usage: python pipeline/S09_phase1_evaluation/score_test_set.py [--rescore "reason"]   (about 25 minutes)
"""

from __future__ import annotations

import argparse
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
from urbanwind.data import EXCLUDE, case_targets  # noqa: E402
from urbanwind.metrics import case_metrics, regression_metrics  # noqa: E402
from urbanwind.splits import load_splits  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
GUARD = OUT / "test_scored.json"
SHARED_CITY = {"city-AU-SURREYHILLS", "city-FI-MS"}
CASE_MODELS = ["mean", "ridge", "knn", "svr", "gaussian_process", "random_forest", "xgboost", "lightgbm"]


def load_test() -> pd.DataFrame:
    splits = load_splits()
    feats = pd.read_csv(s06c.FEATURES_CSV)
    df = feats[~feats["case"].isin(EXCLUDE) & feats["split_group"].isin(splits["test_groups"])].copy()
    targets = pd.DataFrame([{"case": c, **case_targets(c)} for c in df["case"]])[["case", *s06c.TARGETS]]
    df = df.merge(targets, on="case", how="inner")
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
                         "sum_y2": (yc ** 2).sum(), "sse": ((yc - pred) ** 2).sum(),
                         **{k: v for k, v in case_metrics(yc, pred).items() if k != "case"}})
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


def write_guard(history: list[dict], status: str, **info) -> None:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    entry = {"date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "git_sha": sha,
             "status": status, **info}
    GUARD.write_text(json.dumps({**entry, "history": history}, indent=1))


def main(n_trials: int = 30, rescore: str | None = None) -> None:
    history = []
    if GUARD.exists():
        previous = json.loads(GUARD.read_text())
        if not rescore:
            sys.exit(f"The test set was already scored ({previous['date']}); refusing to score it again.")
        history = [*previous.pop("history", []), previous]
    OUT.mkdir(parents=True, exist_ok=True)
    reason = {"reason": rescore} if rescore else {}
    write_guard(history, "started", n_trials=n_trials, **reason)  # before any score exists
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
    level_model = json.loads((s07.OUT / "run_info.json").read_text())["level_model"]  # chosen in S07 on development folds
    level = case_preds[(case_preds["model"] == level_model) & (case_preds["target"] == "U_mean")].set_index("case")["y_pred"]
    dev_px = dev.copy()
    per_case = field_level(dev_px, test_px, level)
    per_case.drop(columns=["n_cells", "sum_y", "sum_y2", "sse"]).to_csv(OUT / "test_field_per_case.csv", index=False,
                                                                       float_format="%.5g")
    field_summary(per_case).to_csv(OUT / "test_field_level.csv", index=False, float_format="%.4f")

    write_guard(history, "scored", n_test_cases=len(test), n_trials=n_trials, **reason)
    print(case_table.query("target == 'U_mean' and cases == 'all'").to_string(index=False))
    print(field_summary(per_case).query("cases == 'all'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", help="score again after a fix that changes no modelling choice; give the reason")
    main(rescore=parser.parse_args().rescore)
