"""
S07 diagnostics: where does the pixel model's limit come from?

1. Split: the same model under the random 80/20 case split used by Lu et al. (2025),
   where cities can appear on both sides, scored with their per-case grid R².
   If it scores like the grouped folds, the split is not what limits the model.
2. Capacity: the model scored on cases it was trained on, with the fixed S07 setting
   and with a far less regularised one (HIGH_CAPACITY). If more capacity raises the
   in-sample score but not the held-out score, the extra fit is memorisation, and the
   features, not the model, limit what generalises to unseen cases.

Development cases only; the frozen test groups are never loaded.

Usage: python pipeline/S07_pixel_model/s07_diagnostics.py   (about 30 minutes)
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from s07_pixel_model import N_SAMPLE, OUT, PARAMS, SEED, case_pixels, dev_cases

N_INSAMPLE = 40  # training cases re-scored in full for the capacity check
HIGH_CAPACITY = {"n_estimators": 2000, "num_leaves": 1023, "min_child_samples": 5}
MODELS = {"fixed setting": PARAMS, "high capacity": {**PARAMS, **HIGH_CAPACITY}}


def per_case_r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def score(model, df: pd.DataFrame, full: dict, cases: set, label: dict) -> list[dict]:
    rows, acc = [], {}
    for i in sorted(cases):
        X, y = full[i]
        p = model.predict(X)
        subset = df.loc[i, "subset"]
        rows.append({**label, "case": df.loc[i, "case"], "subset": subset, "per_case_r2": per_case_r2(y, p)})
        for key in (subset, "all"):  # pooled sums per subset and over all cases
            acc[key] = acc.get(key, np.zeros(4)) + [len(y), y.sum(), (y ** 2).sum(), ((y - p) ** 2).sum()]
    for key, (n, s, ss, sse) in acc.items():
        rows.append({**label, "case": "ALL (pooled R²)", "subset": key, "per_case_r2": 1 - sse / (ss - s * s / n)})
    return rows


def main() -> None:
    df, _ = dev_cases()
    rng = np.random.default_rng(SEED)
    test = set(rng.choice(df.index, size=round(0.2 * len(df)), replace=False))
    insample = set(rng.choice([i for i in df.index if i not in test], size=N_INSAMPLE, replace=False))
    Xs, ys, full = [], [], {}
    for i, row in df.iterrows():
        X, y, *_ = case_pixels(row["case"], row)
        if i in test or i in insample:
            full[i] = (X, y)
        if i not in test:
            pick = rng.permutation(len(y))[:N_SAMPLE]
            Xs.append(X[pick])
            ys.append(y[pick])
    X, y = np.vstack(Xs), np.concatenate(ys)

    rows = []
    for name, params in MODELS.items():
        model = lgb.LGBMRegressor(**params).fit(X, y)
        for setting, cases in (("random 80/20 split, held-out cases", test), ("training cases (in-sample)", insample)):
            rows += score(model, df, full, cases, {"model": name, "setting": setting})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "diagnostics_per_case.csv", index=False, float_format="%.4f")
    cases = out[out["case"] != "ALL (pooled R²)"]
    keys = ["model", "setting"]
    summary = cases.groupby([*keys, "subset"])["per_case_r2"].agg(["count", "median"]).reset_index()
    overall = cases.groupby(keys)["per_case_r2"].agg(["count", "median"]).reset_index().assign(subset="all")
    pooled = out[out["case"] == "ALL (pooled R²)"][[*keys, "subset", "per_case_r2"]].rename(
        columns={"per_case_r2": "pooled_r2"})
    summary = pd.concat([summary, overall]).merge(pooled, on=[*keys, "subset"]).rename(
        columns={"count": "n_cases", "median": "median_per_case_r2"})
    summary.to_csv(OUT / "diagnostics_summary.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
