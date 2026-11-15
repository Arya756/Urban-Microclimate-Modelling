"""
uq.py: uncertainty for the predicted wind maps (S14). Pure NumPy, so it is cheap to test.

The spread of a set of predictions (three training seeds, or MC dropout passes) is the uncertainty
map. Split-conformal calibration turns it into intervals with a stated coverage. Every quantile for
the cases of outer fold k comes from the other folds only, so a city or layout family never
calibrates its own intervals.

Two conformity scores:
- constant width: |y - pred|, which gives the same half-width q everywhere;
- normalised:     |y - pred| / (sigma + eps), which gives the half-width q * (sigma + eps), wider
                  where the predictions disagree. This is what makes the uncertainty map useful.

Public API
----------
conformal_quantile(scores, alpha)           : finite-sample (1 - alpha) quantile of conformity scores
fold_quantiles(scores, folds, alphas, groups): per outer fold (and group), the quantiles of the OTHER folds
normalised_score(y, pred, sigma, eps)        : |y - pred| / (sigma + eps)
coverage(y, lo, hi)                          : share of cells inside [lo, hi]
spearman(a, b)                               : Spearman rank correlation
selective_mae(abs_err, sigma, drop)          : MAE after dropping the most uncertain share of cells
periodic_wall_distance(buildings, dx)        : metres from each cell to the nearest building, wrapping round
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

EPS = 0.05  # u*; keeps the normalised score finite where the members agree exactly


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score: the split-conformal threshold for 1 - alpha coverage."""
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = s.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(rank, n) - 1])


def fold_quantiles(scores: np.ndarray, folds: np.ndarray, alphas: list[float],
                   groups: np.ndarray | None = None) -> dict[tuple[int, int | None], dict[float, float]]:
    """{(fold, group): {alpha: quantile}}, each from the cells of the other folds (and the same group).

    ``groups`` (for example 0 for idealized and 1 for realistic cells) calibrates each group on its own;
    None pools every cell.
    """
    out = {}
    group_ids = [None] if groups is None else sorted(np.unique(groups).tolist())
    for k in sorted(np.unique(folds).tolist()):
        for g in group_ids:
            pick = folds != k if g is None else (folds != k) & (groups == g)
            s = np.sort(scores[pick])  # the scores' own dtype: float32 halves the memory on 88 million cells
            n = s.size
            out[(k, g)] = {a: float(s[min(int(np.ceil((n + 1) * (1 - a))), n) - 1]) for a in alphas}
    return out


def normalised_score(y: np.ndarray, pred: np.ndarray, sigma: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.abs(y - pred) / (sigma + eps)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean()) if inside.size else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


def selective_mae(abs_err: np.ndarray, sigma: np.ndarray, drop: float) -> float:
    """MAE over the cells left after dropping the ``drop`` share with the largest sigma."""
    order = np.argsort(sigma, kind="stable")
    keep = order[: len(order) - int(round(drop * len(order)))]
    return float(abs_err[keep].mean()) if keep.size else float("nan")


def periodic_wall_distance(buildings: np.ndarray, dx: float) -> np.ndarray:
    """Distance in metres from every cell to the nearest building cell, on a domain that repeats.

    The domain is tiled 3 x 3 so distances wrap across the edges, then the centre tile is kept.
    A domain without buildings gets infinity everywhere.
    """
    if not buildings.any():
        return np.full(buildings.shape, np.inf)
    ny, nx = buildings.shape
    tiled = np.tile(~buildings, (3, 3))
    return ndimage.distance_transform_edt(tiled)[ny:2 * ny, nx:2 * nx] * dx
