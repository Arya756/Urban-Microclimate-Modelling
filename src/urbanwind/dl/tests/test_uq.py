"""Tests for the S14 uncertainty helpers (urbanwind.dl.uq)."""

from __future__ import annotations

import numpy as np
import pytest

from urbanwind.dl import uq


def test_conformal_quantile_is_the_finite_sample_rank():
    # n = 9, alpha = 0.2: rank ceil(10 * 0.8) = 8, so the 8th smallest score
    assert uq.conformal_quantile(np.arange(1.0, 10.0), alpha=0.2) == 8.0
    assert uq.conformal_quantile(np.array([]), alpha=0.1) == float("inf")


def test_a_fold_is_never_calibrated_on_its_own_cells():
    scores = np.array([100.0, 100.0, 1.0, 2.0])
    folds = np.array([0, 0, 1, 1])
    q = uq.fold_quantiles(scores, folds, alphas=[0.5])
    # fold 0 sees only fold 1's scores (1, 2): rank ceil(3 * 0.5) = 2 -> 2.0; fold 1 sees only 100s
    assert q[(0, None)][0.5] == 2.0
    assert q[(1, None)][0.5] == 100.0


def test_groups_are_calibrated_separately():
    scores = np.array([1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    q = uq.fold_quantiles(scores, folds, alphas=[0.5], groups=groups)
    assert q[(0, 0)][0.5] == 1.0 and q[(0, 1)][0.5] == 10.0


def test_normalised_intervals_stay_calibrated_where_a_constant_width_fails():
    # errors whose size follows sigma: a constant width over-covers calm cells and under-covers uncertain ones
    rng = np.random.default_rng(0)
    n = 40_000
    sigma = rng.uniform(0.1, 2.0, n)
    err = np.abs(rng.normal(0.0, sigma))
    folds = rng.integers(0, 5, n)
    norm = uq.fold_quantiles(err / (sigma + uq.EPS), folds, alphas=[0.1])
    const = uq.fold_quantiles(err, folds, alphas=[0.1])
    hw_norm = np.array([norm[(k, None)][0.1] for k in folds]) * (sigma + uq.EPS)
    hw_const = np.array([const[(k, None)][0.1] for k in folds])
    uncertain = sigma > np.quantile(sigma, 0.75)
    assert 0.88 < (err <= hw_norm).mean() < 0.92
    assert 0.86 < (err <= hw_norm)[uncertain].mean() < 0.94
    assert (err <= hw_const)[uncertain].mean() < 0.85


def test_coverage_counts_cells_inside_the_interval():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert uq.coverage(y, y - 0.5, y + 0.5) == 1.0
    assert uq.coverage(y, y + 1.0, y + 2.0) == 0.0


def test_wall_distance_wraps_around_the_domain():
    buildings = np.zeros((1, 8), dtype=bool)
    buildings[0, 0] = True
    dist = uq.periodic_wall_distance(buildings, dx=0.5)
    assert dist[0, 0] == 0.0
    assert dist[0, 7] == pytest.approx(0.5)  # one cell away across the edge
    assert dist[0, 4] == pytest.approx(2.0)  # four cells either way
    assert np.isinf(uq.periodic_wall_distance(np.zeros((3, 3), dtype=bool), dx=1.0)).all()


def test_spearman():
    a = np.arange(10.0)
    assert uq.spearman(a, a ** 3) == pytest.approx(1.0)
    assert uq.spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(uq.spearman(a, np.ones(10)))


def test_selective_mae_drops_the_most_uncertain_cells():
    abs_err = np.array([1.0, 2.0, 3.0, 10.0])
    sigma = np.array([0.1, 0.2, 0.3, 5.0])
    assert uq.selective_mae(abs_err, sigma, drop=0.25) == pytest.approx(2.0)
    assert uq.selective_mae(abs_err, sigma, drop=0.0) == pytest.approx(4.0)
