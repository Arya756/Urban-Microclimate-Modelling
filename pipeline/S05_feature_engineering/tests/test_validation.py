"""
test_validation.py – Unit tests for domain validation and synthetic benchmark.
"""

import numpy as np
import pytest

from pipeline.S05_feature_engineering.validation import (
    create_synthetic_benchmark,
    validate_height_map,
)


def test_synthetic_benchmark_analytical_values():
    """Verify analytical assertions of the synthetic 40x40 benchmark."""
    H, dx = create_synthetic_benchmark()
    assert H.shape == (40, 40)
    assert dx == 1.0

    geom = validate_height_map(H, dx=dx)
    assert geom.nx == 40
    assert geom.ny == 40
    assert geom.dx == 1.0
    assert geom.dy == 1.0
    assert geom.domain_x_m == 40.0
    assert geom.domain_y_m == 40.0
    assert geom.domain_area_m2 == 1600.0
    assert geom.n_total_cells == 1600
    assert geom.n_bldg_cells == 57
    assert geom.n_open_cells == 1543
    assert abs(geom.lambda_p - 0.035625) < 1e-9
    assert geom.is_valid is True
    assert geom.is_all_open is False
    assert geom.is_all_building is False


def test_all_open_domain():
    """Verify validation behavior on empty / all-open domain."""
    H = np.zeros((30, 20), dtype=float)
    geom = validate_height_map(H, dx=0.5, dy=0.5)
    assert geom.n_bldg_cells == 0
    assert geom.n_open_cells == 600
    assert geom.lambda_p == 0.0
    assert geom.is_all_open is True
    assert geom.is_all_building is False
    assert geom.is_valid is True


def test_all_building_domain():
    """Verify validation behavior on fully built domain."""
    H = np.full((10, 10), 15.0, dtype=float)
    geom = validate_height_map(H, dx=1.0)
    assert geom.n_bldg_cells == 100
    assert geom.n_open_cells == 0
    assert geom.lambda_p == 1.0
    assert geom.is_all_open is False
    assert geom.is_all_building is True


def test_invalid_inputs_raise_errors():
    """Verify that non-finite values and negative elevations raise ValueError."""
    # 1D array
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_height_map(np.array([1.0, 2.0]), dx=1.0)

    # NaNs
    H_nan = np.zeros((10, 10))
    H_nan[2, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_height_map(H_nan, dx=1.0)

    # Negative elevations
    H_neg = np.zeros((10, 10))
    H_neg[3, 3] = -5.0
    with pytest.raises(ValueError, match="negative elevations"):
        validate_height_map(H_neg, dx=1.0)

    # Non-positive grid spacing
    with pytest.raises(ValueError, match="strictly positive"):
        validate_height_map(np.zeros((10, 10)), dx=0.0)
