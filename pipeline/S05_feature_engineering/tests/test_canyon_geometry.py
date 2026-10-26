"""
test_canyon_geometry.py – Unit tests for canyon width and aspect ratio computation.
"""

import numpy as np

from pipeline.S05_feature_engineering.spatial_features import (
    compute_canyon_geometry,
    compute_periodic_distance_transform,
)
from pipeline.S05_feature_engineering.validation import create_synthetic_benchmark


def test_canyon_width_is_twice_dist_to_building():
    """Canyon width = 2 * dist_to_building, clamped to >= 1m."""
    H, dx = create_synthetic_benchmark()
    dist_map, _ = compute_periodic_distance_transform(H, dx)
    canyon_w, _ = compute_canyon_geometry(dist_map, H)

    expected_canyon_w = np.maximum(1.0, 2.0 * dist_map).astype(np.float32)
    np.testing.assert_allclose(canyon_w, expected_canyon_w, rtol=1e-5,
                               err_msg="Canyon width must equal max(1, 2*dist_map)")


def test_aspect_ratio_zero_on_building_cells():
    """Aspect ratio must be 0.0 on building cells."""
    H, dx = create_synthetic_benchmark()
    dist_map, _ = compute_periodic_distance_transform(H, dx)
    _, aspect_r = compute_canyon_geometry(dist_map, H)

    assert np.all(aspect_r[H > 0] == 0.0), (
        "Aspect ratio must be exactly 0.0 on all building cells"
    )


def test_aspect_ratio_positive_on_fluid_cells():
    """Aspect ratio must be positive on open-space cells (buildings present)."""
    H, dx = create_synthetic_benchmark()
    dist_map, _ = compute_periodic_distance_transform(H, dx)
    _, aspect_r = compute_canyon_geometry(dist_map, H)

    fluid_mask = (H == 0.0)
    assert np.all(aspect_r[fluid_mask] > 0.0), (
        "Aspect ratio must be strictly positive on all fluid cells when buildings exist"
    )


def test_aspect_ratio_clipped_to_10():
    """Aspect ratio is clipped to the physical maximum of 10."""
    # Very tall building, very narrow canyon -> H/W >> 10
    H = np.zeros((10, 10), dtype=float)
    H[5, 5] = 100.0  # 100m tower
    dist_map, _ = compute_periodic_distance_transform(H, 1.0)
    _, aspect_r = compute_canyon_geometry(dist_map, H)

    assert float(np.max(aspect_r)) <= 10.0, (
        f"Aspect ratio must be clipped to 10.0. Max found: {float(np.max(aspect_r)):.2f}"
    )


def test_aspect_ratio_zero_all_open_domain():
    """All-open domain (no buildings): aspect ratio = 0 everywhere."""
    H_open = np.zeros((10, 10), dtype=float)
    dist_map, _ = compute_periodic_distance_transform(H_open, 1.0)
    canyon_w, aspect_r = compute_canyon_geometry(dist_map, H_open)

    assert np.all(aspect_r == 0.0), (
        "All-open domain: aspect ratio must be 0 everywhere (no buildings)"
    )


def test_canyon_geometry_finite_values():
    """No NaN or Inf in canyon_width or aspect_ratio."""
    H, dx = create_synthetic_benchmark()
    dist_map, _ = compute_periodic_distance_transform(H, dx)
    canyon_w, aspect_r = compute_canyon_geometry(dist_map, H)

    assert np.all(np.isfinite(canyon_w)), "Canyon width contains NaN or Inf"
    assert np.all(np.isfinite(aspect_r)), "Aspect ratio contains NaN or Inf"


def test_canyon_geometry_nonnegative():
    """Canyon width and aspect ratio are always non-negative."""
    H, dx = create_synthetic_benchmark()
    dist_map, _ = compute_periodic_distance_transform(H, dx)
    canyon_w, aspect_r = compute_canyon_geometry(dist_map, H)

    assert np.all(canyon_w >= 0.0), "Canyon width must be non-negative"
    assert np.all(aspect_r >= 0.0), "Aspect ratio must be non-negative"
