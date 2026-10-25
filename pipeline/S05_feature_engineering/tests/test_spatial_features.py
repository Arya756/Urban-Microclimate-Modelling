"""
test_spatial_features.py – Unit tests for periodic distance transform and multi-scale maps.
"""

import math
import numpy as np
import pytest

from pipeline.S05_feature_engineering.spatial_features import (
    compute_multiscale_density_maps,
    compute_multiscale_height_maps,
    compute_periodic_distance_transform,
)
from pipeline.S05_feature_engineering.validation import create_synthetic_benchmark


def _brute_force_periodic_distance(
    bldg_mask: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Exact brute-force reference periodic Euclidean distance calculation."""
    ny, nx = bldg_mask.shape
    if not np.any(bldg_mask):
        diag = math.sqrt((nx * dx) ** 2 + (ny * dy) ** 2)
        return np.full((ny, nx), diag, dtype=np.float64)

    bldg_coords = np.argwhere(bldg_mask)  # shape (K, 2)
    dist = np.zeros((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            if bldg_mask[i, j]:
                dist[i, j] = 0.0
                continue
            # Minimal periodic delta along y and x
            delta_y = np.abs(i - bldg_coords[:, 0])
            dy_periodic = np.minimum(delta_y, ny - delta_y) * dy
            delta_x = np.abs(j - bldg_coords[:, 1])
            dx_periodic = np.minimum(delta_x, nx - delta_x) * dx
            dist[i, j] = np.min(np.sqrt(dy_periodic**2 + dx_periodic**2))

    return dist


def test_periodic_distance_transform_wrapping_corner():
    """Verify that periodic boundaries wrap correctly in distance transform."""
    # Place a single building cell at the corner (0, 0)
    H = np.zeros((20, 20), dtype=float)
    H[0, 0] = 10.0
    dx, dy = 1.0, 1.0

    dist_map, summary = compute_periodic_distance_transform(H, dx=dx, dy=dy)

    # (0, 0) is a building, so dist is 0
    assert dist_map[0, 0] == 0.0

    # Due to periodic wrapping:
    # Cell (0, 19) is adjacent to (0, 0) across the periodic boundary!
    assert abs(dist_map[0, 19] - 1.0) < 1e-4
    # Cell (19, 0) is adjacent to (0, 0) across Y periodic boundary!
    assert abs(dist_map[19, 0] - 1.0) < 1e-4
    # Cell (19, 19) is diagonally adjacent across both boundaries!
    assert abs(dist_map[19, 19] - math.sqrt(2.0)) < 1e-4

    assert summary.dist_mean_m > 0.0
    assert summary.dist_max_m <= math.sqrt(10**2 + 10**2) + 0.5


def test_periodic_distance_across_all_boundaries():
    """Verify exact distance across top, bottom, left, and right boundaries."""
    ny, nx = 25, 30
    dx, dy = 1.0, 1.0

    # Test top boundary: building at row 0, col 15
    H_top = np.zeros((ny, nx), dtype=float)
    H_top[0, 15] = 15.0
    dist_top, _ = compute_periodic_distance_transform(H_top, dx=dx, dy=dy)
    # Cell at row ny-1 (bottom) is 1 grid cell away across periodic top/bottom boundary
    assert abs(dist_top[ny - 1, 15] - 1.0) < 1e-4
    assert abs(dist_top[ny - 2, 15] - 2.0) < 1e-4

    # Test right boundary: building at row 12, col nx-1
    H_right = np.zeros((ny, nx), dtype=float)
    H_right[12, nx - 1] = 15.0
    dist_right, _ = compute_periodic_distance_transform(H_right, dx=dx, dy=dy)
    # Cell at col 0 (left) is 1 grid cell away across periodic left/right boundary
    assert abs(dist_right[12, 0] - 1.0) < 1e-4
    assert abs(dist_right[12, 1] - 2.0) < 1e-4


def test_periodic_distance_matches_brute_force_reference():
    """Compare EDT implementation against independent brute-force reference on small synthetic arrays."""
    np.random.seed(123)
    for ny, nx in [(12, 16), (15, 15), (20, 10)]:
        for dx, dy in [(1.0, 1.0), (1.5, 2.0), (0.8, 1.4)]:
            # Create synthetic building mask with multiple obstacles
            mask = np.random.rand(ny, nx) < 0.20
            # Ensure at least one building
            mask[3, 4] = True
            H = mask.astype(float) * 12.0

            d_edt, _ = compute_periodic_distance_transform(H, dx=dx, dy=dy)
            d_ref = _brute_force_periodic_distance(mask, dx=dx, dy=dy)

            max_err = float(np.max(np.abs(d_edt - d_ref)))
            assert max_err < 1e-4, (
                f"Discrepancy with brute force on grid ({ny}, {nx}), dx={dx}, dy={dy}: max_err={max_err}"
            )


def test_sparse_layout_distance_exceeding_80m():
    """Verify that distances exceeding 80m are computed with exact Euclidean precision."""
    # 200m x 200m domain with dx = 1.0m (200 x 200 cells)
    # Single building of 1 cell in the exact center: (100, 100)
    ny, nx = 200, 200
    dx, dy = 1.0, 1.0
    H = np.zeros((ny, nx), dtype=float)
    H[100, 100] = 20.0

    dist_map, summary = compute_periodic_distance_transform(H, dx=dx, dy=dy)

    # In a periodic domain of 200x200m with building at (100, 100),
    # the maximum periodic distance occurs at the corners (0, 0), (0, 199), (199, 0), (199, 199).
    # Distance from (0, 0) to (100, 100):
    # dx_dist = min(|0 - 100|, 200 - 100) = 100m
    # dy_dist = min(|0 - 100|, 200 - 100) = 100m
    # Exact distance = sqrt(100^2 + 100^2) = 141.421356... m, which is > 80m!
    expected_corner_dist = math.sqrt(100.0**2 + 100.0**2)
    actual_corner_dist = float(dist_map[0, 0])

    assert abs(actual_corner_dist - expected_corner_dist) < 1e-3, (
        f"Distance > 80m failed: expected {expected_corner_dist:.4f}m, got {actual_corner_dist:.4f}m"
    )
    assert summary.dist_max_m >= 141.0, f"Max distance {summary.dist_max_m} is unexpectedly small."


def test_rectangular_grid_unequal_dx_dy():
    """Verify exact distance with rectangular grid dimensions and unequal dx and dy."""
    ny, nx = 30, 50
    dx = 2.0
    dy = 3.0

    # Building at (0, 0)
    H = np.zeros((ny, nx), dtype=float)
    H[0, 0] = 10.0

    dist_map, _ = compute_periodic_distance_transform(H, dx=dx, dy=dy)

    # Check cell (0, 1): physical distance is 1 * dx = 2.0 m
    assert abs(dist_map[0, 1] - 2.0) < 1e-4
    # Check cell (1, 0): physical distance is 1 * dy = 3.0 m
    assert abs(dist_map[1, 0] - 3.0) < 1e-4
    # Check cell (1, 1): physical distance is sqrt(dx^2 + dy^2) = sqrt(4 + 9) = sqrt(13)
    assert abs(dist_map[1, 1] - math.sqrt(13.0)) < 1e-4
    # Periodic wrapped cell (0, nx-1): physical distance is 1 * dx = 2.0 m
    assert abs(dist_map[0, nx - 1] - 2.0) < 1e-4
    # Periodic wrapped cell (ny-1, 0): physical distance is 1 * dy = 3.0 m
    assert abs(dist_map[ny - 1, 0] - 3.0) < 1e-4


def test_edge_cases_all_open_and_all_building():
    """Verify all-open (no buildings) and all-building edge cases."""
    ny, nx = 20, 25
    dx, dy = 1.0, 1.5

    # 1. All open
    H_open = np.zeros((ny, nx), dtype=float)
    dist_open, summary_open = compute_periodic_distance_transform(H_open, dx=dx, dy=dy)
    expected_diag = math.sqrt((nx * dx) ** 2 + (ny * dy) ** 2)

    assert np.all(dist_open == expected_diag)
    assert summary_open.dist_mean_m == expected_diag
    assert summary_open.dist_max_m == expected_diag
    assert np.all(np.isfinite(dist_open))

    # 2. All building
    H_bldg = np.full((ny, nx), 15.0, dtype=float)
    dist_bldg, summary_bldg = compute_periodic_distance_transform(H_bldg, dx=dx, dy=dy)

    assert np.all(dist_bldg == 0.0)
    assert summary_bldg.dist_mean_m == 0.0
    assert summary_bldg.dist_max_m == 0.0
    assert np.all(np.isfinite(dist_bldg))


def test_multiscale_density_spatial_mean_invariant():
    """Verify that domain spatial mean of local density equals global plan-area density."""
    H, dx = create_synthetic_benchmark()
    # Global lambda_p = 57 / 1600 = 0.035625
    expected_lp = 0.035625

    dens_maps = compute_multiscale_density_maps(H, dx=dx, radii_m=(8.0, 16.0, 32.0, 64.0))

    for r in (8, 16, 32, 64):
        key = f"lambda_p_{r}m"
        assert key in dens_maps
        spatial_mean = float(np.mean(dens_maps[key]))
        # Conservation of area under periodic boundary convolution
        assert abs(spatial_mean - expected_lp) < 1e-4, (
            f"Mean of {key} ({spatial_mean}) differs from global lambda_p ({expected_lp})"
        )


def test_multiscale_height_spatial_mean_invariant():
    """Verify that domain spatial mean of local height maps equals raster mean elevation."""
    H, dx = create_synthetic_benchmark()
    expected_raster_mean_h = float(np.mean(H))  # 556 / 1600 = 0.3475

    height_maps = compute_multiscale_height_maps(H, dx=dx, radii_m=(8.0, 16.0, 32.0, 64.0))

    for r in (8, 16, 32, 64):
        key = f"h_mean_{r}m"
        assert key in height_maps
        spatial_mean = float(np.mean(height_maps[key]))
        # Conservation of volume under periodic boundary convolution
        assert abs(spatial_mean - expected_raster_mean_h) < 1e-4, (
            f"Mean of {key} ({spatial_mean}) differs from raster mean height ({expected_raster_mean_h})"
        )
