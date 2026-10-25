"""
test_fetch_features.py – Unit tests for directional fetch raymarching, upwind height bands, and flow orientations.
"""

import math
import numpy as np
import pytest

from pipeline.S05_feature_engineering.fetch_features import compute_directional_fetch_maps


def test_fetch_with_periodic_wall():
    """Verify upwind and downwind fetch calculations on an idealized obstacle."""
    # 20x20 domain, dx = 1.0m
    # Building barrier placed at col 10 (rows 0 to 19)
    H = np.zeros((20, 20), dtype=float)
    H[:, 10] = 15.0

    # Wind blowing along +x (0 degrees, West to East)
    # Upwind direction is -x (West). Downwind is +x (East).
    fetch_maps, summary = compute_directional_fetch_maps(
        H,
        dx=1.0,
        dy=1.0,
        forcing_dir_deg=0.0,
        upwind_cap_m=150.0,
        downwind_cap_m=60.0,
    )

    upwind = fetch_maps["upwind_fetch"]
    downwind = fetch_maps["downwind_fetch"]

    # Barrier is at col 10 (fetch = 0)
    assert np.all(upwind[:, 10] == 0.0)

    # For cell at col 12: upwind obstacle (col 10) is 2 metres to the West!
    assert np.all(abs(upwind[:, 12] - 2.0) < 1e-4)

    # For cell at col 9: looking downwind (+x), obstacle (col 10) is 1 metre to the East!
    assert np.all(abs(downwind[:, 9] - 1.0) < 1e-4)

    # Periodic wrapping:
    # Cell at col 5: upwind (-x) encounters periodic wrap, distance = 5 + (20 - 10) = 15 m!
    assert np.all(abs(upwind[:, 5] - 15.0) < 1e-4)

    assert summary.upwind_fetch_mean_m > 0.0


def test_fetch_cardinal_directions():
    """Verify fetch calculations across all four cardinal forcing directions (0, 90, 180, 270 deg)."""
    # 30x30 domain, dx = 1.0m
    # Single horizontal barrier at row 15 (y = 15)
    H_horiz = np.zeros((30, 30), dtype=float)
    H_horiz[15, :] = 12.0

    # 1. Flow along +y (90 deg, South to North):
    # Upwind is -y (South).
    # For cell at row 18: upwind obstacle at row 15 is 3 metres South.
    # Downwind is +y (North): for cell at row 12, downwind obstacle at row 15 is 3 metres North.
    f_90, _ = compute_directional_fetch_maps(H_horiz, dx=1.0, dy=1.0, forcing_dir_deg=90.0)
    assert np.all(abs(f_90["upwind_fetch"][18, :] - 3.0) < 1e-4)
    assert np.all(abs(f_90["downwind_fetch"][12, :] - 3.0) < 1e-4)

    # 2. Flow along -y (270 deg, North to South):
    # Upwind is +y (North).
    # For cell at row 12: upwind obstacle at row 15 is 3 metres North.
    # Downwind is -y (South): for cell at row 18, downwind obstacle is 3 metres South.
    f_270, _ = compute_directional_fetch_maps(H_horiz, dx=1.0, dy=1.0, forcing_dir_deg=270.0)
    assert np.all(abs(f_270["upwind_fetch"][12, :] - 3.0) < 1e-4)
    assert np.all(abs(f_270["downwind_fetch"][18, :] - 3.0) < 1e-4)

    # Vertical barrier at col 15 (x = 15)
    H_vert = np.zeros((30, 30), dtype=float)
    H_vert[:, 15] = 12.0

    # 3. Flow along -x (180 deg, East to West):
    # Upwind is +x (East).
    # For cell at col 12: looking upwind (+x) hits obstacle at col 15 (distance = 3m)
    f_180, _ = compute_directional_fetch_maps(H_vert, dx=1.0, dy=1.0, forcing_dir_deg=180.0)
    assert np.all(abs(f_180["upwind_fetch"][:, 12] - 3.0) < 1e-4)
    assert np.all(abs(f_180["downwind_fetch"][:, 18] - 3.0) < 1e-4)


def test_upwind_height_bands_alignment_and_numerical():
    """Verify that upwind building height bands correctly record obstacle heights at expected distances."""
    # 100x100 domain, dx = dy = 1.0 m (Lx = 100m ensures 2nd periodic wrap at > 100m)
    # Place a building barrier at col 10 of height 24.0 m
    H = np.zeros((100, 100), dtype=float)
    H[:, 10] = 24.0

    # Wind along +x (0 deg): upwind direction is -x (West)
    fetch_maps, _ = compute_directional_fetch_maps(
        H, dx=1.0, dy=1.0, forcing_dir_deg=0.0
    )

    band_1 = fetch_maps["upwind_h_1_10m"]   # 1-10m
    band_2 = fetch_maps["upwind_h_11_30m"]  # 11-30m
    band_3 = fetch_maps["upwind_h_31_60m"]  # 31-60m

    # 1. At col 15: barrier is at col 10 (distance = 5 m upwind)
    # Distance 5m falls in Band 1 [1-10m], but NOT in Band 2 or 3!
    assert np.all(abs(band_1[:, 15] - 24.0) < 1e-4), "Band 1 [1-10m] must record 24.0m"
    assert np.all(band_2[:, 15] == 0.0), "Band 2 [11-30m] must be 0.0 at distance 5m"
    assert np.all(band_3[:, 15] == 0.0), "Band 3 [31-60m] must be 0.0 at distance 5m"

    # 2. At col 30: barrier is at col 10 (distance = 20 m upwind)
    # Distance 20m falls in Band 2 [11-30m], but NOT in Band 1 or 3!
    assert np.all(band_1[:, 30] == 0.0), "Band 1 [1-10m] must be 0.0 at distance 20m"
    assert np.all(abs(band_2[:, 30] - 24.0) < 1e-4), "Band 2 [11-30m] must record 24.0m"
    assert np.all(band_3[:, 30] == 0.0), "Band 3 [31-60m] must be 0.0 at distance 20m"

    # 3. At col 48: barrier is at col 10 (distance = 38 m upwind)
    # Distance 38m falls in Band 3 [31-60m]!
    assert np.all(band_1[:, 48] == 0.0)
    assert np.all(band_2[:, 48] == 0.0)
    assert np.all(abs(band_3[:, 48] - 24.0) < 1e-4), "Band 3 [31-60m] must record 24.0m"

    # 4. Building cells must have 0.0 across all bands
    assert np.all(band_1[:, 10] == 0.0)
    assert np.all(band_2[:, 10] == 0.0)
    assert np.all(band_3[:, 10] == 0.0)
