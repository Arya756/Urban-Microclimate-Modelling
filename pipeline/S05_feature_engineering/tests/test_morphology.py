"""
test_morphology.py – Unit tests for building height statistics, plan-area density, and directional frontal-area index.
"""

import math
import numpy as np

from pipeline.S05_feature_engineering.morphology import compute_morphology_features
from pipeline.S05_feature_engineering.orientation_features import compute_directional_morphology
from pipeline.S05_feature_engineering.validation import create_synthetic_benchmark


def test_morphology_on_synthetic_benchmark():
    """Verify analytical height statistics and volume on synthetic layout."""
    H, dx = create_synthetic_benchmark()
    m = compute_morphology_features(H, dx=dx)

    # Building 1: 25 cells @ 12m; Building 2: 32 cells @ 8m
    # Total building cells = 57
    # Total cells = 40 * 40 = 1600
    expected_volume = 556.0
    assert abs(m.total_volume_m3 - expected_volume) < 1e-6

    # Plan-area density: 57 / 1600 = 0.035625 exactly
    assert abs(m.lambda_p - 0.035625) < 1e-7

    # Mean building height = 556 / 57 = 9.75438596...
    expected_mean_h = 556.0 / 57.0
    assert abs(m.mean_h - expected_mean_h) < 1e-6

    assert m.min_h == 8.0
    assert m.max_h == 12.0
    assert m.median_h == 8.0  # 32 out of 57 cells are 8m, so median is 8m
    assert m.p25_h == 8.0
    assert m.p75_h == 12.0
    assert m.iqr_h == 4.0

    # Volume per unit area = 556 / 1600 = 0.3475 m
    assert abs(m.volume_per_area_m - 0.3475) < 1e-6
    assert abs(m.raster_mean_h - 0.3475) < 1e-6

    # Connected components: exactly 2 distinct buildings
    assert m.n_connected_components == 2
    # Mean footprint: (25 + 32) / 2 = 28.5 m2
    assert abs(m.mean_footprint_area_m2 - 28.5) < 1e-6


def test_plan_area_density_regular_array():
    """Verify exact analytical lambda_p on a regular array of uniform blocks."""
    # 20x20 domain, dx = dy = 1.0 m (area = 400 m2)
    # 4 distinct blocks of 4x4 cells (area = 16 m2 each)
    # Total building footprint = 4 * 16 = 64 m2
    # Expected lambda_p = 64 / 400 = 0.1600 exactly
    H = np.zeros((20, 20), dtype=float)
    H[2:6, 2:6] = 10.0
    H[2:6, 12:16] = 10.0
    H[12:16, 2:6] = 10.0
    H[12:16, 12:16] = 10.0

    m = compute_morphology_features(H, dx=1.0, dy=1.0)
    assert abs(m.lambda_p - 0.1600) < 1e-7
    assert m.n_connected_components == 4
    assert abs(m.mean_footprint_area_m2 - 16.0) < 1e-7


def test_directional_frontal_area_index_analytical():
    """Verify exact analytical directional frontal-area index, wall area, and windward share."""
    # 20x20 domain, dx = dy = 1.0 m (Total ground area = 400 m2)
    # Place a single rectangular building:
    # rows 5 to 10 (height along y = 6 cells = 6m)
    # cols 5 to 8  (width along x = 4 cells = 4m)
    # height H = 10m
    H = np.zeros((20, 20), dtype=float)
    H[5:11, 5:9] = 10.0  # rows 5..10 (6 cells), cols 5..8 (4 cells)

    # 1. Flow along +x (0 deg):
    # Frontal face seen by +x wind is the west wall: width = 6m, height = 10m
    # Frontal area = 6 * 10 = 60 m2
    # lambda_f_x = 60 / 400 = 0.1500
    orient_0 = compute_directional_morphology(H, dx=1.0, dy=1.0, forcing_dir_deg=0.0)
    assert abs(orient_0.lambda_f_x - 0.1500) < 1e-6
    assert abs(orient_0.lambda_f_y - 0.1000) < 1e-6
    assert abs(orient_0.lambda_f_dir - 0.1500) < 1e-6

    # Total wall area: 2 * (6*10) + 2 * (4*10) = 120 + 80 = 200 m2
    # lambda_w = 200 / 400 = 0.5000
    assert abs(orient_0.lambda_w - 0.5000) < 1e-6
    # Windward wall share: 60 / 200 = 0.3000
    assert abs(orient_0.windward_wall_share - 0.3000) < 1e-6

    # 2. Flow along +y (90 deg):
    # Frontal face seen by +y wind is south wall: width = 4m, height = 10m
    # Frontal area = 4 * 10 = 40 m2
    # lambda_f_dir = 40 / 400 = 0.1000
    orient_90 = compute_directional_morphology(H, dx=1.0, dy=1.0, forcing_dir_deg=90.0)
    assert abs(orient_90.lambda_f_dir - 0.1000) < 1e-6
    assert abs(orient_90.windward_wall_share - (40.0 / 200.0)) < 1e-6

    # 3. Flow at 45 deg:
    # lambda_f_dir = cos(45)*0.15 + sin(45)*0.10 = (sqrt(2)/2) * 0.25
    expected_lf_45 = (math.sqrt(2.0) / 2.0) * (0.15 + 0.10)
    orient_45 = compute_directional_morphology(H, dx=1.0, dy=1.0, forcing_dir_deg=45.0)
    assert abs(orient_45.lambda_f_dir - expected_lf_45) < 1e-6


def test_morphology_all_open_domain():
    """Verify all-open domain edge case produces valid zeros and no NaNs."""
    H = np.zeros((20, 20), dtype=float)
    m = compute_morphology_features(H, dx=1.0)

    assert m.mean_h == 0.0
    assert m.std_h == 0.0
    assert m.lambda_p == 0.0
    assert m.total_volume_m3 == 0.0
    assert m.n_connected_components == 0
    assert m.skew_h == 0.0
    assert m.kurt_h == 0.0
