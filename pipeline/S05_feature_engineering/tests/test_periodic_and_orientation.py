"""
test_periodic_and_orientation.py – Regression tests for edge-crossing buildings, facet
orientation and the upwind height bands on 0.5 m grids.
"""

import numpy as np

from pipeline.S05_feature_engineering.fetch_features import compute_directional_fetch_maps
from pipeline.S05_feature_engineering.morphology import compute_morphology_features
from pipeline.S05_feature_engineering.orientation_features import compute_directional_morphology


def test_building_across_the_domain_edge_counts_once():
    """A 16 m block split over the periodic edge is one building of 256 m², not two."""
    H = np.zeros((64, 64))
    H[-8:, 10:26] = 16.0
    H[:8, 10:26] = 16.0
    m = compute_morphology_features(H, dx=1.0)
    assert m.n_connected_components == 1
    assert m.mean_footprint_area_m2 == 256.0


def test_corner_crossing_building_counts_once():
    """A block over a corner touches all four edges and is still one building."""
    H = np.zeros((40, 40))
    H[:4, :4] = H[:4, -4:] = H[-4:, :4] = H[-4:, -4:] = 10.0
    assert compute_morphology_features(H, dx=1.0).n_connected_components == 1


def test_windward_share_counts_both_facing_wall_sets_at_45_degrees():
    """For a cube at 45 deg, half of the wall area faces the wind; at 0 deg a quarter does."""
    H = np.zeros((64, 64))
    H[24:40, 24:40] = 16.0
    assert np.isclose(compute_directional_morphology(H, dx=1.0, forcing_dir_deg=45.0).windward_wall_share, 0.5)
    assert np.isclose(compute_directional_morphology(H, dx=1.0, forcing_dir_deg=0.0).windward_wall_share, 0.25)


def test_grid_aligned_facets_and_resolution_independent_entropy():
    """Axis-aligned walls read 0 or 90 deg, and the same cube gives the same entropy at 1 m and 0.5 m."""
    H1 = np.zeros((64, 64))
    H1[24:40, 24:40] = 16.0
    H05 = np.kron(H1, np.ones((2, 2)))  # the same layout on a 0.5 m grid
    o1 = compute_directional_morphology(H1, dx=1.0, forcing_dir_deg=0.0)
    o05 = compute_directional_morphology(H05, dx=0.5, forcing_dir_deg=0.0)
    assert o1.dominant_facet_angle_deg in (0.0, 90.0)
    assert o1.relative_wind_facet_angle_deg == 0.0
    assert np.isclose(o1.facet_orientation_entropy, o05.facet_orientation_entropy)


def test_upwind_bands_leave_no_gap_on_half_metre_grids():
    """A one-cell wall 10.5 m upwind falls into the 11-30 m band on a 0.5 m grid."""
    H = np.zeros((40, 200))
    H[:, 100 - 21] = 12.0  # 21 cells of 0.5 m = 10.5 m upwind of column 100 for a wind towards +x
    maps, _ = compute_directional_fetch_maps(H, dx=0.5, forcing_dir_deg=0.0)
    assert maps["upwind_h_11_30m"][20, 100] == 12.0
    assert maps["upwind_h_1_10m"][20, 100] == 0.0
