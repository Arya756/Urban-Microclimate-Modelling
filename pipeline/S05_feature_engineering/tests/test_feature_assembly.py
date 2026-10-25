"""
test_feature_assembly.py – Unit tests for feature tensor assembly and leakage prevention.
"""

import numpy as np
import pytest

from pipeline.S05_feature_engineering.config import DEFAULT_CONFIG
from pipeline.S05_feature_engineering.feature_assembly import (
    assemble_case_features,
    assemble_pixel_feature_tensor,
)
from pipeline.S05_feature_engineering.validation import create_synthetic_benchmark


def test_zero_target_leakage():
    """Verify that no target wind-speed or target profile columns exist in predictor features."""
    H, dx = create_synthetic_benchmark()
    record = assemble_case_features(
        H=H,
        dx=dx,
        case_id="synth_01",
        forcing_dir_deg=45.0,
        metadata_row={"subset": "idealized", "split_group": "ideal-synth", "u_tau_metadata": 0.21},
    )

    # Assert no forbidden target-derived keys are present in the predictor dictionary
    forbidden_targets = [
        "u_ped_mean",
        "u_ped_std",
        "u_median",
        "u_p95",
        "u_profile_175m",
        "Uped",
        "TKEped",
        "target",
        "y_true",
    ]
    for key in forbidden_targets:
        assert key not in record, f"Target leakage detected! Found forbidden key '{key}' in predictor record."

    # Verify essential predictor keys are present
    assert "lambda_p" in record
    assert "mean_h" in record
    assert "d_macdonald_m" in record
    assert "z0_macdonald_m" in record
    assert "d_kanda_m" in record
    assert "d_over_hmax_kanda" in record
    assert "z0_kanda_m" in record
    assert "z0_over_hmax_kanda" in record
    assert "dist_mean_m" in record
    assert "upwind_fetch_mean_m" in record
    assert "lambda_f_dir" in record


def test_pixel_tensor_shape_and_22_channels():
    """Verify aligned pixel tensor shape (22, Ny, Nx), channels, and float32 dtype."""
    H, dx = create_synthetic_benchmark()
    tensor, channel_names = assemble_pixel_feature_tensor(
        H=H,
        dx=dx,
        forcing_dir_deg=45.0,
        cfg=DEFAULT_CONFIG,
    )

    assert isinstance(tensor, np.ndarray)
    assert tensor.dtype == np.float32
    assert tensor.shape == (22, 40, 40)
    assert len(channel_names) == 22

    expected_channels = [
        "building_height",
        "fluid_mask",
        "dist_to_building",
        "lambda_p_8m",
        "lambda_p_16m",
        "lambda_p_32m",
        "lambda_p_64m",
        "h_mean_8m",
        "h_mean_16m",
        "h_mean_32m",
        "h_mean_64m",
        "upwind_fetch",
        "downwind_fetch",
        "lateral_left_fetch",
        "lateral_right_fetch",
        "upwind_h_1_10m",
        "upwind_h_11_30m",
        "upwind_h_31_60m",
        "canyon_width",
        "aspect_ratio_hw",
        "wind_cos",
        "wind_sin",
    ]
    assert channel_names == expected_channels

    # Check alignment and value bounds:
    # 1. Building height channel matches input
    np.testing.assert_allclose(tensor[0], H, atol=1e-5)

    # 2. Fluid mask is exactly 1 where H==0 and 0 where H>0
    bldg_mask = (H > 0.0)
    assert np.all(tensor[1][bldg_mask] == 0.0)
    assert np.all(tensor[1][~bldg_mask] == 1.0)

    # 3. Distance channel is 0 at building cells and strictly positive at open cells
    assert np.all(tensor[2][bldg_mask] == 0.0)
    assert np.all(tensor[2][~bldg_mask] > 0.0)

    # 4. Upwind building height bands are non-negative and zero at building cells
    assert np.all(tensor[15] >= 0.0)
    assert np.all(tensor[16] >= 0.0)
    assert np.all(tensor[17] >= 0.0)
    assert np.all(tensor[15][bldg_mask] == 0.0)
    assert np.all(tensor[16][bldg_mask] == 0.0)
    assert np.all(tensor[17][bldg_mask] == 0.0)

    # 5. Wind cos and sin fields are spatially uniform constants
    expected_cos = float(np.cos(np.radians(45.0)))
    expected_sin = float(np.sin(np.radians(45.0)))
    np.testing.assert_allclose(tensor[20], expected_cos, atol=1e-5)
    np.testing.assert_allclose(tensor[21], expected_sin, atol=1e-5)

    # Verify all tensor values are finite
    assert np.all(np.isfinite(tensor))
