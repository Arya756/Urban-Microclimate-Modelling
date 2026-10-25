"""
test_aerodynamic.py – Unit tests for Macdonald (1998) and Kanda et al. (2013) aerodynamic parameterizations.
"""

import math
import pytest

from pipeline.S05_feature_engineering.aerodynamic import compute_aerodynamic_parameters
from pipeline.S05_feature_engineering.config import DEFAULT_CONFIG


def test_macdonald_exact_analytical_values():
    """Verify exact analytical values of Macdonald et al. (1998) equations."""
    # Parameters
    lp = 0.25
    lf = 0.25
    mean_h = 16.0

    aero = compute_aerodynamic_parameters(
        lambda_p=lp,
        lambda_f=lf,
        mean_h=mean_h,
        std_h=0.0,
        max_h=16.0,
    )

    # 1. Displacement height: d / H = 1 + alpha^(-lp) * (lp - 1)
    # alpha = 4.43
    expected_d_over_h = 1.0 + (4.43 ** (-0.25)) * (0.25 - 1.0)
    expected_d_m = expected_d_over_h * mean_h

    assert abs(aero.d_over_h_macdonald - expected_d_over_h) < 1e-6
    assert abs(aero.d_macdonald_m - expected_d_m) < 1e-6
    assert abs(aero.d_over_h_macdonald - 0.48303596) < 1e-5

    # 2. Roughness length:
    # drag_term = 0.5 * beta * (CD / kappa^2) * (1 - d/H) * lf
    # beta = 1.0, CD = 1.2, kappa = 0.4 -> CD / kappa^2 = 7.5
    drag_term = 0.5 * 1.0 * (1.2 / (0.4**2)) * (1.0 - expected_d_over_h) * lf
    expected_z0_over_h = (1.0 - expected_d_over_h) * math.exp(-1.0 / math.sqrt(drag_term))
    expected_z0_m = expected_z0_over_h * mean_h

    assert abs(aero.z0_over_h_macdonald - expected_z0_over_h) < 1e-6
    assert abs(aero.z0_macdonald_m - expected_z0_m) < 1e-6
    assert abs(aero.z0_over_h_macdonald - 0.122921) < 1e-5


def test_kanda_uniform_height_exact_values():
    """Verify exact analytical values of Kanda et al. (2013) on uniform canopy (std_h=0)."""
    lp = 0.25
    lf = 0.25
    mean_h = 16.0
    max_h = 16.0
    std_h = 0.0

    aero = compute_aerodynamic_parameters(
        lambda_p=lp,
        lambda_f=lf,
        mean_h=mean_h,
        std_h=std_h,
        max_h=max_h,
    )

    # X = (0 + 16) / 16 = 1.0, Y = 0.25 * (0 / 16) = 0.0
    # a0=1.29, b0=0.36, c0=0.17, a1=0.17, b1=0.05
    # X2 = 0.17 * 1^2 = 0.17
    # a0 * lp^b0 = 1.29 * 0.25^0.36 = 0.78320708
    # term_d = (0.78320708 - 0.17) * (1 - 0.17 * exp(0)) = 0.61320708 * 0.83 = 0.50896188
    # d / H_max = 0.17 + 0.50896188 = 0.67896188
    expected_d_over_hmax = 0.17 + (1.29 * (0.25**0.36) - 0.17) * (1.0 - 0.17)
    expected_d_m = expected_d_over_hmax * max_h

    assert abs(aero.d_over_hmax_kanda - expected_d_over_hmax) < 1e-6
    assert abs(aero.d_kanda_m - expected_d_m) < 1e-6
    assert abs(aero.d_over_hmax_kanda - 0.67896188) < 1e-4

    # Roughness length:
    # a2=0.71, b2=0.65, c2=1.55
    # z0_term = 0.71 * 0.25^0.65 * exp(-1.55 * 0.25) = 0.1957193
    # f_var = 1.0
    # z0 / H_max = (1 - 0.67896188) * 0.1957193 * 1.0 = 0.0628331
    expected_z0_term = 0.71 * (0.25**0.65) * math.exp(-1.55 * 0.25)
    expected_z0_over_hmax = (1.0 - expected_d_over_hmax) * expected_z0_term * 1.0
    expected_z0_m = expected_z0_over_hmax * max_h

    assert abs(aero.z0_over_hmax_kanda - expected_z0_over_hmax) < 1e-6
    assert abs(aero.z0_kanda_m - expected_z0_m) < 1e-6
    assert abs(aero.z0_over_hmax_kanda - 0.0628331) < 1e-4

    # Backward compatibility alias
    assert aero.d_over_h_kanda == aero.d_over_hmax_kanda
    assert aero.z0_over_h_kanda == aero.z0_over_hmax_kanda


def test_kanda_with_height_variability_referenced_to_hmax():
    """Verify Kanda parameterization with realistic height variability (H_max > mean_h)."""
    lp = 0.25
    lf = 0.25
    mean_h = 16.0
    std_h = 8.0
    max_h = 32.0

    aero = compute_aerodynamic_parameters(
        lambda_p=lp,
        lambda_f=lf,
        mean_h=mean_h,
        std_h=std_h,
        max_h=max_h,
    )

    # 1. d_over_hmax_kanda must be strictly between 0 and 1
    assert 0.0 < aero.d_over_hmax_kanda < 1.0
    # 2. d_kanda_m must strictly equal d_over_hmax_kanda * max_h
    assert abs(aero.d_kanda_m - aero.d_over_hmax_kanda * max_h) < 1e-6
    # 3. Displacement height must not exceed max_h
    assert aero.d_kanda_m < max_h

    # 4. Roughness length ratio z0 / H_max must be positive and bounded (< 0.5)
    assert 0.0 < aero.z0_over_hmax_kanda < 0.5
    # 5. z0_kanda_m must strictly equal z0_over_hmax_kanda * max_h
    assert abs(aero.z0_kanda_m - aero.z0_over_hmax_kanda * max_h) < 1e-6

    # 6. Verify analytical d_over_hmax calculation
    X = (8.0 + 16.0) / 32.0  # 0.75
    Y = 0.25 * (8.0 / 16.0)  # 0.125
    X2 = 0.17 * (X**2)       # 0.095625
    term_d = (1.29 * (0.25**0.36) - X2) * (1.0 - 0.17 * math.exp(-0.05 * Y))
    expected_d_over_hmax = X2 + term_d
    assert abs(aero.d_over_hmax_kanda - expected_d_over_hmax) < 1e-6

    # 7. Height variability increases turbulent drag: variable canopy z0 > uniform canopy z0
    aero_uniform = compute_aerodynamic_parameters(
        lambda_p=lp, lambda_f=lf, mean_h=mean_h, std_h=0.0, max_h=mean_h
    )
    assert aero.z0_kanda_m > aero_uniform.z0_kanda_m


def test_aerodynamic_limiting_case_flat_terrain():
    """Verify limiting behavior when lambda_p = 0 or H = 0."""
    aero = compute_aerodynamic_parameters(
        lambda_p=0.0,
        lambda_f=0.0,
        mean_h=0.0,
        std_h=0.0,
    )
    assert aero.d_macdonald_m == 0.0
    assert aero.d_over_h_macdonald == 0.0
    assert aero.z0_macdonald_m == DEFAULT_CONFIG.default_ground_roughness_m
    assert aero.z0_over_h_macdonald == 0.0

    assert aero.d_kanda_m == 0.0
    assert aero.d_over_hmax_kanda == 0.0
    assert aero.z0_kanda_m == DEFAULT_CONFIG.default_ground_roughness_m
    assert aero.z0_over_hmax_kanda == 0.0
