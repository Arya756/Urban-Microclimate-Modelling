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


def test_kanda_invalid_inputs_raise_errors():
    """Verify that non-finite (NaN, Inf) and negative inputs raise ValueError."""
    # NaN inputs
    with pytest.raises(ValueError, match="Input lambda_p must be finite"):
        compute_aerodynamic_parameters(float("nan"), 0.25, 16.0, 0.0, 16.0)

    with pytest.raises(ValueError, match="Input lambda_f must be finite"):
        compute_aerodynamic_parameters(0.25, float("nan"), 16.0, 0.0, 16.0)

    with pytest.raises(ValueError, match="Input mean_h must be finite"):
        compute_aerodynamic_parameters(0.25, 0.25, float("nan"), 0.0, 16.0)

    with pytest.raises(ValueError, match="Input std_h must be finite"):
        compute_aerodynamic_parameters(0.25, 0.25, 16.0, float("nan"), 16.0)

    with pytest.raises(ValueError, match="Input max_h must be finite"):
        compute_aerodynamic_parameters(0.25, 0.25, 16.0, 0.0, float("nan"))

    # Inf inputs
    with pytest.raises(ValueError, match="Input lambda_p must be finite"):
        compute_aerodynamic_parameters(float("inf"), 0.25, 16.0, 0.0, 16.0)

    # Negative inputs
    with pytest.raises(ValueError, match="Input lambda_p must be non-negative"):
        compute_aerodynamic_parameters(-0.1, 0.25, 16.0, 0.0, 16.0)

    with pytest.raises(ValueError, match="Input mean_h must be non-negative"):
        compute_aerodynamic_parameters(0.25, 0.25, -5.0, 0.0, 16.0)


def test_kanda_max_h_guard():
    """Verify that max_h < mean_h is guarded safely to max(mean_h, max_h)."""
    # If caller specifies max_h=10 < mean_h=20, max_h is clamped to 20
    aero = compute_aerodynamic_parameters(
        lambda_p=0.25,
        lambda_f=0.25,
        mean_h=20.0,
        std_h=0.0,
        max_h=10.0,
    )
    # With max_h clamped to mean_h=20, behavior matches uniform canopy
    aero_expected = compute_aerodynamic_parameters(
        lambda_p=0.25,
        lambda_f=0.25,
        mean_h=20.0,
        std_h=0.0,
        max_h=20.0,
    )
    assert abs(aero.d_over_hmax_kanda - aero_expected.d_over_hmax_kanda) < 1e-6
    assert abs(aero.d_kanda_m - aero_expected.d_kanda_m) < 1e-6


def test_kanda_duplicate_feature_aliases_exact_match():
    """Verify that d_over_h_kanda and z0_over_h_kanda are exact aliases of their hmax counterparts."""
    aero = compute_aerodynamic_parameters(
        lambda_p=0.3,
        lambda_f=0.2,
        mean_h=18.0,
        std_h=6.0,
        max_h=30.0,
    )
    # Properties match
    assert aero.d_over_h_kanda == aero.d_over_hmax_kanda
    assert aero.z0_over_h_kanda == aero.z0_over_hmax_kanda

    # Dictionary serialization produces exact duplicate pairs for backwards compatibility
    d = aero.to_dict()
    assert d["d_over_h_kanda"] == d["d_over_hmax_kanda"]
    assert d["z0_over_h_kanda"] == d["z0_over_hmax_kanda"]


def test_kanda_extreme_densities_bounded():
    """Verify behavior at extreme densities (near-zero and near-one)."""
    # Very sparse canopy
    aero_sparse = compute_aerodynamic_parameters(
        lambda_p=0.001, lambda_f=0.001, mean_h=15.0, std_h=2.0, max_h=20.0
    )
    assert 0.0 < aero_sparse.d_over_hmax_kanda < 1.0
    assert 0.0 < aero_sparse.z0_over_hmax_kanda < 0.5
    assert aero_sparse.d_kanda_m < 20.0

    # Very dense canopy (skimming flow)
    aero_dense = compute_aerodynamic_parameters(
        lambda_p=0.95, lambda_f=0.8, mean_h=15.0, std_h=2.0, max_h=20.0
    )
    assert 0.0 < aero_dense.d_over_hmax_kanda < 1.0
    # In dense canopies, d approaches canopy height and z0 decreases
    assert aero_dense.d_over_hmax_kanda > aero_sparse.d_over_hmax_kanda
    assert aero_dense.z0_over_hmax_kanda < aero_sparse.z0_over_hmax_kanda


def test_kanda_height_variability_exceeding_max_h():
    """Verify that cases with left-skewed distributions where std_h + mean_h > max_h (X > 1)

    are mathematically well-behaved and physically plausible without clipping.
    """
    # Parameters from real case KO-SE-V9_d00 (Seoul):
    # mean_h = 40.828, std_h = 15.999, max_h = 50.0 -> X = (40.828 + 15.999)/50.0 = 1.1365 > 1.0
    lp = 0.28
    lf = 0.22
    mean_h = 40.828
    std_h = 15.999
    max_h = 50.0

    X = (std_h + mean_h) / max_h
    assert X > 1.0

    aero = compute_aerodynamic_parameters(
        lambda_p=lp,
        lambda_f=lf,
        mean_h=mean_h,
        std_h=std_h,
        max_h=max_h,
    )

    # 1. d / H_max must be strictly within physical bounds (0, 1)
    assert 0.0 < aero.d_over_hmax_kanda < 1.0
    # Must not hit the artificial boundary clips (0.0 or 0.999)
    assert aero.d_over_hmax_kanda < 0.99
    assert aero.d_over_hmax_kanda > 0.1

    # 2. Physical displacement height is strictly less than max_h
    assert 0.0 < aero.d_kanda_m < max_h

    # 3. Roughness length is positive and bounded
    assert 0.0 < aero.z0_over_hmax_kanda < 0.5
    assert aero.z0_kanda_m > DEFAULT_CONFIG.default_ground_roughness_m


def test_kanda_preservation_of_real_case_features():
    """Verify that recomputing aerodynamic parameters on actual cases matches case_features.csv exactly."""
    import pathlib
    import pandas as pd

    csv_path = pathlib.Path("pipeline/S05_feature_engineering/outputs/case_features.csv")
    if not csv_path.exists():
        pytest.skip("case_features.csv not present")

    df = pd.read_csv(csv_path)
    # Test a sample of 20 diverse cases: including uniform (X=1), right-skewed (X<1), and left-skewed (X>1)
    sample_cases = df.sample(n=min(20, len(df)), random_state=42)

    for _, row in sample_cases.iterrows():
        aero = compute_aerodynamic_parameters(
            lambda_p=row["lambda_p"],
            lambda_f=row["lambda_f_dir"],
            mean_h=row["mean_h"],
            std_h=row["std_h"],
            max_h=row["max_h"],
        )
        assert abs(aero.d_kanda_m - row["d_kanda_m"]) < 1e-5
        assert abs(aero.z0_kanda_m - row["z0_kanda_m"]) < 1e-5
        assert abs(aero.d_over_hmax_kanda - row["d_over_hmax_kanda"]) < 1e-5
        assert abs(aero.z0_over_hmax_kanda - row["z0_over_hmax_kanda"]) < 1e-5
