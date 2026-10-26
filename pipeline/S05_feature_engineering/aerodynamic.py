"""
aerodynamic.py – Aerodynamic parameterizations (Macdonald 1998, Kanda et al. 2013).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from .config import DEFAULT_CONFIG, FeatureConfig


@dataclass(frozen=True)
class AerodynamicFeatures:
    """Aerodynamic parameters including displacement height and roughness length."""
    # Macdonald et al. (1998)
    d_macdonald_m: float
    d_over_h_macdonald: float
    z0_macdonald_m: float
    z0_over_h_macdonald: float

    # Kanda et al. (2013) height-variability adjusted (referenced to H_max)
    d_kanda_m: float
    d_over_hmax_kanda: float
    z0_kanda_m: float
    z0_over_hmax_kanda: float

    @property
    def d_over_h_kanda(self) -> float:
        """Alias for d_over_hmax_kanda (dimensionless ratio relative to H_max)."""
        return self.d_over_hmax_kanda

    @property
    def z0_over_h_kanda(self) -> float:
        """Alias for z0_over_hmax_kanda (dimensionless ratio relative to H_max)."""
        return self.z0_over_hmax_kanda

    def to_dict(self) -> Dict[str, Any]:
        """Convert aerodynamic parameters to dictionary."""
        return {
            "d_macdonald_m": self.d_macdonald_m,
            "d_over_h_macdonald": self.d_over_h_macdonald,
            "z0_macdonald_m": self.z0_macdonald_m,
            "z0_over_h_macdonald": self.z0_over_h_macdonald,
            "d_kanda_m": self.d_kanda_m,
            "d_over_hmax_kanda": self.d_over_hmax_kanda,
            "d_over_h_kanda": self.d_over_hmax_kanda,
            "z0_kanda_m": self.z0_kanda_m,
            "z0_over_hmax_kanda": self.z0_over_hmax_kanda,
            "z0_over_h_kanda": self.z0_over_hmax_kanda,
        }


def compute_aerodynamic_parameters(
    lambda_p: float,
    lambda_f: float,
    mean_h: float,
    std_h: float = 0.0,
    max_h: float | None = None,
    cfg: FeatureConfig = DEFAULT_CONFIG,
) -> AerodynamicFeatures:
    """Calculate displacement height and aerodynamic roughness length.

    Formulations:
    ------------
    1. Macdonald et al. (1998):
       d / H = 1 + alpha^(-lambda_p) * (lambda_p - 1)
       z0 / H = (1 - d/H) * exp( - [ 0.5 * beta * (CD / kappa^2) * (1 - d/H) * lambda_f ]^(-0.5) )

    2. Kanda et al. (2013):
       Accounts for height variability (sigma_H, H_max) over realistic urban canopies.

    Parameters
    ----------
    lambda_p : float
        Plan-area density in [0, 1].
    lambda_f : float
        Frontal-area density in [0, inf).
    mean_h : float
        Mean building height in metres.
    std_h : float, optional
        Standard deviation of building height in metres. Defaults to 0.0.
    max_h : float, optional
        Maximum building height in metres. Defaults to mean_h.
    cfg : FeatureConfig, optional
        Configuration with empirical coefficients.

    Returns
    -------
    AerodynamicFeatures
        Calculated aerodynamic properties.
    """
    # Defensive validation: inputs must be finite and non-negative
    for name, val in [("lambda_p", lambda_p), ("lambda_f", lambda_f), ("mean_h", mean_h), ("std_h", std_h)]:
        if not math.isfinite(val):
            raise ValueError(f"Input {name} must be finite, got {val}")
        if val < 0.0:
            raise ValueError(f"Input {name} must be non-negative, got {val}")

    if max_h is not None:
        if not math.isfinite(max_h):
            raise ValueError(f"Input max_h must be finite, got {max_h}")
        if max_h < 0.0:
            raise ValueError(f"Input max_h must be non-negative, got {max_h}")

    if max_h is None or max_h <= 0.0:
        max_h = mean_h
    else:
        max_h = max(max_h, mean_h)

    z0_ground = cfg.default_ground_roughness_m

    # Limiting case: no buildings or flat terrain
    if mean_h <= 1e-4 or lambda_p <= 1e-6:
        return AerodynamicFeatures(
            d_macdonald_m=0.0,
            d_over_h_macdonald=0.0,
            z0_macdonald_m=z0_ground,
            z0_over_h_macdonald=0.0,
            d_kanda_m=0.0,
            d_over_hmax_kanda=0.0,
            z0_kanda_m=z0_ground,
            z0_over_hmax_kanda=0.0,
        )

    # Clip densities to physical bounds
    lp = float(np.clip(lambda_p, 1e-6, 0.999))
    lf = float(max(1e-6, lambda_f))

    # --- 1. Macdonald et al. (1998) ---
    alpha = cfg.macdonald_alpha
    kappa = cfg.von_karman_constant
    cd = cfg.macdonald_cd
    beta = cfg.macdonald_beta

    # d / H = 1 + alpha^(-lp) * (lp - 1)
    d_over_h_mac = 1.0 + (alpha ** (-lp)) * (lp - 1.0)
    d_over_h_mac = float(np.clip(d_over_h_mac, 0.0, 0.999))
    d_mac_m = float(d_over_h_mac * mean_h)

    # Term: 0.5 * beta * (CD / kappa^2) * (1 - d/H) * lf
    drag_term = 0.5 * beta * (cd / (kappa**2)) * (1.0 - d_over_h_mac) * lf
    if drag_term > 1e-8:
        exponent = -1.0 / math.sqrt(drag_term)
        # Avoid underflow
        z0_over_h_mac = float((1.0 - d_over_h_mac) * math.exp(max(-50.0, exponent)))
        z0_mac_m = float(max(z0_ground, z0_over_h_mac * mean_h))
    else:
        z0_over_h_mac = 0.0
        z0_mac_m = z0_ground

    # --- 2. Kanda et al. (2013) ---
    # Reference:
    #   Kanda, M., Inagaki, A., Miyamoto, T., Gryschka, M., & Raasch, S. (2013).
    #   A New Aerodynamic Parametrization for Real Urban Surfaces.
    #   Boundary-Layer Meteorology, 148(2), 357-377. DOI: 10.1007/s10546-013-9818-x.
    #
    # Formulations:
    #   - Displacement height ratio d / H_max: Eq. (10)-(11)
    #     d/H_max = c0 * X^2 + (a0 * lp^b0 - c0 * X^2) * (1 - a1 * exp(-b1 * Y))
    #     where:
    #       X = (std_h + mean_h) / max_h  (height variability ratio; X > 0;
    #         X = 1 for uniform height, X < 1 for right-skewed arrays with tall outliers,
    #         and X > 1 is mathematically and physically valid for left-skewed arrays
    #         where std_h + mean_h > max_h; observed up to ~1.14 in realistic cities)
    #       Y = lambda_f * (std_h / mean_h) (interaction term)
    #       a0 = 1.29, b0 = 0.36, c0 = 0.17, a1 = 0.17, b1 = 0.05
    #   - Roughness length ratio z0 / H_max: Eq. (12)-(13)
    #     z0/H_max = (1 - d/H_max) * (a2 * lf^b2 * exp(-c2 * lf)) * [1 + 0.8 * (std_h / mean_h)]
    #     where:
    #       a2 = 0.71, b2 = 0.65, c2 = 1.55
    a0, b0, c0 = 1.29, 0.36, 0.17
    a1, b1 = 0.17, 0.05

    X = float((std_h + mean_h) / max_h) if max_h > 0 else 1.0
    Y = float(lf * (std_h / mean_h)) if mean_h > 0 else 0.0

    X2 = c0 * (X**2)
    # d / H_max
    term_d = (a0 * (lp**b0) - X2) * (1.0 - a1 * math.exp(-b1 * Y))
    d_over_hmax_kanda = float(np.clip(X2 + term_d, 0.0, 0.999))
    d_kanda_m = float(d_over_hmax_kanda * max_h)

    # Roughness length z0_kanda
    # Kanda et al. (2013) Eq. (12)
    # z0 / H_max = (1 - d/H_max) * (a2 * lf^b2) * exp(-c2 * lf)
    # where a2 = 0.71, b2 = 0.65, c2 = 1.55
    a2, b2, c2 = 0.71, 0.65, 1.55
    z0_term = a2 * (lf**b2) * math.exp(-c2 * lf)
    # Enhancement factor due to height variability (Kanda 2013 Eq 13)
    if mean_h > 0 and std_h > 0:
        f_var = 1.0 + 0.8 * (std_h / mean_h)
    else:
        f_var = 1.0

    z0_over_hmax_kanda = float((1.0 - d_over_hmax_kanda) * z0_term * f_var)
    z0_kanda_m = float(max(z0_ground, z0_over_hmax_kanda * max_h))

    return AerodynamicFeatures(
        d_macdonald_m=d_mac_m,
        d_over_h_macdonald=d_over_h_mac,
        z0_macdonald_m=z0_mac_m,
        z0_over_h_macdonald=z0_over_h_mac,
        d_kanda_m=d_kanda_m,
        d_over_hmax_kanda=d_over_hmax_kanda,
        z0_kanda_m=z0_kanda_m,
        z0_over_hmax_kanda=z0_over_hmax_kanda,
    )
