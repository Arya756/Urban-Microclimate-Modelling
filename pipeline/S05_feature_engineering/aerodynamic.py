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

    2. Kanda et al. (2013), for canopies with variable heights:
       d / H_max = c0 X^2 + (a0 lambda_p^b0 - c0) X,   X = (sigma_H + H_mean) / H_max
       z0 = (b1 Y^2 + c1 Y + a1) z0_Macdonald,          Y = lambda_p sigma_H / H_mean

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
    # Kanda, M., Inagaki, A., Miyamoto, T., Gryschka, M., & Raasch, S. (2013).
    # A New Aerodynamic Parametrization for Real Urban Surfaces.
    # Boundary-Layer Meteorology, 148(2), 357-377. DOI: 10.1007/s10546-013-9818-x.
    # Fitted to LES of real city blocks; constants as implemented in UMEP
    # (Kent et al. 2017, RoughnessCalcFunction):
    #   d / H_max = c0 * X^2 + (a0 * lp^b0 - c0) * X,   X = (std_h + mean_h) / H_max,  for 0 < X <= 1
    #   d         = a0 * lp^b0 * mean_h                  when X > 1 (outside the fitted range)
    #   z0        = (b1 * Y^2 + c1 * Y + a1) * z0_Macdonald,   Y = lp * std_h / mean_h
    #   a0 = 1.29, b0 = 0.36, c0 = -0.17, a1 = 0.71, b1 = 20.21, c1 = -0.77
    # For uniform heights (std_h = 0) X = 1 and Y = 0, so d = a0 * lp^b0 * H and z0 = a1 * z0_Macdonald.
    a0, b0, c0 = 1.29, 0.36, -0.17
    a1, b1, c1 = 0.71, 20.21, -0.77

    X = (std_h + mean_h) / max_h
    if X <= 1.0:
        d_over_hmax_kanda = c0 * X**2 + (a0 * lp**b0 - c0) * X
    else:
        d_over_hmax_kanda = a0 * lp**b0 * mean_h / max_h
    # d cannot exceed the tallest building (the fit overshoots for lp above about 0.55)
    d_over_hmax_kanda = float(np.clip(d_over_hmax_kanda, 0.0, 0.999))
    d_kanda_m = float(d_over_hmax_kanda * max_h)

    Y = lp * std_h / mean_h
    z0_kanda_m = float(max(z0_ground, (b1 * Y**2 + c1 * Y + a1) * z0_over_h_mac * mean_h))
    z0_over_hmax_kanda = float(z0_kanda_m / max_h)

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
