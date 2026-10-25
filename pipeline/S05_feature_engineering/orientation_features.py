"""
orientation_features.py – Frontal area index, wall area, facet orientations, and directional exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class OrientationFeatures:
    """Directional morphology, frontal area, and orientation features."""
    lambda_f_x: float
    lambda_f_y: float
    lambda_f_dir: float
    lambda_w: float
    windward_wall_share: float
    forcing_dir_deg: float
    wind_cos: float
    wind_sin: float
    dominant_facet_angle_deg: float
    facet_orientation_entropy: float
    relative_wind_facet_angle_deg: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary."""
        return {
            "lambda_f_x": self.lambda_f_x,
            "lambda_f_y": self.lambda_f_y,
            "lambda_f_dir": self.lambda_f_dir,
            "lambda_w": self.lambda_w,
            "windward_wall_share": self.windward_wall_share,
            "forcing_dir_deg": self.forcing_dir_deg,
            "wind_cos": self.wind_cos,
            "wind_sin": self.wind_sin,
            "dominant_facet_angle_deg": self.dominant_facet_angle_deg,
            "facet_orientation_entropy": self.facet_orientation_entropy,
            "relative_wind_facet_angle_deg": self.relative_wind_facet_angle_deg,
        }


def compute_directional_morphology(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
    forcing_dir_deg: float = 0.0,
) -> OrientationFeatures:
    """Compute direction-dependent frontal area index, wall area, and facet orientations.

    Parameters
    ----------
    H : np.ndarray
        2D array of building heights in metres.
    dx : float
        Grid resolution in x-direction (metres).
    dy : float, optional
        Grid resolution in y-direction (metres). Defaults to dx.
    forcing_dir_deg : float
        Prevailing flow direction in degrees (0 deg = along +x / West-to-East).

    Returns
    -------
    OrientationFeatures
        Calculated directional and facade morphology metrics.
    """
    if dy is None:
        dy = dx

    ny, nx = H.shape
    domain_area_m2 = float(nx * dx * ny * dy)

    # Wrap periodic steps along X (wind along X encounters jumps H(y, x) - H(y, x-1))
    H_roll_x = np.roll(H, shift=1, axis=1)
    jump_x = np.maximum(0.0, H - H_roll_x)
    frontal_area_x = float(np.sum(jump_x) * dy)
    lambda_f_x = float(frontal_area_x / domain_area_m2) if domain_area_m2 > 0 else 0.0

    # Wrap periodic steps along Y (wind along Y encounters jumps H(y, x) - H(y-1, x))
    H_roll_y = np.roll(H, shift=1, axis=0)
    jump_y = np.maximum(0.0, H - H_roll_y)
    frontal_area_y = float(np.sum(jump_y) * dx)
    lambda_f_y = float(frontal_area_y / domain_area_m2) if domain_area_m2 > 0 else 0.0

    # Total wall area (each step has an upward face and a downward face)
    total_wall_area_m2 = 2.0 * frontal_area_x + 2.0 * frontal_area_y
    lambda_w = float(total_wall_area_m2 / domain_area_m2) if domain_area_m2 > 0 else 0.0

    # Direction-dependent frontal area index for arbitrary forcing angle theta
    theta_rad = math.radians(forcing_dir_deg % 360.0)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Projected frontal area index along wind vector (Grimmond & Oke 1999)
    lambda_f_dir = float(abs(cos_t) * lambda_f_x + abs(sin_t) * lambda_f_y)

    # Windward wall share (fraction of total vertical facades exposed to incoming flow)
    windward_wall_share = float(lambda_f_dir / lambda_w) if lambda_w > 1e-7 else 0.0

    # Facet orientation analysis via perimeter gradients
    bldg_mask = (H > 0.0).astype(float)
    if np.any(bldg_mask > 0):
        # Sobel gradients with periodic wrapping
        grad_y = ndimage.sobel(bldg_mask, axis=0, mode="wrap")
        grad_x = ndimage.sobel(bldg_mask, axis=1, mode="wrap")
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        edge_mask = grad_mag > 1e-3
        if np.any(edge_mask):
            angles_rad = np.arctan2(grad_y[edge_mask], grad_x[edge_mask])
            angles_deg = (np.degrees(angles_rad) % 180.0)  # fold to [0, 180) for facet lines

            # 18 bins of 10 degrees across [0, 180)
            hist, bin_edges = np.histogram(angles_deg, bins=18, range=(0.0, 180.0), density=True)
            dominant_bin = int(np.argmax(hist))
            dominant_facet_angle_deg = float(0.5 * (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]))

            # Shannon entropy of facet orientation distribution
            probs = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)
            nonzero_p = probs[probs > 0]
            entropy = float(-np.sum(nonzero_p * np.log2(nonzero_p))) if len(nonzero_p) > 0 else 0.0

            # Angle difference between wind direction and dominant facet orientation
            # Reduced modulo 90 degrees (canyon alignment symmetry)
            angle_diff = abs((forcing_dir_deg % 180.0) - dominant_facet_angle_deg)
            rel_angle = float(min(angle_diff, 180.0 - angle_diff))
            rel_angle_90 = float(min(rel_angle, abs(90.0 - rel_angle)))
        else:
            dominant_facet_angle_deg = 0.0
            entropy = 0.0
            rel_angle_90 = 0.0
    else:
        dominant_facet_angle_deg = 0.0
        entropy = 0.0
        rel_angle_90 = 0.0

    return OrientationFeatures(
        lambda_f_x=lambda_f_x,
        lambda_f_y=lambda_f_y,
        lambda_f_dir=lambda_f_dir,
        lambda_w=lambda_w,
        windward_wall_share=windward_wall_share,
        forcing_dir_deg=float(forcing_dir_deg),
        wind_cos=float(cos_t),
        wind_sin=float(sin_t),
        dominant_facet_angle_deg=dominant_facet_angle_deg,
        facet_orientation_entropy=entropy,
        relative_wind_facet_angle_deg=rel_angle_90,
    )
