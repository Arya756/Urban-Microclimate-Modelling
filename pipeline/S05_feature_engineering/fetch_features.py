"""
fetch_features.py – Upwind, downwind, lateral fetch, and directional building exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class FetchSummary:
    """Case-level summary statistics of directional fetch across valid fluid cells."""
    upwind_fetch_mean_m: float
    upwind_fetch_median_m: float
    downwind_fetch_mean_m: float
    lateral_fetch_mean_m: float
    unobstructed_fraction_gt50m: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert fetch summaries to dictionary."""
        return {
            "upwind_fetch_mean_m": self.upwind_fetch_mean_m,
            "upwind_fetch_median_m": self.upwind_fetch_median_m,
            "downwind_fetch_mean_m": self.downwind_fetch_mean_m,
            "lateral_fetch_mean_m": self.lateral_fetch_mean_m,
            "unobstructed_fraction_gt50m": self.unobstructed_fraction_gt50m,
        }


def _trace_periodic_ray_fetch(
    bldg_mask: np.ndarray,
    vx: float,
    vy: float,
    dx: float,
    dy: float,
    cap_m: float,
    step_m: float = 1.0,
) -> np.ndarray:
    """Trace directional ray over periodic grid to find distance to nearest obstacle.

    Parameters
    ----------
    bldg_mask : np.ndarray
        Boolean 2D array of building cells (True for building, False for open).
    vx, vy : float
        Direction unit vector in physical coordinates (x East, y North).
    dx, dy : float
        Grid spacing in metres.
    cap_m : float
        Maximum distance cutoff in metres.
    step_m : float, optional
        Raymarch step length in metres. Defaults to 1.0 m.

    Returns
    -------
    np.ndarray
        2D array of distances to nearest building obstacle in metres (Ny, Nx).
    """
    ny, nx = bldg_mask.shape
    fetch_map = np.full((ny, nx), cap_m, dtype=np.float32)

    # Building cells have zero fetch
    fetch_map[bldg_mask] = 0.0
    active_open = ~bldg_mask
    still_searching = active_open.copy()

    n_steps = max(1, int(math.ceil(cap_m / step_m)))

    for step in range(1, n_steps + 1):
        dist = min(step * step_m, cap_m)
        # Pixel offsets along ray
        shift_x = int(round((dist * vx) / dx))
        shift_y = int(round((dist * vy) / dy))

        if shift_x == 0 and shift_y == 0:
            continue

        # Look in direction (vx, vy): obstacle is at (y + shift_y, x + shift_x)
        # Shift mask so that cell (y, x) inspects (y + shift_y, x + shift_x)
        inspected = np.roll(bldg_mask, shift=(-shift_y, -shift_x), axis=(0, 1))
        hits = still_searching & inspected

        if np.any(hits):
            fetch_map[hits] = float(dist)
            still_searching[hits] = False

        if not np.any(still_searching):
            break

    return fetch_map


def compute_directional_fetch_maps(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
    forcing_dir_deg: float = 0.0,
    upwind_cap_m: float = 150.0,
    downwind_cap_m: float = 60.0,
    lateral_cap_m: float = 60.0,
) -> Tuple[Dict[str, np.ndarray], FetchSummary]:
    """Compute upwind, downwind, and lateral fetch maps with periodic boundaries.

    Directions:
    -----------
    - Wind vector: (cos(theta), sin(theta)) where theta = forcing_dir_deg.
    - Upwind (opposite flow): (-cos(theta), -sin(theta)).
    - Downwind (along flow): (+cos(theta), +sin(theta)).
    - Lateral left (+90 deg): (-sin(theta), +cos(theta)).
    - Lateral right (-90 deg): (+sin(theta), -cos(theta)).

    Parameters
    ----------
    H : np.ndarray
        Building height raster array (Ny, Nx).
    dx : float
        Grid resolution in x-direction (metres).
    dy : float, optional
        Grid resolution in y-direction (metres).
    forcing_dir_deg : float
        Wind forcing direction in degrees (0 deg = along +x).
    upwind_cap_m : float
        Upwind fetch ceiling in metres (default 150m).
    downwind_cap_m : float
        Downwind fetch ceiling in metres (default 60m).
    lateral_cap_m : float
        Lateral fetch ceiling in metres (default 60m).

    Returns
    -------
    Tuple[Dict[str, np.ndarray], FetchSummary]
        - fetch_maps: Dictionary of 2D spatial fetch arrays.
        - summary: Summary statistics over fluid cells.
    """
    if dy is None:
        dy = dx

    bldg_mask = (H > 0.0)
    theta_rad = math.radians(forcing_dir_deg % 360.0)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Unit vectors
    # Upwind (looking backward into incoming flow)
    vx_up, vy_up = -cos_t, -sin_t
    # Downwind (looking forward with flow)
    vx_down, vy_down = cos_t, sin_t
    # Lateral left (+90 deg counterclockwise)
    vx_left, vy_left = -sin_t, cos_t
    # Lateral right (-90 deg clockwise)
    vx_right, vy_right = sin_t, -cos_t

    step_size = min(dx, dy)

    upwind_map = _trace_periodic_ray_fetch(bldg_mask, vx_up, vy_up, dx, dy, upwind_cap_m, step_size)
    downwind_map = _trace_periodic_ray_fetch(bldg_mask, vx_down, vy_down, dx, dy, downwind_cap_m, step_size)
    left_map = _trace_periodic_ray_fetch(bldg_mask, vx_left, vy_left, dx, dy, lateral_cap_m, step_size)
    right_map = _trace_periodic_ray_fetch(bldg_mask, vx_right, vy_right, dx, dy, lateral_cap_m, step_size)

    # Upwind building height within distance bands [1-10m, 11-30m, 31-60m]
    bands = (
        ("upwind_h_1_10m", 1.0, 10.0),
        ("upwind_h_11_30m", 11.0, 30.0),
        ("upwind_h_31_60m", 31.0, 60.0),
    )
    band_height_maps = {}

    for b_idx, (b_name, d_min, d_max) in enumerate(bands, start=1):
        band_max_h = np.zeros_like(H, dtype=np.float32)
        n_band_steps = max(1, int(math.ceil((d_max - d_min) / step_size)))
        for s in range(n_band_steps + 1):
            d = d_min + s * step_size
            if d > d_max:
                break
            sx = int(round((d * vx_up) / dx))
            sy = int(round((d * vy_up) / dy))
            h_shifted = np.roll(H, shift=(-sy, -sx), axis=(0, 1))
            band_max_h = np.maximum(band_max_h, h_shifted)

        band_max_h[bldg_mask] = 0.0
        band_height_maps[b_name] = band_max_h.astype(np.float32)
        band_height_maps[f"upwind_h_band_{b_idx}"] = band_max_h.astype(np.float32)

    fetch_maps = {
        "upwind_fetch": upwind_map,
        "downwind_fetch": downwind_map,
        "lateral_left_fetch": left_map,
        "lateral_right_fetch": right_map,
        **band_height_maps,
    }

    # Summary statistics across valid fluid cells
    fluid_mask = ~bldg_mask
    if np.any(fluid_mask):
        up_fluid = upwind_map[fluid_mask]
        down_fluid = downwind_map[fluid_mask]
        left_fluid = left_map[fluid_mask]
        right_fluid = right_map[fluid_mask]
        lat_fluid = 0.5 * (left_fluid + right_fluid)

        up_mean = float(np.mean(up_fluid))
        up_med = float(np.median(up_fluid))
        down_mean = float(np.mean(down_fluid))
        lat_mean = float(np.mean(lat_fluid))
        unobstructed_gt50 = float(np.mean(up_fluid >= 50.0))
    else:
        up_mean = up_med = down_mean = lat_mean = unobstructed_gt50 = 0.0

    summary = FetchSummary(
        upwind_fetch_mean_m=up_mean,
        upwind_fetch_median_m=up_med,
        downwind_fetch_mean_m=down_mean,
        lateral_fetch_mean_m=lat_mean,
        unobstructed_fraction_gt50m=unobstructed_gt50,
    )

    return fetch_maps, summary
