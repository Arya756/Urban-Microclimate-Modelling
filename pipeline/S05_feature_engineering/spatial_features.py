"""
spatial_features.py – Periodic Euclidean distance, multi-scale local density, and height maps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple
import numpy as np
from scipy import ndimage



@dataclass(frozen=True)
class DistanceSummary:
    """Case-level summary of distance to nearest building across valid open-space cells."""
    dist_mean_m: float
    dist_std_m: float
    dist_median_m: float
    dist_p25_m: float
    dist_p75_m: float
    dist_p90_m: float
    dist_max_m: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert distance summaries to dictionary."""
        return {
            "dist_mean_m": self.dist_mean_m,
            "dist_std_m": self.dist_std_m,
            "dist_median_m": self.dist_median_m,
            "dist_p25_m": self.dist_p25_m,
            "dist_p75_m": self.dist_p75_m,
            "dist_p90_m": self.dist_p90_m,
            "dist_max_m": self.dist_max_m,
        }


def compute_periodic_distance_transform(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
) -> Tuple[np.ndarray, DistanceSummary]:
    """Compute periodic Euclidean distance transform to the nearest building.

    Periodic boundaries are handled exactly by 3x3 periodic tiling before
    evaluating the Euclidean distance transform.

    Parameters
    ----------
    H : np.ndarray
        2D height map array (Ny, Nx).
    dx : float
        Grid resolution in x-direction (metres).
    dy : float, optional
        Grid resolution in y-direction (metres). Defaults to dx.

    Returns
    -------
    Tuple[np.ndarray, DistanceSummary]
        - dist_map: 2D array of distances in metres, shape (Ny, Nx).
        - summary: Summary statistics over fluid cells (where H == 0).
    """
    if dy is None:
        dy = dx

    ny, nx = H.shape
    bldg_mask = (H > 0.0)

    # Edge case: No buildings
    if not np.any(bldg_mask):
        # Distance to buildings is undefined/inf; cap at domain diagonal
        diag_dist = float(np.sqrt((nx * dx) ** 2 + (ny * dy) ** 2))
        dist_map = np.full((ny, nx), diag_dist, dtype=np.float32)
        summary = DistanceSummary(
            dist_mean_m=diag_dist,
            dist_std_m=0.0,
            dist_median_m=diag_dist,
            dist_p25_m=diag_dist,
            dist_p75_m=diag_dist,
            dist_p90_m=diag_dist,
            dist_max_m=diag_dist,
        )
        return dist_map, summary

    # Edge case: All building
    if np.all(bldg_mask):
        dist_map = np.zeros((ny, nx), dtype=np.float32)
        summary = DistanceSummary(
            dist_mean_m=0.0,
            dist_std_m=0.0,
            dist_median_m=0.0,
            dist_p25_m=0.0,
            dist_p75_m=0.0,
            dist_p90_m=0.0,
            dist_max_m=0.0,
        )
        return dist_map, summary

    # Mathematically exact periodic Euclidean distance:
    # Tile ~bldg_mask into a 3x3 periodic grid so that each cell in the central
    # block is surrounded by all 8 periodic neighbor blocks.
    # Because any periodic minimum distance on the torus [0, Ly) x [0, Lx) is
    # attained within +/- 1 period, 3x3 tiling guarantees exact Euclidean distance
    # to the nearest building obstacle across all periodic boundaries, for arbitrary
    # building arrangements, unequal dx/dy, and distances exceeding 80 m.
    tiled_open = np.tile(~bldg_mask, (3, 3))
    tiled_dist = ndimage.distance_transform_edt(tiled_open, sampling=(dy, dx))

    # Extract central unpadded domain [ny:2*ny, nx:2*nx]
    dist_map = tiled_dist[ny : 2 * ny, nx : 2 * nx].astype(np.float32)

    # Building cells have distance 0.0
    dist_map[bldg_mask] = 0.0

    # Summary statistics calculated strictly over valid open space cells
    fluid_distances = dist_map[~bldg_mask]
    if len(fluid_distances) > 0:
        d_mean = float(np.mean(fluid_distances))
        d_std = float(np.std(fluid_distances))
        d_med = float(np.median(fluid_distances))
        d_p25, d_p75, d_p90 = [float(p) for p in np.percentile(fluid_distances, [25, 75, 90])]
        d_max = float(np.max(fluid_distances))
    else:
        d_mean = d_std = d_med = d_p25 = d_p75 = d_p90 = d_max = 0.0

    summary = DistanceSummary(
        dist_mean_m=d_mean,
        dist_std_m=d_std,
        dist_median_m=d_med,
        dist_p25_m=d_p25,
        dist_p75_m=d_p75,
        dist_p90_m=d_p90,
        dist_max_m=d_max,
    )

    return dist_map, summary


def create_circular_disk_kernel(radius_m: float, dx: float, dy: float) -> np.ndarray:
    """Create normalized circular disk convolution kernel with specified physical radius."""
    rx_cells = int(math.ceil(radius_m / dx))
    ry_cells = int(math.ceil(radius_m / dy))

    y_grid, x_grid = np.ogrid[-ry_cells : ry_cells + 1, -rx_cells : rx_cells + 1]
    dist_sq = (x_grid * dx) ** 2 + (y_grid * dy) ** 2
    kernel = (dist_sq <= radius_m**2).astype(np.float64)

    total_weight = np.sum(kernel)
    if total_weight > 0:
        kernel /= total_weight
    return kernel


def _periodic_circular_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Fast, exact O(N log N) circular convolution on a periodic 2D torus via FFT."""
    ny, nx = image.shape
    ky, kx = kernel.shape

    # For small kernels (<= 9x9), ndimage.convolve with wrap mode is already optimal
    if ky <= 9 and kx <= 9 and ky <= ny and kx <= nx:
        return ndimage.convolve(image, kernel, mode="wrap").astype(np.float32)

    # For larger kernels or kernels wrapping around small domains, fold periodically
    cy, cx = ky // 2, kx // 2
    y_idx = (np.arange(ky) - cy) % ny
    x_idx = (np.arange(kx) - cx) % nx

    k_full = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(k_full, (y_idx[:, None], x_idx[None, :]), kernel)

    # Frequency domain circular convolution
    conv = np.real(np.fft.ifft2(np.fft.fft2(image.astype(np.float64)) * np.fft.fft2(k_full)))
    return conv.astype(np.float32)


def compute_multiscale_density_maps(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
    radii_m: Sequence[float] = (8.0, 16.0, 32.0, 64.0),
) -> Dict[str, np.ndarray]:
    """Compute periodic multi-scale local plan-area density maps lambda_p,R via FFT."""
    if dy is None:
        dy = dx

    bldg_mask = (H > 0.0).astype(np.float64)
    density_maps = {}

    for r in radii_m:
        kernel = create_circular_disk_kernel(r, dx, dy)
        dens = _periodic_circular_convolve(bldg_mask, kernel)
        key = f"lambda_p_{int(r)}m"
        density_maps[key] = np.clip(dens, 0.0, 1.0).astype(np.float32)

    return density_maps


def compute_multiscale_height_maps(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
    radii_m: Sequence[float] = (8.0, 16.0, 32.0, 64.0),
) -> Dict[str, np.ndarray]:
    """Compute periodic multi-scale local mean height maps H_local,R via FFT."""
    if dy is None:
        dy = dx

    h_float = H.astype(np.float64)
    bldg_mask = (H > 0.0).astype(np.float64)
    height_maps = {}

    for r in radii_m:
        kernel = create_circular_disk_kernel(r, dx, dy)
        # Local raster mean height (including open space zeros)
        h_raster = _periodic_circular_convolve(h_float, kernel)
        key_raster = f"h_mean_{int(r)}m"
        height_maps[key_raster] = np.maximum(0.0, h_raster).astype(np.float32)

        # Local building-only mean height
        local_bldg_weight = _periodic_circular_convolve(bldg_mask, kernel)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_bldg = np.where(local_bldg_weight > 1e-5, h_raster / local_bldg_weight, 0.0)
        key_bldg = f"h_bldg_{int(r)}m"
        height_maps[key_bldg] = np.maximum(0.0, h_bldg).astype(np.float32)

    return height_maps


def compute_canyon_geometry(
    dist_map: np.ndarray,
    H: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local street canyon width and canyon aspect ratio (H/W).

    Parameters
    ----------
    dist_map : np.ndarray
        Periodic Euclidean distance to nearest building [m].
    H : np.ndarray
        Building height raster [m].

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - canyon_width: Local open-space canyon width [m] (2 * dist_to_building).
        - aspect_ratio: Local height-to-width ratio H / W, where H is the
          **domain-mean building height** (a deliberate domain-averaged
          approximation appropriate for surrogate model inputs).
          Building cells are set to 0.0. Values clipped to [0, 10].
    """
    # Open canyon width estimated from boundary distance: W = 2 * dist_to_building
    canyon_width = np.maximum(1.0, 2.0 * dist_map).astype(np.float32)

    # Local aspect ratio H / W using domain-mean building height as numerator.
    # Note: We use the global mean building height (over H > 0 cells) as a
    # domain-representative canopy height. This is a deliberate approximation
    # for surrogate model features: per-pixel local height context is captured
    # separately through the multi-scale h_mean channels (Ch7-10 in the tensor).
    mean_bldg_h = float(np.mean(H[H > 0])) if np.any(H > 0) else 0.0
    aspect_ratio = np.where(H == 0.0, mean_bldg_h / canyon_width, 0.0)
    aspect_ratio = np.clip(aspect_ratio, 0.0, 10.0).astype(np.float32)

    return canyon_width, aspect_ratio
