"""
validation.py – Input validation, domain geometry verification, and synthetic benchmark generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class DomainGeometry:
    """Domain geometry and basic spatial metrics."""
    nx: int
    ny: int
    dx: float
    dy: float
    domain_x_m: float
    domain_y_m: float
    domain_area_m2: float
    aspect_ratio: float
    n_total_cells: int
    n_bldg_cells: int
    n_open_cells: int
    lambda_p: float
    is_valid: bool
    is_all_open: bool
    is_all_building: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for table export."""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "domain_x_m": self.domain_x_m,
            "domain_y_m": self.domain_y_m,
            "domain_area_m2": self.domain_area_m2,
            "aspect_ratio": self.aspect_ratio,
            "n_total_cells": self.n_total_cells,
            "n_bldg_cells": self.n_bldg_cells,
            "n_open_cells": self.n_open_cells,
            "lambda_p": self.lambda_p,
            "is_valid": self.is_valid,
            "is_all_open": self.is_all_open,
            "is_all_building": self.is_all_building,
        }


def validate_height_map(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
) -> DomainGeometry:
    """Validate height-map dimensions, finite values, and valid grid resolution.

    Parameters
    ----------
    H : np.ndarray
        2D raster array of building elevations in metres.
    dx : float
        Grid spacing in the horizontal x-direction in metres.
    dy : float, optional
        Grid spacing in the horizontal y-direction in metres. Defaults to dx.

    Returns
    -------
    DomainGeometry
        Validated domain geometry attributes and cell counts.

    Raises
    ------
    ValueError
        If H is not 2D, contains non-finite values, contains negative heights,
        or if grid spacings are non-positive.
    """
    if dy is None:
        dy = dx

    if not isinstance(H, np.ndarray):
        H = np.asarray(H, dtype=float)

    # Dimensionality check
    if H.ndim != 2:
        raise ValueError(f"Height map must be 2-dimensional (Ny, Nx). Got {H.ndim}D shape: {H.shape}")

    ny, nx = H.shape
    if ny < 2 or nx < 2:
        raise ValueError(f"Height map dimensions must be at least 2x2. Got shape: ({ny}, {nx})")

    # Grid spacing check
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Grid spacing must be strictly positive. Got dx={dx}, dy={dy}")

    # Finite values check
    if not np.all(np.isfinite(H)):
        n_nan = int(np.isnan(H).sum())
        n_inf = int(np.isinf(H).sum())
        raise ValueError(f"Height map contains non-finite values: {n_nan} NaNs, {n_inf} Infs")

    # Non-negativity check
    min_h = float(np.min(H))
    if min_h < 0.0:
        if min_h > -1e-6:
            # Clean floating precision noise near zero
            H = np.clip(H, 0.0, None)
        else:
            raise ValueError(f"Height map contains negative elevations: min(H) = {min_h:.4f} m")

    n_total = nx * ny
    bldg_mask = H > 0.0
    n_bldg = int(np.count_nonzero(bldg_mask))
    n_open = n_total - n_bldg
    lambda_p = float(n_bldg / n_total) if n_total > 0 else 0.0

    domain_x_m = float(nx * dx)
    domain_y_m = float(ny * dy)
    domain_area_m2 = float(domain_x_m * domain_y_m)
    aspect_ratio = float(domain_x_m / domain_y_m)

    is_all_open = (n_bldg == 0)
    is_all_building = (n_bldg == n_total)
    is_valid = bool(n_total > 0 and np.isfinite(domain_area_m2))

    return DomainGeometry(
        nx=nx,
        ny=ny,
        dx=float(dx),
        dy=float(dy),
        domain_x_m=domain_x_m,
        domain_y_m=domain_y_m,
        domain_area_m2=domain_area_m2,
        aspect_ratio=aspect_ratio,
        n_total_cells=n_total,
        n_bldg_cells=n_bldg,
        n_open_cells=n_open,
        lambda_p=lambda_p,
        is_valid=is_valid,
        is_all_open=is_all_open,
        is_all_building=is_all_building,
    )


def create_synthetic_benchmark() -> Tuple[np.ndarray, float]:
    """Create reproducible 40 x 40 height map for synthetic benchmarking.

    Geometry:
    ---------
    - Domain: 40 x 40 cells, dx = 1.0 m (40 m x 40 m, 1,600 total cells).
    - Building 1: 5 x 5 cells (height = 12.0 m) at rows [5:10], cols [5:10].
    - Building 2: 8 x 4 cells (height = 8.0 m) at rows [20:28], cols [20:24].
    - Total building cells: 25 + 32 = 57 cells.
    - Open-space cells: 1,600 - 57 = 1,543 cells.
    - Plan-area density: 57 / 1,600 = 0.035625.

    Returns
    -------
    Tuple[np.ndarray, float]
        Height map array of shape (40, 40) and grid resolution dx = 1.0 m.
    """
    ny, nx = 40, 40
    dx = 1.0
    H = np.zeros((ny, nx), dtype=np.float64)

    # Building 1: 5 x 5 cells, height 12 m
    H[5:10, 5:10] = 12.0

    # Building 2: 8 x 4 cells (8 rows, 4 cols), height 8 m
    H[20:28, 20:24] = 8.0

    # Analytical assertions
    geom = validate_height_map(H, dx=dx)
    assert geom.n_total_cells == 1600, f"Expected 1,600 total cells, got {geom.n_total_cells}"
    assert geom.n_bldg_cells == 57, f"Expected 57 building cells, got {geom.n_bldg_cells}"
    assert geom.n_open_cells == 1543, f"Expected 1,543 open cells, got {geom.n_open_cells}"
    assert abs(geom.lambda_p - 0.035625) < 1e-9, f"Expected lambda_p 0.035625, got {geom.lambda_p}"

    return H, dx
