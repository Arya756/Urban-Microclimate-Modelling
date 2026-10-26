"""
feature_assembly.py – Pixel-level tensor assembly, case-level feature compilation, and separate target loading.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple
import h5py
import numpy as np

from .aerodynamic import compute_aerodynamic_parameters
from .config import DEFAULT_CONFIG, FeatureConfig
from .fetch_features import compute_directional_fetch_maps
from .morphology import compute_morphology_features
from .orientation_features import compute_directional_morphology
from .spatial_features import (
    compute_canyon_geometry,
    compute_multiscale_density_maps,
    compute_multiscale_height_maps,
    compute_periodic_distance_transform,
)
from .validation import validate_height_map


def assemble_case_features(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
    case_id: str = "",
    forcing_dir_deg: float = 0.0,
    metadata_row: Optional[Dict[str, Any]] = None,
    cfg: FeatureConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Compile comprehensive case-level feature dictionary.

    Zero Target Leakage Guarantee:
    ------------------------------
    Only building geometry H, grid resolution (dx, dy), and prevailing forcing
    direction are used. No wind-speed targets or profile values are included.

    Parameters
    ----------
    H : np.ndarray
        2D building height array.
    dx : float
        Grid spacing in x-direction (metres).
    dy : float, optional
        Grid spacing in y-direction (metres).
    case_id : str, optional
        Unique case identifier.
    forcing_dir_deg : float, optional
        Physical wind forcing direction in degrees.
    metadata_row : Dict[str, Any], optional
        Case descriptors (split_group, subset, city, etc.).
    cfg : FeatureConfig, optional
        Configuration settings.

    Returns
    -------
    Dict[str, Any]
        Flattened record containing all engineered case-level features.
    """
    if dy is None:
        dy = dx

    # 1. Validation & Domain Geometry
    geom = validate_height_map(H, dx=dx, dy=dy)
    geom_dict = geom.to_dict()

    # 2. Building Height Statistics & Morphology
    morph = compute_morphology_features(H, dx=dx, dy=dy)
    morph_dict = morph.to_dict()

    # 3. Orientation & Directional Facades
    orient = compute_directional_morphology(H, dx=dx, dy=dy, forcing_dir_deg=forcing_dir_deg)
    orient_dict = orient.to_dict()

    # 4. Aerodynamic Parameterizations (Macdonald & Kanda)
    aero = compute_aerodynamic_parameters(
        lambda_p=morph.lambda_p,
        lambda_f=orient.lambda_f_dir,
        mean_h=morph.mean_h,
        std_h=morph.std_h,
        max_h=morph.max_h,
        cfg=cfg,
    )
    aero_dict = aero.to_dict()

    # 5. Distance-to-Building Summary
    _, dist_summary = compute_periodic_distance_transform(H, dx=dx, dy=dy)
    dist_dict = dist_summary.to_dict()

    # 6. Directional Fetch Summary
    _, fetch_summary = compute_directional_fetch_maps(
        H,
        dx=dx,
        dy=dy,
        forcing_dir_deg=forcing_dir_deg,
        upwind_cap_m=cfg.upwind_fetch_cap_m,
        downwind_cap_m=cfg.downwind_fetch_cap_m,
        lateral_cap_m=cfg.lateral_fetch_cap_m,
    )
    fetch_dict = fetch_summary.to_dict()

    record: Dict[str, Any] = {
        "case": case_id,
        **geom_dict,
        **morph_dict,
        **orient_dict,
        **aero_dict,
        **dist_dict,
        **fetch_dict,
    }

    # Integrate permitted environmental and group metadata if provided
    if metadata_row:
        permitted_keys = [
            "subset",
            "split_group",
            "country",
            "city",
            "density_class",
            "recommended_split",
            "fold",
            "u_tau_metadata",
            "roughness_length",
        ]
        for k in permitted_keys:
            if k in metadata_row:
                record[k] = metadata_row[k]

    return record


def assemble_pixel_feature_tensor(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
    forcing_dir_deg: float = 0.0,
    cfg: FeatureConfig = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble aligned multi-channel spatial feature tensor (C, Ny, Nx).

    Channel Structure (22 channels):
    --------------------------------
    0. building_height      [m] Raw elevation raster
    1. fluid_mask          [0/1] 1.0 for open fluid, 0.0 for building
    2. dist_to_building    [m] Periodic Euclidean distance to nearest building
    3. lambda_p_8m         [-] Local plan-area density at 8m radius
    4. lambda_p_16m        [-] Local plan-area density at 16m radius
    5. lambda_p_32m        [-] Local plan-area density at 32m radius
    6. lambda_p_64m        [-] Local plan-area density at 64m radius
    7. h_mean_8m           [m] Local mean raster height at 8m radius
    8. h_mean_16m          [m] Local mean raster height at 16m radius
    9. h_mean_32m          [m] Local mean raster height at 32m radius
    10. h_mean_64m         [m] Local mean raster height at 64m radius
    11. upwind_fetch       [m] Unobstructed upwind distance (capped at 150m)
    12. downwind_fetch     [m] Unobstructed downwind distance (capped at 60m)
    13. lateral_left_fetch [m] Left lateral distance (capped at 60m)
    14. lateral_right_fetch[m] Right lateral distance (capped at 60m)
    15. upwind_h_1_10m     [m] Upwind building height within 1-10m
    16. upwind_h_11_30m    [m] Upwind building height within 11-30m
    17. upwind_h_31_60m    [m] Upwind building height within 31-60m
    18. canyon_width       [m] Local canyon width (2 * distance)
    19. aspect_ratio_hw    [-] Local height-to-width ratio
    20. wind_cos           [-] cos(forcing_direction_rad)
    21. wind_sin           [-] sin(forcing_direction_rad)

    Parameters
    ----------
    H : np.ndarray
        2D building height array (Ny, Nx).
    dx : float
        Grid spacing in x-direction (metres).
    dy : float, optional
        Grid spacing in y-direction (metres).
    forcing_dir_deg : float
        Prevailing flow direction in degrees.
    cfg : FeatureConfig, optional
        Configuration settings.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        - tensor: (C, Ny, Nx) float32 array.
        - channel_names: List of channel names matching axis 0.
    """
    if dy is None:
        dy = dx

    ny, nx = H.shape
    bldg_mask = (H > 0.0)
    fluid_mask = (~bldg_mask).astype(np.float32)

    # Spatial features
    dist_map, _ = compute_periodic_distance_transform(H, dx=dx, dy=dy)
    density_maps = compute_multiscale_density_maps(H, dx=dx, dy=dy, radii_m=cfg.density_radii_m)
    height_maps = compute_multiscale_height_maps(H, dx=dx, dy=dy, radii_m=cfg.density_radii_m)
    fetch_maps, _ = compute_directional_fetch_maps(
        H,
        dx=dx,
        dy=dy,
        forcing_dir_deg=forcing_dir_deg,
        upwind_cap_m=cfg.upwind_fetch_cap_m,
        downwind_cap_m=cfg.downwind_fetch_cap_m,
        lateral_cap_m=cfg.lateral_fetch_cap_m,
    )
    canyon_width, aspect_ratio = compute_canyon_geometry(dist_map, H)

    # Direction cosine / sine fields
    theta_rad = math.radians(forcing_dir_deg % 360.0)
    cos_field = np.full((ny, nx), math.cos(theta_rad), dtype=np.float32)
    sin_field = np.full((ny, nx), math.sin(theta_rad), dtype=np.float32)

    channels: List[np.ndarray] = [
        H.astype(np.float32),
        fluid_mask,
        dist_map.astype(np.float32),
        density_maps["lambda_p_8m"],
        density_maps["lambda_p_16m"],
        density_maps["lambda_p_32m"],
        density_maps["lambda_p_64m"],
        height_maps["h_mean_8m"],
        height_maps["h_mean_16m"],
        height_maps["h_mean_32m"],
        height_maps["h_mean_64m"],
        fetch_maps["upwind_fetch"],
        fetch_maps["downwind_fetch"],
        fetch_maps["lateral_left_fetch"],
        fetch_maps["lateral_right_fetch"],
        fetch_maps["upwind_h_1_10m"],
        fetch_maps["upwind_h_11_30m"],
        fetch_maps["upwind_h_31_60m"],
        canyon_width,
        aspect_ratio,
        cos_field,
        sin_field,
    ]

    tensor = np.stack(channels, axis=0)
    channel_names = list(cfg.channel_names)

    assert tensor.shape == (len(channel_names), ny, nx), (
        f"Tensor shape {tensor.shape} does not match expected ({len(channel_names)}, {ny}, {nx})"
    )

    return tensor, channel_names


def load_normalized_target(
    ped_path: str | pathlib.Path,
    u_tau: float,
    topo_H: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pedestrian-level target velocity and turbulence fields normalized by u_tau.

    Y = Uped(z = 1.75m) / u_tau
    TKE_norm = TKEped(z = 1.75m) / (u_tau^2)

    Parameters
    ----------
    ped_path : str or pathlib.Path
        Path to <case>_ped.nc file.
    u_tau : float
        Friction velocity for this case.
    topo_H : np.ndarray, optional
        2D building height array (Ny, Nx), already corrected for row orientation
        via np.flipud.  When provided, cells where topo_H > 0 are
        unconditionally treated as building cells and excluded from valid_mask,
        regardless of whether _ped.nc stores NaN, zero, or a spurious finite value
        at that location.  Legitimate zero-valued wind targets on open ground
        (topo_H == 0) are correctly preserved.
        163 cases (160 realistic, 3 idealized) contain CFD grid-staggering
        artefacts: sub-grid interpolation leaks finite wind values into
        building-perimeter cells even though those cells are inside buildings
        according to the topography. Without topo_H the NaN-only mask incorrectly
        includes 1,141,511 such cells as valid targets across the full dataset.
        When topo_H is None the function falls back to the original NaN-only mask
        for backward compatibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - Y: Normalized pedestrian wind speed (Ny, Nx).  Zero at invalid cells.
        - tke_norm: Normalized turbulence kinetic energy (Ny, Nx).  Zero at invalid cells.
        - valid_mask: Boolean mask — True only for open-ground cells with finite,
          non-NaN wind values.
    """
    if u_tau <= 0.0:
        raise ValueError(f"Friction velocity u_tau must be strictly positive. Got: {u_tau}")

    with h5py.File(str(ped_path), "r") as f:
        u_arr = np.array(f["Uped"], dtype=np.float32)
        tke_arr = np.array(f["TKEped"], dtype=np.float32) if "TKEped" in f else np.zeros_like(u_arr)

    # Build valid mask: start from finite, non-NaN cells in the wind map.
    valid_mask = np.isfinite(u_arr)

    # Additionally exclude building cells identified by the
    # topography matrix, which is authoritative for building/open classification.
    # This removes CFD perimeter artefacts that ped.nc does not mark as NaN.
    if topo_H is not None:
        if topo_H.shape != u_arr.shape:
            raise ValueError(
                f"topo_H shape {topo_H.shape} does not match Uped shape {u_arr.shape}. "
                "Ensure topo_H has been corrected with np.flipud before calling."
            )
        valid_mask &= (topo_H == 0.0)  # exclude building cells (H > 0)

    # Normalize by friction velocity
    Y = np.zeros_like(u_arr, dtype=np.float32)
    Y[valid_mask] = u_arr[valid_mask] / float(u_tau)

    tke_norm = np.zeros_like(tke_arr, dtype=np.float32)
    valid_tke = np.isfinite(tke_arr) & valid_mask
    tke_norm[valid_tke] = tke_arr[valid_tke] / float(u_tau**2)

    return Y, tke_norm, valid_mask
