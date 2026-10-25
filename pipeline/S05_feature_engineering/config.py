"""
config.py – Configuration parameters, constants, and paths for Stage S05 (Feature Engineering).
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Sequence

# Repository root detection
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration settings for urban microclimate feature engineering."""

    # Paths
    repo_root: pathlib.Path = REPO_ROOT
    data_dir: pathlib.Path = REPO_ROOT / "data"
    raw_data_dir: pathlib.Path = REPO_ROOT / "data" / "raw"
    processed_dir: pathlib.Path = REPO_ROOT / "data" / "processed"
    eda_dir: pathlib.Path = REPO_ROOT / "pipeline" / "S04_eda"
    stage_dir: pathlib.Path = REPO_ROOT / "pipeline" / "S05_feature_engineering"
    output_dir: pathlib.Path = REPO_ROOT / "pipeline" / "S05_feature_engineering" / "outputs"

    catalog_path: pathlib.Path = REPO_ROOT / "data" / "catalog.csv"
    metadata_path: pathlib.Path = REPO_ROOT / "data" / "metadata.csv"
    splits_path: pathlib.Path = REPO_ROOT / "data" / "splits.json"
    master_case_table_path: pathlib.Path = (
        REPO_ROOT / "pipeline" / "S04_eda" / "S04g_combined_report" / "master_case_table.csv"
    )
    model_ready_cases_path: pathlib.Path = (
        REPO_ROOT / "pipeline" / "S04_eda" / "S04g_combined_report" / "model_ready_cases.csv"
    )

    # Output file paths
    case_features_csv: pathlib.Path = (
        REPO_ROOT / "pipeline" / "S05_feature_engineering" / "outputs" / "case_features.csv"
    )
    pixel_features_dir: pathlib.Path = (
        REPO_ROOT / "pipeline" / "S05_feature_engineering" / "outputs" / "pixel_features"
    )
    feature_dictionary_json: pathlib.Path = (
        REPO_ROOT / "pipeline" / "S05_feature_engineering" / "outputs" / "feature_dictionary.json"
    )
    qc_report_md: pathlib.Path = (
        REPO_ROOT / "pipeline" / "S05_feature_engineering" / "outputs" / "S05_gate_report.md"
    )

    # Physical Constants & Reference Parameters
    von_karman_constant: float = 0.40
    default_ground_roughness_m: float = 0.01  # standard aerodynamic ground roughness (m)
    pedestrian_height_m: float = 1.75          # standard pedestrian target height (m)
    reference_friction_velocity: float = 0.2137 # nominal realistic u_tau (m/s)

    # Macdonald (1998) Empirical Parameters
    macdonald_alpha: float = 4.43
    macdonald_cd: float = 1.20
    macdonald_beta: float = 1.00

    # Multi-scale Spatial Radii (in metres)
    density_radii_m: Sequence[float] = field(default_factory=lambda: (8.0, 16.0, 32.0, 64.0))

    # Fetch Parameterization Limits (in metres)
    upwind_fetch_cap_m: float = 150.0
    downwind_fetch_cap_m: float = 60.0
    lateral_fetch_cap_m: float = 60.0
    upwind_height_bands_m: Sequence[tuple[float, float]] = field(
        default_factory=lambda: ((1.0, 10.0), (11.0, 30.0), (31.0, 60.0))
    )

    # Channel Ordering for Pixel-Level Spatial Tensors (22 channels)
    channel_names: Sequence[str] = field(
        default_factory=lambda: (
            "building_height",      # [m] Raw elevation raster
            "fluid_mask",          # [0/1] 1 for fluid/open space, 0 for building
            "dist_to_building",    # [m] Periodic Euclidean distance to nearest building
            "lambda_p_8m",         # [-] Local plan area density (radius 8m)
            "lambda_p_16m",        # [-] Local plan area density (radius 16m)
            "lambda_p_32m",        # [-] Local plan area density (radius 32m)
            "lambda_p_64m",        # [-] Local plan area density (radius 64m)
            "h_mean_8m",           # [m] Local mean raster height (radius 8m)
            "h_mean_16m",          # [m] Local mean raster height (radius 16m)
            "h_mean_32m",          # [m] Local mean raster height (radius 32m)
            "h_mean_64m",          # [m] Local mean raster height (radius 64m)
            "upwind_fetch",        # [m] Upwind distance to building (capped at 150m)
            "downwind_fetch",      # [m] Downwind distance to building (capped at 60m)
            "lateral_left_fetch",  # [m] Left lateral distance (capped at 60m)
            "lateral_right_fetch", # [m] Right lateral distance (capped at 60m)
            "upwind_h_1_10m",      # [m] Upwind building height within 1-10m
            "upwind_h_11_30m",     # [m] Upwind building height within 11-30m
            "upwind_h_31_60m",     # [m] Upwind building height within 31-60m
            "canyon_width",        # [m] Local open-space canyon width
            "aspect_ratio_hw",     # [-] Local height-to-width ratio
            "wind_cos",            # [-] cos(forcing_direction_rad)
            "wind_sin",            # [-] sin(forcing_direction_rad)
        )
    )


DEFAULT_CONFIG = FeatureConfig()
