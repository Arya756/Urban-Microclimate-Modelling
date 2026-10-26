"""
features.py – High-level features API exposed through urbanwind shared library.

Re-exports core feature extraction functions from pipeline.S05_feature_engineering.
"""

from __future__ import annotations

import sys
import pathlib

# Ensure repo root is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.S05_feature_engineering import (  # noqa: E402  (needs the repository root on sys.path)
    DEFAULT_CONFIG,
    AerodynamicFeatures,
    DistanceSummary,
    DomainGeometry,
    FeatureConfig,
    FetchSummary,
    MorphologyFeatures,
    OrientationFeatures,
    assemble_case_features,
    assemble_pixel_feature_tensor,
    build_feature_dictionary,
    compute_aerodynamic_parameters,
    compute_canyon_geometry,
    compute_directional_fetch_maps,
    compute_directional_morphology,
    compute_morphology_features,
    compute_multiscale_density_maps,
    compute_multiscale_height_maps,
    compute_periodic_distance_transform,
    create_synthetic_benchmark,
    load_normalized_target,
    process_single_case,
    run_pipeline,
    validate_height_map,
)

__all__ = [
    "DEFAULT_CONFIG",
    "FeatureConfig",
    "DomainGeometry",
    "validate_height_map",
    "create_synthetic_benchmark",
    "MorphologyFeatures",
    "compute_morphology_features",
    "OrientationFeatures",
    "compute_directional_morphology",
    "AerodynamicFeatures",
    "compute_aerodynamic_parameters",
    "DistanceSummary",
    "compute_periodic_distance_transform",
    "compute_multiscale_density_maps",
    "compute_multiscale_height_maps",
    "compute_canyon_geometry",
    "FetchSummary",
    "compute_directional_fetch_maps",
    "assemble_case_features",
    "assemble_pixel_feature_tensor",
    "load_normalized_target",
    "process_single_case",
    "run_pipeline",
    "build_feature_dictionary",
]
