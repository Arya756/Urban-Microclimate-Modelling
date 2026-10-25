"""
Stage S05: Urban Microclimate Feature Engineering.

Transforms preprocessed building height rasters and forcing parameters into
scientifically validated case-level and pixel-level features for ML and DL surrogates.
"""

from .aerodynamic import AerodynamicFeatures, compute_aerodynamic_parameters
from .config import DEFAULT_CONFIG, FeatureConfig
from .feature_assembly import (
    assemble_case_features,
    assemble_pixel_feature_tensor,
    load_normalized_target,
)
from .fetch_features import FetchSummary, compute_directional_fetch_maps
from .morphology import MorphologyFeatures, compute_morphology_features
from .orientation_features import OrientationFeatures, compute_directional_morphology
from .process_cases import build_feature_dictionary, process_single_case, run_pipeline
from .spatial_features import (
    DistanceSummary,
    compute_canyon_geometry,
    compute_multiscale_density_maps,
    compute_multiscale_height_maps,
    compute_periodic_distance_transform,
)
from .validation import DomainGeometry, create_synthetic_benchmark, validate_height_map

__all__ = [
    "FeatureConfig",
    "DEFAULT_CONFIG",
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
