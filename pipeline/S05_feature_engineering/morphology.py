"""
morphology.py – Building-height statistics, footprint morphometry, and canopy volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from scipy import ndimage, stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class MorphologyFeatures:
    """Urban morphology and height distribution features."""
    # Height Distribution (Calculated over building cells H > 0)
    mean_h: float
    std_h: float
    median_h: float
    min_h: float
    max_h: float
    p10_h: float
    p25_h: float
    p75_h: float
    p90_h: float
    iqr_h: float
    cv_h: float
    max_to_mean_h: float
    skew_h: float
    kurt_h: float
    raster_mean_h: float

    # Footprint & Canopy Morphometry
    lambda_p: float
    total_volume_m3: float
    volume_per_area_m: float
    n_connected_components: int
    components_per_hectare: float
    mean_footprint_area_m2: float
    median_footprint_area_m2: float
    max_footprint_area_m2: float
    gamma_h: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary for tabular export."""
        return {
            "mean_h": self.mean_h,
            "std_h": self.std_h,
            "median_h": self.median_h,
            "min_h": self.min_h,
            "max_h": self.max_h,
            "p10_h": self.p10_h,
            "p25_h": self.p25_h,
            "p75_h": self.p75_h,
            "p90_h": self.p90_h,
            "iqr_h": self.iqr_h,
            "cv_h": self.cv_h,
            "max_to_mean_h": self.max_to_mean_h,
            "skew_h": self.skew_h,
            "kurt_h": self.kurt_h,
            "raster_mean_h": self.raster_mean_h,
            "lambda_p": self.lambda_p,
            "total_volume_m3": self.total_volume_m3,
            "volume_per_area_m": self.volume_per_area_m,
            "n_connected_components": self.n_connected_components,
            "components_per_hectare": self.components_per_hectare,
            "mean_footprint_area_m2": self.mean_footprint_area_m2,
            "median_footprint_area_m2": self.median_footprint_area_m2,
            "max_footprint_area_m2": self.max_footprint_area_m2,
            "gamma_h": self.gamma_h,
        }


def periodic_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components on a doubly periodic grid.

    ndimage.label treats the domain edges as walls, so a building crossing an edge
    would count as two (or four) buildings; labels touching across opposite edges are merged.
    """
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return labels, 0
    pairs = []
    for a, b in ((labels[0, :], labels[-1, :]), (labels[:, 0], labels[:, -1])):
        for shift in (-1, 0, 1):  # diagonal neighbours across the edge, including the corners
            b_shifted = np.roll(b, shift)
            touch = (a > 0) & (b_shifted > 0)
            pairs.append(np.stack([a[touch], b_shifted[touch]], axis=1))
    edges = np.concatenate(pairs) - 1
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    n_merged, merged = connected_components(graph, directed=False)
    out = np.zeros_like(labels)
    out[labels > 0] = merged[labels[labels > 0] - 1] + 1
    return out, int(n_merged)


def compute_morphology_features(
    H: np.ndarray,
    dx: float,
    dy: float | None = None,
) -> MorphologyFeatures:
    """Compute building-height statistics, footprint morphometry, and canopy volume.

    Parameters
    ----------
    H : np.ndarray
        2D array of building heights in metres.
    dx : float
        Grid resolution in x-direction (metres).
    dy : float, optional
        Grid resolution in y-direction (metres). Defaults to dx.

    Returns
    -------
    MorphologyFeatures
        Calculated morphological metrics.
    """
    if dy is None:
        dy = dx

    ny, nx = H.shape
    cell_area = float(dx * dy)
    domain_area_m2 = float(nx * dx * ny * dy)
    n_total_cells = nx * ny

    bldg_mask = H > 0.0
    n_bldg_cells = int(np.count_nonzero(bldg_mask))
    lambda_p = float(n_bldg_cells / n_total_cells) if n_total_cells > 0 else 0.0
    raster_mean_h = float(np.mean(H)) if n_total_cells > 0 else 0.0

    # Height statistics (calculated strictly over valid building cells)
    if n_bldg_cells > 0:
        h_vals = H[bldg_mask].astype(np.float64)
        mean_h = float(np.mean(h_vals))
        std_h = float(np.std(h_vals))
        median_h = float(np.median(h_vals))
        min_h = float(np.min(h_vals))
        max_h = float(np.max(h_vals))
        p10_h, p25_h, p75_h, p90_h = [float(p) for p in np.percentile(h_vals, [10, 25, 75, 90])]
        iqr_h = float(p75_h - p25_h)
        cv_h = float(std_h / mean_h) if mean_h > 0 else 0.0
        max_to_mean_h = float(max_h / mean_h) if mean_h > 0 else 0.0

        if len(h_vals) >= 3 and std_h > 1e-6:
            skew_h = float(stats.skew(h_vals))
        else:
            skew_h = 0.0

        if len(h_vals) >= 4 and std_h > 1e-6:
            kurt_h = float(stats.kurtosis(h_vals))
        else:
            kurt_h = 0.0

        # Building volume
        total_volume_m3 = float(np.sum(h_vals) * cell_area)
        volume_per_area_m = float(total_volume_m3 / domain_area_m2) if domain_area_m2 > 0 else 0.0
        gamma_h = float((std_h / mean_h) * 100.0) if mean_h > 0 else 0.0

    else:
        # All-open domain edge case
        mean_h = std_h = median_h = min_h = max_h = 0.0
        p10_h = p25_h = p75_h = p90_h = iqr_h = cv_h = max_to_mean_h = 0.0
        skew_h = kurt_h = total_volume_m3 = volume_per_area_m = gamma_h = 0.0

    # Connected Components (Raster Building Patches)
    # Using 8-connectivity to identify contiguous building footprints
    if n_bldg_cells > 0:
        labeled_array, num_features = periodic_labels(bldg_mask)
        n_connected = int(num_features)

        # Component area distribution
        component_sizes = ndimage.sum(bldg_mask, labeled_array, range(1, num_features + 1))
        if isinstance(component_sizes, np.ndarray):
            comp_areas = component_sizes.astype(np.float64) * cell_area
            mean_footprint = float(np.mean(comp_areas))
            median_footprint = float(np.median(comp_areas))
            max_footprint = float(np.max(comp_areas))
        else:
            mean_footprint = median_footprint = max_footprint = float(component_sizes) * cell_area

        domain_hectares = domain_area_m2 / 10000.0
        components_per_hectare = float(n_connected / domain_hectares) if domain_hectares > 0 else 0.0
    else:
        n_connected = 0
        components_per_hectare = 0.0
        mean_footprint = median_footprint = max_footprint = 0.0

    return MorphologyFeatures(
        mean_h=mean_h,
        std_h=std_h,
        median_h=median_h,
        min_h=min_h,
        max_h=max_h,
        p10_h=p10_h,
        p25_h=p25_h,
        p75_h=p75_h,
        p90_h=p90_h,
        iqr_h=iqr_h,
        cv_h=cv_h,
        max_to_mean_h=max_to_mean_h,
        skew_h=skew_h,
        kurt_h=kurt_h,
        raster_mean_h=raster_mean_h,
        lambda_p=lambda_p,
        total_volume_m3=total_volume_m3,
        volume_per_area_m=volume_per_area_m,
        n_connected_components=n_connected,
        components_per_hectare=components_per_hectare,
        mean_footprint_area_m2=mean_footprint,
        median_footprint_area_m2=median_footprint,
        max_footprint_area_m2=max_footprint,
        gamma_h=gamma_h,
    )
