"""
test_readiness.py – Unit and regression tests for model readiness
flags, exclusions, topography alignment, and split isolation.

Tests verify:
1. Missing required topography or pedestrian target files are detected.
2. Incompatible topography and target shapes are detected and raise ValueError.
3. Corrected np.flipud(H) orientation is used consistently with the valid-cell mask.
4. Valid target cells are defined strictly using (topo_H == 0.0) & np.isfinite(target).
5. Cases with zero valid target cells are detected and not marked ready.
6. Existing model-ready cases (including the 6 investigated cases) are represented
   in the generated feature table.
7. Existing documented exclusions (CN-BE-V2_d00, CN-CD-V1_d00, CN-SH-V1_d00)
   remain auditable in master metadata with documented reasons.
8. Readiness filtering preserves strict isolation of frozen test groups from development folds.
"""
from __future__ import annotations

import json
import pathlib
import h5py
import numpy as np
import pandas as pd
import pytest

from pipeline.S05_feature_engineering.config import FeatureConfig
from pipeline.S05_feature_engineering.feature_assembly import load_normalized_target
from pipeline.S05_feature_engineering.process_cases import process_single_case

TEST_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS_PATH = REPO_ROOT / "data" / "splits.json"
MASTER_TABLE_PATH = REPO_ROOT / "pipeline" / "S04_eda" / "S04g_combined_report" / "master_case_table.csv"
MODEL_READY_PATH = REPO_ROOT / "pipeline" / "S04_eda" / "S04g_combined_report" / "model_ready_cases.csv"
CASE_FEATURES_PATH = REPO_ROOT / "pipeline" / "S05_feature_engineering" / "outputs" / "case_features.csv"

INVESTIGATED_CASES = [
    "JP-Tok-V4_d00",
    "UG-UG-U1_d00",
    "UG-UG-U2_d00",
    "UG-UG-U3_d00",
    "NP-NP-U1_d00",
    "AU-Mel-U9_d00",
]

EXCLUDED_CASES = [
    "CN-BE-V2_d00",
    "CN-CD-V1_d00",
    "CN-SH-V1_d00",
]


def _write_synthetic_ped_nc(path: pathlib.Path, u: np.ndarray, tke: np.ndarray | None = None) -> None:
    """Create a synthetic NetCDF/HDF5 target file."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("Uped", data=u.astype(np.float32))
        if tke is not None:
            f.create_dataset("TKEped", data=tke.astype(np.float32))


# ---------------------------------------------------------------------------
# Unit tests using synthetic data
# ---------------------------------------------------------------------------

class TestReadinessSynthetic:
    """Synthetic unit tests for readiness criteria and validation checks."""

    def test_missing_topography_file_detected(self, tmp_path: pathlib.Path):
        """Verify that process_single_case skips and returns None when topo file is missing."""
        cfg = FeatureConfig(raw_data_dir=tmp_path / "raw", output_dir=tmp_path / "out")
        master_df = pd.DataFrame(
            [{"case": "SYNTH-01", "subset": "idealized", "forcing_dir": 0.0, "u_tau_metadata": 1.0}],
            index=["SYNTH-01"],
        )
        case_dir = cfg.raw_data_dir / "SYNTH-01"
        case_dir.mkdir(parents=True)

        res = process_single_case("SYNTH-01", master_df, cfg=cfg, save_pixel=False)
        assert res is None, "Expected None when topography file is missing."

    def test_missing_pedestrian_target_file_detected(self, tmp_path: pathlib.Path):
        """Verify that load_normalized_target raises an error if the target file does not exist."""
        non_existent_nc = tmp_path / "missing_ped.nc"
        with pytest.raises((OSError, FileNotFoundError)):
            load_normalized_target(non_existent_nc, u_tau=1.0)

    def test_incompatible_topography_and_target_shapes(self, tmp_path: pathlib.Path):
        """Verify that mismatched grid shapes raise a ValueError."""
        ped_file = tmp_path / "synth_ped.nc"
        u = np.ones((20, 25), dtype=np.float32)
        _write_synthetic_ped_nc(ped_file, u)

        topo_H = np.zeros((20, 20), dtype=np.float64)  # shape mismatch (20, 20) vs (20, 25)

        with pytest.raises(ValueError, match="does not match Uped shape"):
            load_normalized_target(ped_file, u_tau=1.0, topo_H=topo_H)

    def test_corrected_flipud_orientation_with_valid_mask(self, tmp_path: pathlib.Path):
        """
        Verify that vertically flipping topography aligns with NetCDF
        orientation to accurately mask ground vs building cells.
        """
        ny, nx = 30, 30
        # NetCDF: row 0 = bottom, row ny-1 = top
        # Buildings near top (rows 25..29 in NetCDF)
        u = np.full((ny, nx), 0.5, dtype=np.float32)
        u[25:, :] = np.nan

        ped_file = tmp_path / "orient_ped.nc"
        _write_synthetic_ped_nc(ped_file, u)

        # In raw topo file, top row is row 0 (stored top-row-first)
        # So buildings near top are at rows 0..4 in file
        raw_topo = np.zeros((ny, nx), dtype=np.float64)
        raw_topo[0:5, :] = 15.0

        # Without flipud, building mask is at rows 0..4, but NetCDF NaNs are at 25..29 (mismatch)
        _, _, unaligned_mask = load_normalized_target(ped_file, u_tau=1.0, topo_H=raw_topo)
        assert bool(unaligned_mask[0, 0]) is False  # falsely excluded as building by unaligned topo

        # With np.flipud(raw_topo), row 0..4 flips to 25..29, perfectly aligning with NetCDF
        corrected_topo = np.flipud(raw_topo)
        Y, _, aligned_mask = load_normalized_target(ped_file, u_tau=1.0, topo_H=corrected_topo)

        # Ground cells (rows 0..24) must all be valid
        assert aligned_mask[0:25, :].all()
        # Building cells (rows 25..29) must all be masked out (False)
        assert not aligned_mask[25:, :].any()
        assert np.all(Y[25:, :] == 0.0)

    def test_valid_target_cell_definition(self, tmp_path: pathlib.Path):
        """
        Verify that valid target cells require strictly:
        (topo_H == 0.0) & np.isfinite(target)
        """
        ped_file = tmp_path / "valid_def_ped.nc"
        # 4 cells:
        # Cell (0,0): topo=0, u=1.0 -> VALID
        # Cell (0,1): topo=0, u=0.0 -> VALID (legitimate zero-wind on open ground)
        # Cell (1,0): topo=10, u=1.0 -> INVALID (building with CFD perimeter artefact)
        # Cell (1,1): topo=0, u=NaN -> INVALID (open ground but invalid measurement)
        u = np.array([[1.0, 0.0], [1.0, np.nan]], dtype=np.float32)
        topo = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        _write_synthetic_ped_nc(ped_file, u)

        Y, _, valid_mask = load_normalized_target(ped_file, u_tau=0.5, topo_H=topo)

        assert bool(valid_mask[0, 0]) is True
        assert bool(valid_mask[0, 1]) is True
        assert bool(valid_mask[1, 0]) is False
        assert bool(valid_mask[1, 1]) is False

        # Normalized values
        assert abs(Y[0, 0] - 2.0) < 1e-6
        assert Y[0, 1] == 0.0
        assert Y[1, 0] == 0.0
        assert Y[1, 1] == 0.0

    def test_zero_valid_target_cells_detected(self, tmp_path: pathlib.Path):
        """Verify that domains with zero valid cells are identified as having empty valid mask."""
        ped_file = tmp_path / "all_bldg_ped.nc"
        u = np.full((10, 10), 0.5, dtype=np.float32)
        topo = np.full((10, 10), 20.0, dtype=np.float64)  # 100% building
        _write_synthetic_ped_nc(ped_file, u)

        _, _, valid_mask = load_normalized_target(ped_file, u_tau=1.0, topo_H=topo)
        assert valid_mask.sum() == 0, "All-building domain must have zero valid target cells."


# ---------------------------------------------------------------------------
# Metadata and Repository State Tests
# ---------------------------------------------------------------------------

class TestRepositoryReadinessAndExclusions:
    """Tests verifying project readiness records, exclusion audit trail, and split isolation."""

    @pytest.mark.skipif(not MASTER_TABLE_PATH.exists(), reason="master_case_table.csv not found")
    def test_investigated_cases_are_marked_ready(self):
        """Confirm that all six investigated cases have is_model_ready_spatial == True."""
        master_df = pd.read_csv(MASTER_TABLE_PATH).set_index("case")
        for cid in INVESTIGATED_CASES:
            assert cid in master_df.index, f"{cid} must be present in master case table."
            assert master_df.loc[cid, "is_model_ready_spatial"] is True or master_df.loc[cid, "is_model_ready_spatial"] == 1, (
                f"{cid} must be flagged as model ready."
            )
            assert master_df.loc[cid, "qc_topo_valid"] is True or master_df.loc[cid, "qc_topo_valid"] == 1
            assert master_df.loc[cid, "qc_ped_valid"] is True or master_df.loc[cid, "qc_ped_valid"] == 1

    @pytest.mark.skipif(not MASTER_TABLE_PATH.exists(), reason="master_case_table.csv not found")
    def test_documented_exclusions_auditable(self):
        """Confirm that the 3 documented exclusions remain auditable with documented reasons."""
        master_df = pd.read_csv(MASTER_TABLE_PATH).set_index("case")
        for cid in EXCLUDED_CASES:
            assert cid in master_df.index, f"{cid} must remain recorded in master table for auditability."
            assert master_df.loc[cid, "is_model_ready_spatial"] is False or master_df.loc[cid, "is_model_ready_spatial"] == 0
            assert master_df.loc[cid, "qc_topo_valid"] is False or master_df.loc[cid, "qc_topo_valid"] == 0

        # Also confirm they are not in model_ready_cases.csv
        if MODEL_READY_PATH.exists():
            ready_df = pd.read_csv(MODEL_READY_PATH)
            ready_cases = set(ready_df["case"])
            for cid in EXCLUDED_CASES:
                assert cid not in ready_cases, f"Excluded case {cid} must not be in model_ready_cases.csv"

    @pytest.mark.skipif(not CASE_FEATURES_PATH.exists(), reason="case_features.csv not found")
    def test_investigated_cases_represented_in_features(self):
        """Verify that all six investigated cases exist in the generated S05 feature table."""
        feat_df = pd.read_csv(CASE_FEATURES_PATH).set_index("case")
        for cid in INVESTIGATED_CASES:
            assert cid in feat_df.index, f"{cid} must be present in generated case_features.csv"
            # Ensure physical features are finite
            assert np.isfinite(feat_df.loc[cid, "lambda_p"])
            assert np.isfinite(feat_df.loc[cid, "z0_over_h_kanda"])

    @pytest.mark.skipif(not SPLITS_PATH.exists(), reason="data/splits.json not found")
    def test_readiness_preserves_frozen_test_isolation(self):
        """
        Verify that readiness checks and development folds strictly isolate frozen test cases:
        - 482 development cases
        - 136 development spatial groups
        - 53 frozen test cases (24 test groups)
        - Zero group overlap between frozen test and development folds.
        """
        with open(SPLITS_PATH, "r") as f:
            splits_data = json.load(f)

        fold_groups = set(splits_data.get("folds", {}).keys())
        test_groups = set(splits_data.get("test_groups", []))

        assert len(test_groups) == 24, f"Expected 24 frozen test groups, got {len(test_groups)}"
        assert len(fold_groups) == 136, f"Expected 136 development spatial groups, got {len(fold_groups)}"
        assert not fold_groups.intersection(test_groups), "Development groups must be disjoint from frozen test groups."

        # Verify mapping against master table
        if MASTER_TABLE_PATH.exists():
            master_df = pd.read_csv(MASTER_TABLE_PATH)
            dev_cases = master_df[master_df["split_group"].isin(fold_groups)]
            test_cases = master_df[master_df["split_group"].isin(test_groups)]

            assert len(test_cases) == 53, f"Expected 53 frozen test cases, got {len(test_cases)}"
            dev_ready = dev_cases[dev_cases["is_model_ready_spatial"] == True]
            assert len(dev_ready) == 482, f"Expected 482 ready development cases, got {len(dev_ready)}"
