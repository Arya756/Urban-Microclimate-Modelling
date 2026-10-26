"""
test_valid_mask.py - Regression tests for S05 Issue 2: valid pedestrian
wind target cell identification.

Root cause: 163 cases (160 realistic, 3 idealized) have finite (non-NaN)
wind values at building perimeter cells due to CFD grid-staggering artefacts.
The NaN-only mask in load_normalized_target incorrectly included 1,141,511
such cells as valid targets.

Fix: load_normalized_target now accepts topo_H (the flipud-corrected
topography matrix) and unconditionally excludes cells where H > 0.

These tests verify:
  1. Building cells marked by NaN in ped.nc are excluded.
  2. Building cells marked by zero in ped.nc are excluded.
  3. Building cells with spurious positive (artefact) values are excluded.
  4. Legitimate zero-valued wind targets on open ground are preserved.
  5. NaN values at otherwise open cells are excluded.
  6. Infinite values at otherwise open cells are excluded.
  7. Shape mismatch between topo_H and ped.nc raises ValueError.
  8. Correct spatial alignment: topo_H after flipud matches wind map.
  9. Without topo_H (backward compat): NaN-only mask is used.
  10. Real-data: corrected mask removes all building-cell artefacts.
"""
from __future__ import annotations

import pathlib
import tempfile
import struct

import h5py
import numpy as np
import pytest

TEST_DIR  = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[2]
RAW_DIR   = REPO_ROOT / "data" / "raw"

REAL_DATA_AVAILABLE = RAW_DIR.exists() and any(RAW_DIR.iterdir())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ped_nc(path: pathlib.Path, u: np.ndarray, tke: np.ndarray | None = None):
    """Write a synthetic _ped.nc file with Uped (and optionally TKEped)."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("Uped",   data=u.astype(np.float32))
        if tke is not None:
            f.create_dataset("TKEped", data=tke.astype(np.float32))


def _call_target(tmp_path, u, tke=None, topo_H=None, u_tau=1.0):
    """Write synthetic ped.nc and call load_normalized_target."""
    # Import here to get the patched version
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from pipeline.S05_feature_engineering.feature_assembly import load_normalized_target
    ped = tmp_path / "case_ped.nc"
    _write_ped_nc(ped, u, tke)
    return load_normalized_target(ped, u_tau=u_tau, topo_H=topo_H)


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

class TestValidMaskSynthetic:

    def test_building_nan_excluded(self, tmp_path):
        """NaN at building cell must be excluded from valid_mask."""
        u = np.array([[np.nan, 0.5],
                      [0.3,   0.4]], dtype=np.float32)
        H = np.array([[1.0,   0.0],   # row 0 col 0 = building
                      [0.0,   0.0]], dtype=np.float64)
        Y, _, vm = _call_target(tmp_path, u, topo_H=H)
        assert not vm[0, 0], "NaN building cell must be invalid"
        assert vm[0, 1] and vm[1, 0] and vm[1, 1], "Open cells must be valid"

    def test_building_zero_excluded(self, tmp_path):
        """Zero at building cell must be excluded (ped.nc sometimes uses 0 not NaN)."""
        u = np.array([[0.0,   0.5],
                      [0.3,   0.4]], dtype=np.float32)
        H = np.array([[5.0,   0.0],   # building at [0,0]
                      [0.0,   0.0]], dtype=np.float64)
        Y, _, vm = _call_target(tmp_path, u, topo_H=H)
        assert not vm[0, 0], "Zero building cell must be excluded via topo_H"
        assert vm[0, 1] and vm[1, 0] and vm[1, 1]

    def test_building_positive_artefact_excluded(self, tmp_path):
        """Positive finite artefact value at building cell must be excluded."""
        u = np.array([[0.8,   0.5],   # 0.8 is an artefact at building
                      [0.3,   0.4]], dtype=np.float32)
        H = np.array([[10.0,  0.0],   # building at [0,0]
                      [0.0,   0.0]], dtype=np.float64)
        Y, _, vm = _call_target(tmp_path, u, topo_H=H)
        assert not vm[0, 0], "Positive artefact at building must be excluded"
        assert Y[0, 0] == 0.0, "Y at excluded building cell must be zero"
        assert vm[0, 1] and vm[1, 0] and vm[1, 1]

    def test_legit_zero_on_open_ground_preserved(self, tmp_path):
        """Zero wind speed on open ground (calm pocket) must remain valid."""
        u = np.array([[np.nan, 0.0],   # [0,1]: open, zero wind (legitimate)
                      [0.3,   0.4]], dtype=np.float32)
        H = np.array([[8.0,   0.0],    # [0,0]: building; [0,1]: open ground
                      [0.0,   0.0]], dtype=np.float64)
        Y, _, vm = _call_target(tmp_path, u, topo_H=H)
        assert not vm[0, 0], "Building must be excluded"
        assert vm[0, 1], "Legitimate zero on open ground must be valid"
        assert Y[0, 1] == 0.0, "Y at zero open cell must be 0.0"

    def test_nan_at_open_cell_excluded(self, tmp_path):
        """NaN at an open (non-building) cell must be excluded."""
        u = np.array([[0.5,   np.nan],
                      [0.3,   0.4]], dtype=np.float32)
        H = np.array([[0.0,   0.0],    # all open ground
                      [0.0,   0.0]], dtype=np.float64)
        Y, _, vm = _call_target(tmp_path, u, topo_H=H)
        assert vm[0, 0] and vm[1, 0] and vm[1, 1]
        assert not vm[0, 1], "NaN at open cell must be invalid"

    def test_inf_at_open_cell_excluded(self, tmp_path):
        """Infinite value at an open cell must be excluded."""
        u = np.array([[0.5,   np.inf],
                      [0.3,   0.4]], dtype=np.float32)
        H = np.array([[0.0,   0.0],
                      [0.0,   0.0]], dtype=np.float64)
        Y, _, vm = _call_target(tmp_path, u, topo_H=H)
        assert not vm[0, 1], "inf at open cell must be excluded"
        assert vm[0, 0] and vm[1, 0] and vm[1, 1]

    def test_shape_mismatch_raises(self, tmp_path):
        """Mismatched topo_H and Uped shapes must raise ValueError."""
        u = np.zeros((4, 6), dtype=np.float32)
        H = np.zeros((3, 5), dtype=np.float64)   # wrong shape
        with pytest.raises(ValueError, match="shape"):
            _call_target(tmp_path, u, topo_H=H)

    def test_normalisation_correct(self, tmp_path):
        """Y = u / u_tau at valid open cells."""
        u = np.array([[np.nan, 2.0],
                      [3.0,   4.0]], dtype=np.float32)
        H = np.zeros((2, 2), dtype=np.float64)   # all open
        Y, _, vm = _call_target(tmp_path, u, topo_H=H, u_tau=2.0)
        assert not vm[0, 0]
        assert vm[0, 1] and vm[1, 0] and vm[1, 1]
        np.testing.assert_allclose(Y[0, 1], 1.0, atol=1e-6)
        np.testing.assert_allclose(Y[1, 0], 1.5, atol=1e-6)
        np.testing.assert_allclose(Y[1, 1], 2.0, atol=1e-6)

    def test_backward_compat_no_topo_h(self, tmp_path):
        """Without topo_H the mask falls back to isfinite(u) (NaN-based)."""
        u = np.array([[np.nan, 0.5],
                      [0.3,   0.4]], dtype=np.float32)
        Y, _, vm = _call_target(tmp_path, u, topo_H=None)
        assert not vm[0, 0], "NaN cell must still be excluded without topo_H"
        assert vm[0, 1] and vm[1, 0] and vm[1, 1]

    def test_all_building_domain_returns_empty_mask(self, tmp_path):
        """When all cells are buildings, valid_mask must be all False."""
        u = np.ones((3, 4), dtype=np.float32) * 0.5
        H = np.ones((3, 4), dtype=np.float64) * 5.0   # all buildings
        Y, _, vm = _call_target(tmp_path, u, topo_H=H)
        assert not vm.any(), "No valid cells in all-building domain"
        assert (Y == 0.0).all(), "Y must be zero everywhere"

    def test_tke_normalisation_and_mask(self, tmp_path):
        """TKE is normalised by u_tau^2; NaN in TKE only zeroes tke_norm, does not affect valid_mask."""
        u   = np.array([[np.nan, 1.0],  [1.0,  1.0]], dtype=np.float32)
        tke = np.array([[0.5,   2.0],  [3.0,  np.nan]], dtype=np.float32)
        H   = np.zeros((2, 2), dtype=np.float64)
        _, tke_norm, vm = _call_target(tmp_path, u, tke=tke, topo_H=H, u_tau=2.0)
        # valid_mask is governed by Uped + topo_H, not TKEped
        assert not vm[0, 0], "NaN in Uped must make cell invalid"
        assert vm[0, 1] and vm[1, 0] and vm[1, 1], "All other open cells with finite u are valid"
        # TKE normalisation
        np.testing.assert_allclose(tke_norm[0, 1], 2.0 / 4.0, atol=1e-6)
        np.testing.assert_allclose(tke_norm[1, 0], 3.0 / 4.0, atol=1e-6)
        # NaN TKE at valid cell → tke_norm is 0.0 at that cell
        assert tke_norm[1, 1] == 0.0, "NaN TKE at valid cell must yield zero tke_norm"
        assert tke_norm[0, 0] == 0.0, "Invalid cell must have zero tke_norm"

    def test_orientation_consistency_with_issue1_fix(self, tmp_path):
        """
        Simulate the Issue 1 + Issue 2 interaction: topo_H must be the
        flipud-corrected array so it matches the ped.nc spatial frame.
        """
        # Topo as stored (top-row-first): building at row 0
        H_stored = np.array([[5.0, 5.0],
                              [0.0, 0.0]], dtype=np.float64)
        # ped.nc uses bottom-row-first: building at row 1 (after flipud of topo)
        # i.e. open cells at row 0, building at row 1
        u = np.array([[0.8, 0.9],    # open cells (row 0 in ped.nc frame)
                      [0.3, np.nan]], dtype=np.float32)   # row 1: artefact + NaN

        # Issue 1 fix: flip before use
        H_corrected = np.flipud(H_stored)
        # Now H_corrected[1] = [5,5] = building, H_corrected[0] = [0,0] = open
        # ped.nc row 1: artefact value 0.3 at building [1,0] + NaN at [1,1]
        # Both should be excluded.

        Y, _, vm = _call_target(tmp_path, u, topo_H=H_corrected)
        assert vm[0, 0] and vm[0, 1], "Open cells in corrected frame must be valid"
        assert not vm[1, 0], "Building artefact cell must be excluded"
        assert not vm[1, 1], "NaN building cell must be excluded"


# ---------------------------------------------------------------------------
# Real-data tests (skipped when data/raw is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="data/raw not present")
class TestValidMaskRealData:

    def _pick_affected_case(self):
        """Find a case where topo mask removes more cells than NaN mask."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        for case_dir in sorted(RAW_DIR.iterdir()):
            if not case_dir.is_dir():
                continue
            cid = case_dir.name
            tp  = case_dir / f"{cid}_topo"
            pp  = case_dir / f"{cid}_ped.nc"
            if not tp.exists() or not pp.exists():
                continue
            H = np.flipud(np.loadtxt(tp, dtype=np.float64))
            with h5py.File(str(pp), "r") as f:
                u = np.array(f["Uped"], dtype=np.float32)
            is_bldg = H > 0.0
            artefacts = int((is_bldg & np.isfinite(u)).sum())
            if artefacts > 100:   # only cases with substantial artefacts
                return cid, H, u, artefacts
        return None, None, None, 0

    def test_topo_mask_removes_artefact_cells(self, tmp_path):
        """On a real affected case, topo mask removes building artefact cells."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from pipeline.S05_feature_engineering.feature_assembly import load_normalized_target

        cid, H, u, n_artefacts = self._pick_affected_case()
        if cid is None:
            pytest.skip("No case with substantial artefact cells found")

        ped_path = RAW_DIR / cid / f"{cid}_ped.nc"
        Y_nan, _, vm_nan  = load_normalized_target(ped_path, u_tau=0.2137)
        Y_fix, _, vm_fix  = load_normalized_target(ped_path, u_tau=0.2137, topo_H=H)

        extra_removed = int(vm_nan.sum()) - int(vm_fix.sum())
        assert extra_removed == n_artefacts, (
            f"{cid}: expected {n_artefacts} extra cells removed, got {extra_removed}"
        )

    def test_topo_mask_preserves_open_cell_count(self, tmp_path):
        """Total valid open cells must equal (open cells with finite u)."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from pipeline.S05_feature_engineering.feature_assembly import load_normalized_target

        cid, H, u, _ = self._pick_affected_case()
        if cid is None:
            pytest.skip("No affected case found")

        ped_path = RAW_DIR / cid / f"{cid}_ped.nc"
        _, _, vm_fix = load_normalized_target(ped_path, u_tau=0.2137, topo_H=H)

        expected = int(((H == 0.0) & np.isfinite(u)).sum())
        assert int(vm_fix.sum()) == expected, (
            f"{cid}: valid count {vm_fix.sum()} != expected {expected}"
        )

    def test_topo_mask_preserves_legit_zeros(self, tmp_path):
        """Legitimate zero-valued open-ground cells must remain in valid_mask."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from pipeline.S05_feature_engineering.feature_assembly import load_normalized_target

        for case_dir in sorted(RAW_DIR.iterdir())[:50]:
            if not case_dir.is_dir():
                continue
            cid = case_dir.name
            tp  = case_dir / f"{cid}_topo"
            pp  = case_dir / f"{cid}_ped.nc"
            if not tp.exists() or not pp.exists():
                continue
            H = np.flipud(np.loadtxt(tp, dtype=np.float64))
            with h5py.File(str(pp), "r") as f:
                u = np.array(f["Uped"], dtype=np.float32)
            legit_zeros = (H == 0.0) & (u == 0.0) & np.isfinite(u)
            if legit_zeros.sum() == 0:
                continue
            _, _, vm_fix = load_normalized_target(pp, u_tau=0.2137, topo_H=H)
            assert (vm_fix & legit_zeros).sum() == legit_zeros.sum(), (
                f"{cid}: legitimate zeros on open ground were removed"
            )
            return  # tested one case
        pytest.skip("No case with legitimate zeros on open ground found in first 50")

    def test_no_building_cells_in_corrected_mask(self):
        """After correction, no cell where H>0 should appear in valid_mask."""
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from pipeline.S05_feature_engineering.feature_assembly import load_normalized_target

        tested = 0
        for case_dir in sorted(RAW_DIR.iterdir()):
            if not case_dir.is_dir():
                continue
            cid = case_dir.name
            tp  = case_dir / f"{cid}_topo"
            pp  = case_dir / f"{cid}_ped.nc"
            if not tp.exists() or not pp.exists():
                continue
            H = np.flipud(np.loadtxt(tp, dtype=np.float64))
            _, _, vm = load_normalized_target(pp, u_tau=0.2137, topo_H=H)
            # No building cell (H>0) should appear in valid_mask
            assert not (vm & (H > 0.0)).any(), (
                f"{cid}: building cell found in valid_mask after topo correction"
            )
            tested += 1
            if tested >= 20:   # test a sample of 20 cases to keep runtime short
                break
        assert tested > 0
