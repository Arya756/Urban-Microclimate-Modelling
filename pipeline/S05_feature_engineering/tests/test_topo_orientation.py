"""
test_topo_orientation.py - Regression tests for S05 topography row-orientation fix.

Issue: _topo text files are stored top-row-first (row 0 = maximum physical y),
while _ped.nc wind maps use bottom-row-first (row 0 = minimum physical y).
Fix: np.flipud(np.loadtxt(topo_path)) in process_cases.py line ~326.

Tests verify:
  1. A single flipud aligns topo and wind-map building masks correctly.
  2. A second flipud (double-flip) restores the misalignment.
  3. fliplr/rot180 do NOT fix the vertical mismatch.
  4. Perfectly symmetric UA* grids are invariant to flipud.
  5. Real data: flipud gives >=99% (idealized) / >=97% (realistic) agreement.
  6. Real data: no-flip gives <90% agreement for realistic cases.
"""
from __future__ import annotations

import pathlib
import h5py
import numpy as np
import pytest

TEST_DIR  = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[2]
RAW_DIR   = REPO_ROOT / "data" / "raw"

REAL_DATA_AVAILABLE = RAW_DIR.exists() and any(RAW_DIR.iterdir())


def _pixel_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Fraction of pixels where two boolean masks agree."""
    assert mask_a.shape == mask_b.shape, f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}"
    return float((mask_a == mask_b).sum()) / mask_a.size


def _load_case(case_id: str):
    """Load raw topo (no flip) and wind-map building mask (NaN=building)."""
    case_dir  = RAW_DIR / case_id
    H_raw = np.loadtxt(case_dir / f"{case_id}_topo", dtype=np.float64)
    with h5py.File(str(case_dir / f"{case_id}_ped.nc"), "r") as f:
        u = np.array(f["Uped"], dtype=np.float32)
    return H_raw, np.isnan(u)


# ---------------------------------------------------------------------------
# Synthetic tests (no raw data required)
# ---------------------------------------------------------------------------

class TestSyntheticOrientation:

    def _mismatched_pair(self, rows=20, cols=30, seed=0):
        """
        Reference mask = bottom-row-first (wind-map convention).
        H_stored = vertically flipped reference (as topo files are stored).
        """
        rng = np.random.default_rng(seed)
        ref_mask = rng.random((rows, cols)) < 0.25       # ~25% buildings
        H_stored = np.flipud(ref_mask.astype(np.float64) * 10.0)  # simulate file
        return H_stored, ref_mask

    def test_flipud_restores_alignment(self):
        """One flipud must give 100% pixel agreement with the reference mask."""
        H_stored, ref = self._mismatched_pair()
        agree_raw  = _pixel_agreement(H_stored        > 0.0, ref)
        agree_flip = _pixel_agreement(np.flipud(H_stored) > 0.0, ref)
        assert agree_flip == 1.0, f"Expected 1.0, got {agree_flip}"
        assert agree_raw < agree_flip

    def test_double_flip_restores_mismatch(self):
        """Two flipuds cancel each other: agreement returns to pre-fix level."""
        H_stored, ref = self._mismatched_pair()
        H_twice = np.flipud(np.flipud(H_stored))
        assert _pixel_agreement(np.flipud(H_stored) > 0.0, ref) == 1.0
        assert np.allclose(H_twice, H_stored), "Double flip must reproduce original array"
        assert _pixel_agreement(H_twice > 0.0, ref) == _pixel_agreement(H_stored > 0.0, ref)

    def test_fliplr_does_not_fix_mismatch(self):
        """fliplr is a horizontal mirror and must NOT restore vertical alignment."""
        H_stored, ref = self._mismatched_pair(seed=7)
        agree_ud = _pixel_agreement(np.flipud(H_stored) > 0.0, ref)
        agree_lr = _pixel_agreement(np.fliplr(H_stored) > 0.0, ref)
        assert agree_ud > agree_lr, f"flipud={agree_ud:.4f} should beat fliplr={agree_lr:.4f}"

    def test_rot180_does_not_beat_flipud(self):
        """rot180 (flipud+fliplr) must not be strictly better than flipud alone."""
        H_stored, ref = self._mismatched_pair(seed=13)
        agree_ud  = _pixel_agreement(np.flipud(H_stored)   > 0.0, ref)
        agree_180 = _pixel_agreement(np.rot90(H_stored, 2) > 0.0, ref)
        assert agree_ud >= agree_180, f"flipud={agree_ud:.4f} should be >= rot180={agree_180:.4f}"

    def test_symmetric_ua_case_invariant_to_flip(self):
        """Vertically symmetric grids (UA*): flipud(H)==H so agreement is identical."""
        base = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0], dtype=np.float64) * 5.0
        H = np.tile(base, (10, 1))       # all rows identical -> flipud(H) == H
        assert np.array_equal(H, np.flipud(H))
        ref = H > 0.0
        assert _pixel_agreement(H > 0.0, ref) == _pixel_agreement(np.flipud(H) > 0.0, ref) == 1.0

    def test_corrected_mask_contains_buildings(self):
        """After flipud, the building mask must not be empty."""
        H_stored, _ = self._mismatched_pair(seed=99)
        frac = (np.flipud(H_stored) > 0.0).mean()
        assert frac > 0.05, f"Building fraction too low after correction: {frac:.4f}"

    def test_flipud_preserves_shape(self):
        """flipud must not change array dimensions."""
        H = np.random.default_rng(0).random((37, 53)) * 20.0
        assert np.flipud(H).shape == H.shape


# ---------------------------------------------------------------------------
# Real-data tests (skipped when data/raw is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="data/raw not present")
class TestRealDataOrientation:

    def _pick_cases(self):
        ideal = real = None
        for d in sorted(RAW_DIR.iterdir()):
            if not d.is_dir():
                continue
            cid = d.name
            if not (d / f"{cid}_topo").exists() or not (d / f"{cid}_ped.nc").exists():
                continue
            if ideal is None and cid.startswith("VS"):
                ideal = cid
            if real is None and not cid.startswith(("VS", "UA")):
                real = cid
            if ideal and real:
                break
        return ideal, real

    def test_idealized_flipud_gte_99(self):
        """VS* case: flipud pixel agreement must be >= 99%."""
        ideal, _ = self._pick_cases()
        if not ideal:
            pytest.skip("No VS* case found")
        H, wb = _load_case(ideal)
        raw_a  = _pixel_agreement(H          > 0.0, wb)
        flip_a = _pixel_agreement(np.flipud(H) > 0.0, wb)
        assert flip_a >= 0.99, f"{ideal}: flipud={flip_a:.4f} < 0.99"
        assert flip_a > raw_a,  f"{ideal}: flipud={flip_a:.4f} should exceed raw={raw_a:.4f}"

    def test_realistic_flipud_gte_97(self):
        """Realistic case: flipud must give >= 97% pixel agreement."""
        _, real = self._pick_cases()
        if not real:
            pytest.skip("No realistic case found")
        H, wb = _load_case(real)
        raw_a  = _pixel_agreement(H          > 0.0, wb)
        flip_a = _pixel_agreement(np.flipud(H) > 0.0, wb)
        assert flip_a >= 0.97, f"{real}: flipud={flip_a:.4f} < 0.97"
        assert flip_a > raw_a,  f"{real}: flipud={flip_a:.4f} should exceed raw={raw_a:.4f}"

    def test_realistic_no_flip_below_90(self):
        """Realistic case without flipud must have <90% agreement (confirms mismatch exists)."""
        _, real = self._pick_cases()
        if not real:
            pytest.skip("No realistic case found")
        H, wb = _load_case(real)
        assert _pixel_agreement(H > 0.0, wb) < 0.90, "Expected <90% without flip for realistic case"

    def test_flipud_beats_fliplr_on_real(self):
        """flipud must give higher agreement than fliplr for both sample cases."""
        ideal, real = self._pick_cases()
        for cid in [ideal, real]:
            if not cid:
                continue
            H, wb = _load_case(cid)
            assert _pixel_agreement(np.flipud(H) > 0.0, wb) >= _pixel_agreement(np.fliplr(H) > 0.0, wb), \
                f"{cid}: flipud should be >= fliplr"

    def test_shapes_match_after_flipud(self):
        """flipud must preserve shape and match wind-map shape for both cases."""
        ideal, real = self._pick_cases()
        for cid in [ideal, real]:
            if not cid:
                continue
            H, wb = _load_case(cid)
            H_flip = np.flipud(H)
            assert H_flip.shape == H.shape,  f"{cid}: shape changed by flipud"
            assert H_flip.shape == wb.shape, f"{cid}: flipped topo shape != wind map shape"
