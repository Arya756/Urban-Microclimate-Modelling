"""Tests for urbanwind.data. The tests that read case files skip when data/raw is absent."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from urbanwind.data import RAW_DIR, case_path, case_table, case_targets, load_case, parse_p3d

needs_raw = pytest.mark.skipif(not (RAW_DIR / "AU-Ade-V1_d00").exists(), reason="raw UrbanTALES files not downloaded")


@pytest.mark.parametrize(
    "text, direction, dx",
    [
        ("dx = 1, dy = 1, dz = 0.5,\n dpdxy = -0.0003828125, 0.0, ! utau", 0.0, 1.0),
        ("dx = 0.5, dy = 0.5,\ndpdxy = -0.000270689314672976, -0.000270689314672976,", 45.0, 0.5),
        ("dx = 1, dy = 1,\ndpdxy = 0, 0.000382812499999999", 270.0, 1.0),
        ("dx = 1,\ndpdxy = 0.0, -3.8e-4,", 90.0, 1.0),
    ],
)
def test_parse_p3d_direction_and_grid(text, direction, dx):
    assert parse_p3d(text) == (direction, dx)


def test_case_table_joins_renamed_cases():
    table = case_table()
    assert len(table) == 538
    assert table.loc["AR-BA-U1_d00", "metadata_NameE"] == "AR-Bue-U1_d00"
    assert 0.2 < table.loc["AR-BA-U1_d00", "u_tau"] < 0.22
    assert table["u_tau"].isna().sum() == 3  # the three CN cases with no matching metadata row
    assert table.loc["CN-CD-V1_d00", "status"] == "exclude"
    assert table.loc["JP-Tok-V4_d00", "status"] == "flag"


@needs_raw
def test_flip_aligns_topo_with_wind_field():
    raw_topo = np.loadtxt(case_path("AU-Ade-V1_d00", "topo"))
    with h5py.File(case_path("AU-Ade-V1_d00", "ped"), "r") as f:
        wind_open = np.isfinite(np.array(f["Uped"]))
    flipped = load_case("AU-Ade-V1_d00")["fluid"]
    assert np.mean(flipped == wind_open) > 0.97
    assert np.mean((raw_topo == 0) == wind_open) < 0.7


@needs_raw
def test_mask_comes_from_topo_not_nan():
    case = "AU-Mel-U1_d00"  # stores 0 instead of NaN inside buildings
    with h5py.File(case_path(case, "ped"), "r") as f:
        raw = np.array(f["Uped"])
    c = load_case(case)
    assert np.isfinite(raw[~c["fluid"]]).mean() > 0.2  # about a quarter of building cells hold values
    assert np.isnan(c["U"][~c["fluid"]]).all()


@needs_raw
def test_case_targets_reference_values():
    t = case_targets("AU-Ade-V1_d00")
    assert t["U_mean"] == pytest.approx(2.465, abs=1e-3)
    assert t["U_p95"] == pytest.approx(5.124, abs=1e-3)
    assert t["wind_dir_deg"] == 0.0
