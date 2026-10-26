"""
data.py – the single loader for UrbanTALES case files.

Every stage reads simulation data through ``load_case`` so the alignment and
normalisation rules below are applied once, in one place:

- the ``_topo`` raster is stored top row first; it is flipped (``np.flipud``) so
  that row ``i`` is the same physical y as row ``i`` of the ``_ped.nc`` fields;
- open ground is taken from the raster (``H == 0``), never from NaN in the wind
  file (some files store 0 inside buildings);
- wind fields are divided by the case's own friction velocity ``u_tau`` from the
  official metadata sheet (TKE by ``u_tau**2``);
- the wind direction comes from the pressure-gradient forcing ``dpdxy`` in the
  ``_p3d`` file, not from the case name.

Public API
----------
load_case(case)        : dict with H, fluid, U, u, v, TKE, dx, wind_dir_deg, u_tau, subset, split_group
case_targets(case)     : open-ground summary statistics of U/u_tau and TKE/u_tau²
case_table()           : one row per case with subset, split_group, dx and u_tau
EXCLUDE, FLAG          : case name -> reason
"""

from __future__ import annotations

import functools
import math
import pathlib
import re
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Cases removed from every analysis.
EXCLUDE: dict[str, str] = {
    "CN-BE-V2_d00": "downloadable building map does not match the simulated domain",
    "CN-CD-V1_d00": "downloadable building map does not match the simulated domain",
    "CN-SH-V1_d00": "downloadable building map does not match the simulated domain",
}

# Cases kept for training but reported in a sensitivity check without them.
FLAG: dict[str, str] = {
    "JP-Tok-V4_d00": "run stopped at 4.9 h, so its fields average 0.9 h instead of 8 h (flow already stationary)",
    "UG-UG-U1_d00": "building map and wind field disagree on up to 13% of building cells",
    "UG-UG-U2_d00": "building map and wind field disagree on up to 13% of building cells",
    "UG-UG-U3_d00": "building map and wind field disagree on up to 13% of building cells",
    "NP-NP-U1_d00": "building map and wind field disagree on up to 13% of building cells",
    "AU-Mel-U9_d00": "building map and wind field disagree on up to 13% of building cells",
}

_DPDXY = re.compile(r"dpdxy\s*=\s*([-+.\deE]+)\s*,\s*([-+.\deE]+)")
_GRID = re.compile(r"\bdx\s*=\s*([.\d]+)")
TARGETS = ("U_mean", "U_p95", "U_std", "TKE_mean")


@functools.lru_cache(maxsize=1)
def case_table() -> pd.DataFrame:
    """Catalog joined to the official metadata sheet, indexed by portal case name."""
    catalog = pd.read_csv(DATA_DIR / "catalog.csv", usecols=["case", "subset", "split_group"])
    names = pd.read_csv(DATA_DIR / "portal_to_metadata_map.csv")
    meta = pd.read_csv(DATA_DIR / "metadata.csv", usecols=["NameE", "$u_{\\tau}$"])
    meta = meta.rename(columns={"NameE": "metadata_NameE", "$u_{\\tau}$": "u_tau"})
    table = catalog.merge(names, left_on="case", right_on="portal_case", how="left")
    table = table.merge(meta, on="metadata_NameE", how="left")
    table["status"] = np.where(table["case"].isin(EXCLUDE), "exclude",
                               np.where(table["case"].isin(FLAG), "flag", "ok"))
    return table.drop(columns=["portal_case"]).set_index("case")


def case_path(case: str, kind: str, raw_dir: pathlib.Path = RAW_DIR) -> pathlib.Path:
    """Path of one case file; kind is topo, ped, p3d or profile."""
    name = {"topo": f"{case}_topo", "ped": f"{case}_ped.nc", "p3d": f"{case}_p3d",
            "profile": f"profile-{case}.csv"}[kind]
    return raw_dir / case / name


def parse_p3d(text: str) -> tuple[float, float]:
    """Return (wind direction in degrees, grid spacing in metres) from a _p3d namelist.

    The direction is where the mean flow goes, measured from +x (array columns)
    towards +y (array rows): the flow follows the force, minus the pressure gradient.
    """
    match = _DPDXY.search(text)
    if match is None:
        raise ValueError("no dpdxy forcing found in p3d text")
    dpdx, dpdy = float(match.group(1)), float(match.group(2))
    direction = math.degrees(math.atan2(-dpdy, -dpdx)) % 360.0
    return round(direction, 3) % 360.0, float(_GRID.search(text).group(1))


def load_topo(case: str, raw_dir: pathlib.Path = RAW_DIR) -> np.ndarray:
    """Building heights in metres, flipped to match the _ped.nc row order."""
    return np.flipud(np.loadtxt(case_path(case, "topo", raw_dir)))


def load_case(case: str, raw_dir: pathlib.Path = RAW_DIR) -> dict[str, Any]:
    """Load one case with every alignment and normalisation rule applied.

    Returns a dict with:
      H            building height [m], shape (ny, nx); 0 on open ground
      fluid        bool, True on open ground (H == 0)
      U, u, v      wind speed and components divided by u_tau; NaN inside buildings
      TKE          turbulent kinetic energy divided by u_tau**2; NaN inside buildings
      dx           grid spacing [m]
      wind_dir_deg forcing direction, 0 = +columns, 90 = +rows
      u_tau, subset, split_group, status
    """
    info = case_table().loc[case]
    if not np.isfinite(info["u_tau"]):
        raise ValueError(f"{case}: no friction velocity in the metadata sheet ({EXCLUDE.get(case, 'unmatched')})")
    H = load_topo(case, raw_dir)
    wind_dir, dx = parse_p3d(case_path(case, "p3d", raw_dir).read_text())
    fluid = H == 0
    out: dict[str, Any] = {"H": H, "fluid": fluid, "dx": dx, "wind_dir_deg": wind_dir,
                           "u_tau": float(info["u_tau"]), "subset": info["subset"],
                           "split_group": info["split_group"], "status": info["status"]}
    with h5py.File(case_path(case, "ped", raw_dir), "r") as f:
        for key, var, power in (("U", "Uped", 1), ("u", "uped", 1), ("v", "vped", 1), ("TKE", "TKEped", 2)):
            field = np.array(f[var], dtype=np.float64)
            if field.shape != H.shape:
                raise ValueError(f"{case}: {var} shape {field.shape} != topo shape {H.shape}")
            out[key] = np.where(fluid, field, np.nan) / out["u_tau"] ** power
    return out


def case_targets(case: str, raw_dir: pathlib.Path = RAW_DIR) -> dict[str, float]:
    """Open-ground summary statistics used as case-level regression targets."""
    c = load_case(case, raw_dir)
    U = c["U"][np.isfinite(c["U"])]
    tke = c["TKE"][np.isfinite(c["TKE"])]
    return {
        "U_mean": float(U.mean()),
        "U_p95": float(np.percentile(U, 95)),
        "U_std": float(U.std()),
        "TKE_mean": float(tke.mean()),
        "n_open": int(c["fluid"].sum()),
        "open_nan_frac": float(1.0 - U.size / c["fluid"].sum()),
        "wind_dir_deg": c["wind_dir_deg"],
        "dx": c["dx"],
        "u_tau": c["u_tau"],
    }
