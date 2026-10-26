"""Plot the pixel-level feature channels of example development cases.

Writes outputs/<case>_channel_grid.png (all 22 channels) and
outputs/<case>_fetch_analysis.png (upwind fetch on open ground).

Usage: python pipeline/S05_feature_engineering/plot_feature_examples.py [--cases UA0625_d00 CA-Sunset]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from urbanwind.data import case_path, load_topo, parse_p3d
from urbanwind.features import DEFAULT_CONFIG, assemble_pixel_feature_tensor

OUT = DEFAULT_CONFIG.output_dir


def features(case: str) -> tuple[np.ndarray, list[str], float]:
    H = load_topo(case)
    direction, dx = parse_p3d(case_path(case, "p3d").read_text())
    X, names = assemble_pixel_feature_tensor(H, dx=dx, forcing_dir_deg=direction)
    return X, names, direction


def channel_grid(case: str) -> None:
    X, names, direction = features(case)
    fig, axes = plt.subplots(4, 6, figsize=(22, 14))
    for ax, layer, name, idx in zip(axes.flat, X, names, range(len(names)), strict=False):
        im = ax.imshow(layer, origin="lower", cmap="gray" if idx == 1 else "viridis")
        ax.set_title(f"Ch{idx}: {name}", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle(f"22-channel pixel features: {case} (wind toward {direction:g} deg, rows = +y)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"{case}_channel_grid.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def fetch_analysis(case: str) -> None:
    X, names, direction = features(case)
    fluid = X[names.index("fluid_mask")] == 1.0
    fetch = X[names.index("upwind_fetch")]
    lp_local = X[names.index("lambda_p_8m")][fluid]
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))
    im = ax0.imshow(np.where(fluid, fetch, np.nan), origin="lower", cmap="magma")
    ax0.set_title(f"Upwind fetch on open ground (wind toward {direction:g} deg)")
    fig.colorbar(im, ax=ax0, label="m")
    ax1.hist(fetch[fluid], bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
    ax1.axvline(float(np.median(fetch[fluid])), color="red", ls="--", label=f"median {np.median(fetch[fluid]):.1f} m")
    ax1.set_xlabel("upwind fetch (m)")
    ax1.set_ylabel("open-ground cells")
    ax1.legend()
    ax2.scatter(lp_local, fetch[fluid], s=1, alpha=0.1, color="darkorange")
    ax2.set_xlabel("local plan-area density, 8 m radius")
    ax2.set_ylabel("upwind fetch (m)")
    fig.suptitle(f"Open-ground exposure: {case}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{case}_fetch_analysis.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", nargs="+", default=["UA0625_d00", "CA-Sunset"])
    for case in parser.parse_args().cases:
        channel_grid(case)
        fetch_analysis(case)
