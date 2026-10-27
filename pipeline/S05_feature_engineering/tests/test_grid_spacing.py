"""
test_grid_spacing.py – The committed case features use each case's own grid spacing.

Five realistic cases run on 0.5 m grids; computing their features with 1 m doubles
every length and halves every frontal-area density.
"""

import pathlib

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_case_features_use_the_loader_grid_spacing():
    features = pd.read_csv(REPO / "pipeline/S05_feature_engineering/outputs/case_features.csv", usecols=["case", "dx"])
    targets = pd.read_csv(REPO / "pipeline/S02_preprocessing/outputs/case_targets.csv", usecols=["case", "dx"])
    # frozen test cases have no values in the S02 table, which never opens them
    both = features.merge(targets.dropna(), on="case", suffixes=("_features", "_loader"))
    assert len(both) == 482
    assert (both["dx_features"] == both["dx_loader"]).all()
