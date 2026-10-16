"""Data tests: digits.csv integrity and boundaries."""
import os

import pandas as pd

DATA_PATH = "data/digits.csv"


def test_data_file_exists():
    assert os.path.exists(DATA_PATH)


def test_data_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (1797, 65)


def test_feature_columns():
    df = pd.read_csv(DATA_PATH)
    expected = [f"pixel_{i}" for i in range(64)] + ["target"]
    assert list(df.columns) == expected


def test_pixel_range():
    """Edge: all pixels must be within [0, 16]."""
    df = pd.read_csv(DATA_PATH)
    pixels = df.drop(columns=["target"])
    assert pixels.min().min() >= 0
    assert pixels.max().max() <= 16


def test_target_labels():
    df = pd.read_csv(DATA_PATH)
    assert set(df["target"].unique()) <= set(range(10))


def test_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert df.isna().sum().sum() == 0
