"""Model tests: model file, determinism, real-sample accuracy."""
import os

import joblib
import pandas as pd

from ml.train import DATA_PATH, MODEL_PATH


def _model():
    return joblib.load(MODEL_PATH)


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)


def test_model_predicts_deterministically():
    model = _model()
    x = [0.0] * 64
    assert model.predict([x])[0] == model.predict([x])[0]


def test_model_on_real_samples():
    """Model should classify a sample of real digits correctly."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    model = _model()
    preds = model.predict(X[:50])
    correct = sum(p == t for p, t in zip(preds, y[:50]))
    assert correct >= 45  # at least 90% on the first 50 samples
