"""API tests: health and predict endpoints with edge cases."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_input():
    resp = client.post("/predict", json={"pixels": [0.0] * 64})
    assert resp.status_code == 200
    assert resp.json()["prediction"] in range(10)


def test_predict_all_black():
    """Edge: all pixels at max brightness (16.0)."""
    resp = client.post("/predict", json={"pixels": [16.0] * 64})
    assert resp.status_code == 200
    assert resp.json()["prediction"] in range(10)


def test_predict_too_few_pixels():
    """Edge: 63 pixels -> rejected."""
    resp = client.post("/predict", json={"pixels": [0.0] * 63})
    assert resp.status_code == 422


def test_predict_too_many_pixels():
    """Edge: 65 pixels -> rejected."""
    resp = client.post("/predict", json={"pixels": [0.0] * 65})
    assert resp.status_code == 422


def test_predict_empty_pixels():
    """Edge: empty list -> rejected."""
    resp = client.post("/predict", json={"pixels": []})
    assert resp.status_code == 422


def test_predict_out_of_range_value():
    """Edge: pixel value > 16 -> rejected."""
    resp = client.post("/predict", json={"pixels": [17.0] + [0.0] * 63})
    assert resp.status_code == 422


def test_predict_negative_value():
    """Edge: negative pixel -> rejected."""
    resp = client.post("/predict", json={"pixels": [-1.0] + [0.0] * 63})
    assert resp.status_code == 422


def test_predict_non_numeric_value():
    """Edge: non-numeric pixel -> rejected."""
    resp = client.post("/predict", json={"pixels": ["a"] * 64})
    assert resp.status_code == 422
