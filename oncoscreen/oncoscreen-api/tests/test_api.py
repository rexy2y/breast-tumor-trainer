from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("model.joblib").exists(), reason="model.joblib not found"
)


def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_health():
    assert client().get("/health").json() == {"status": "ok"}


def test_predict_wrong_length():
    assert client().post("/predict", json={"features": [1.0]}).status_code == 422


def test_predict_ok():
    c = client()
    n = len(c.get("/features").json())
    r = c.post("/predict", json={"features": [1.0] * n})
    assert r.status_code == 200 and "prediction" in r.json()
