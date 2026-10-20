from pathlib import Path

from fastapi.testclient import TestClient

from heart_mlops import api

SAMPLE = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 0,
    "thal": 6,
}


def test_health_predict_and_metrics(monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", Path("artifacts/heart_disease_model.joblib"))

    with TestClient(api.app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=SAMPLE)
        metrics = client.get("/metrics")

    assert health.status_code == 200
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] in (0, 1)
    assert 0 <= prediction.json()["confidence"] <= 1
    assert prediction.json()["disclaimer"]
    assert metrics.status_code == 200
    assert "heart_api_requests_total" in metrics.text


def test_predict_rejects_incomplete_payload(monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", Path("artifacts/heart_disease_model.joblib"))

    with TestClient(api.app) as client:
        response = client.post("/predict", json={"age": 63})

    assert response.status_code == 422