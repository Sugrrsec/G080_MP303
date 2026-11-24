
import pytest
from fastapi.testclient import TestClient

from mp303 import api


class DummyModel:
    def predict(self, data):
        return [42.5]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api.cloudpickle,
        "load",
        lambda file: DummyModel()
    )

    with TestClient(api.app) as test_client:
        yield test_client


def valid_payload():
    return {
        "worker_exp_yr": 10,
        "rain_mm_week": 20,
        "slope_deg": 15,
        "temp_c": 25,
        "bush_age_yr": 8,
        "hours_worked": 7,
        "flush": "monsoon",
        "section": "Upper Garden"
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(client):
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    assert response.json() == {"prediction": 42.5}


def test_rejects_experience_above_50(client):
    payload = valid_payload()
    payload["worker_exp_yr"] = 51

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_rejects_temperature_below_5(client):
    payload = valid_payload()
    payload["temp_c"] = 4

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
