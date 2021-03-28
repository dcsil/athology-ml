from athology_ml.app import main
from fastapi.testclient import TestClient
import json


def test_docs() -> None:
    with TestClient(main.app) as client:
        response = client.get("/docs")
        assert response.status_code == 200


def test_health_check() -> None:
    with TestClient(main.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "OK"


def test_jump_detection_endpoint(dummy_accelerometer_data: str) -> None:
    with TestClient(main.app) as client:
        response = client.post("/jump-detection", dummy_accelerometer_data)
        assert response.status_code == 200
        assert response.json()["message"] == "OK"

        expected_len = len(json.loads(dummy_accelerometer_data)["x"])
        assert len(response.json()["data"]["is_jumping"]) == expected_len
        assert [pred in [True, False] for pred in response.json()["data"]["is_jumping"]]


def test_jump_detection_endpoint_unequal_timesteps(
    dummy_accelerometer_data_unequal_timesteps: str,
) -> None:
    with TestClient(main.app) as client:
        response = client.post("/jump-detection", dummy_accelerometer_data_unequal_timesteps)
        assert response.status_code == 422
        assert (
            response.json()["detail"][0]["msg"]
            == "x, y, and z timesteps must be of equal length. Got 4, 3, and 3 respectively."
        )
