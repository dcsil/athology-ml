from app.main import app
from fastapi.testclient import TestClient
import json

client = TestClient(app)


def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "OK"


def test_jump_detection(dummy_accelerometer_data: str):
    response = client.post("/jump-detection", dummy_accelerometer_data)
    assert response.status_code == 200
    assert response.json()["message"] == "OK"

    expected_len = len(json.loads(dummy_accelerometer_data)["x"])
    assert len(response.json()["data"]["is_jumping"]) == expected_len
    assert [pred in [0, 1] for pred in response.json()["data"]["is_jumping"]]


def test_jump_detection_unequal_timesteps(dummy_accelerometer_data_unequal_timesteps: str):
    response = client.post("/jump-detection", dummy_accelerometer_data_unequal_timesteps)
    assert response.status_code == 422
    assert (
        response.json()["detail"][0]["msg"]
        == "x, y, and z timesteps must be of equal length. Got 4, 3, and 3 respectively."
    )
