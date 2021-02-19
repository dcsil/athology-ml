import json
import random

import pytest


@pytest.fixture()
def dummy_accelerometer_data() -> str:
    timesteps = 3
    x = [random.random()] * timesteps
    y = [random.random()] * timesteps
    z = [random.random()] * timesteps
    request = {"x": x, "y": y, "z": z}
    return json.dumps(request)


@pytest.fixture()
def dummy_accelerometer_data_unequal_timesteps(
    dummy_accelerometer_data: str,
) -> str:
    request = json.loads(dummy_accelerometer_data)
    request["x"].append(random.random())
    return json.dumps(request)
