import json
import random
from pathlib import Path
from typing import Tuple

import pytest
import tensorflow as tf
from athology_ml.ml.jump_detection.preprocessing import DATASET_KWARGS
from tensorflow.data import Dataset

FIXTURES_DIR = Path(__file__).parent.absolute() / "fixtures"
NUM_TIMESTEPS = 128


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


@pytest.fixture()
def dummy_jump_detection_dataset() -> Tuple[Dataset, Dataset, Dataset]:
    train_file_pattern = str(FIXTURES_DIR / "data/jump_detection/train/*.csv")
    valid_file_pattern = str(FIXTURES_DIR / "data/jump_detection/valid/*.csv")
    test_file_pattern = str(FIXTURES_DIR / "data/jump_detection/test/*.csv")

    train_dataset = tf.data.experimental.make_csv_dataset(train_file_pattern, **DATASET_KWARGS)
    valid_dataset = tf.data.experimental.make_csv_dataset(valid_file_pattern, **DATASET_KWARGS)
    test_dataset = tf.data.experimental.make_csv_dataset(test_file_pattern, **DATASET_KWARGS)

    return train_dataset, valid_dataset, test_dataset
