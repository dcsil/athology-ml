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


# @pytest.fixture()
# def dummy_jump_detection_dataset() -> Tuple[Dataset, Dataset, Dataset]:
#     dataset_kwargs = {
#         "batch_size": NUM_TIMESTEPS,  # This is actually the number of timesteps within a window
#         "label_name": "is_air",
#         "select_columns": ["x-axis (g)", "y-axis (g)", "z-axis (g)", "is_air"],
#         "header": True,
#         "num_epochs": 1,  # Will set num_epochs within model.fit()
#         "shuffle": False,  # False to sample windows as they appear in the input
#     }
#     train_file_pattern = str(Path("tests/fixtures/data/jump_detection") / "train.tsv")
#     valid_file_pattern = str(Path("tests/fixtures/data/jump_detection") / "valid.tsv")
#     test_file_pattern = str(Path("tests/fixtures/data/jump_detection") / "test.tsv")

#     train_dataset = tf.data.experimental.make_csv_dataset(train_file_pattern, **dataset_kwargs)
#     valid_dataset = tf.data.experimental.make_csv_dataset(valid_file_pattern, **dataset_kwargs)
#     test_dataset = tf.data.experimental.make_csv_dataset(test_file_pattern, **dataset_kwargs)

#     return train_dataset, valid_dataset, test_dataset
