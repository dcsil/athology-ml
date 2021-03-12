from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import Dataset

BATCH_SIZE = 1
NUM_TIMESTEPS = 128

TRAIN_FILE_PATTERN = "train/*.csv"
VALID_FILE_PATTERN = "valid/*.csv"
TEST_FILE_PATTERN = "test/*.csv"


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    # Input data for the LSTM must be 3D, add a leading axis
    features = tf.expand_dims(features, axis=0)
    return features, labels


def get_datasets(directory: str, **kwargs) -> Tuple[Dataset, Dataset, Dataset]:
    """Return a tuple of tensorflow `Dataset`s corresponding to the CSV files at
    `directory/TRAIN_FILE_PATTERN`, `directory/VALID_FILE_PATTERN`, and `directory/TEST_FILE_PATTERN`.
    Extra `**kwargs` are passed to `tf.data.experimental.make_csv_dataset`, overwriting sensible
    defaults set in this function. For details on these arguments,
    see: https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset
    """
    dataset_kwargs = {
        "batch_size": NUM_TIMESTEPS,  # This is actually the number of timesteps within a window
        "label_name": "is_air",
        "select_columns": ["x-axis (g)", "y-axis (g)", "z-axis (g)", "is_air"],
        "header": True,
        "num_epochs": 1,  # Will set num_epochs within model.fit()
        "shuffle": False,  # False to sample windows as they appear in the input
    }
    dataset_kwargs.update(kwargs)

    train_file_pattern = str(Path(directory) / TRAIN_FILE_PATTERN)
    valid_file_pattern = str(Path(directory) / VALID_FILE_PATTERN)
    test_file_pattern = str(Path(directory) / TEST_FILE_PATTERN)

    train_dataset = tf.data.experimental.make_csv_dataset(train_file_pattern, **dataset_kwargs)
    valid_dataset = tf.data.experimental.make_csv_dataset(valid_file_pattern, **dataset_kwargs)
    test_dataset = tf.data.experimental.make_csv_dataset(test_file_pattern, **dataset_kwargs)

    train_dataset = train_dataset.map(pack_features_vector, num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.map(pack_features_vector, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(pack_features_vector, num_parallel_calls=tf.data.AUTOTUNE)

    return train_dataset, valid_dataset, test_dataset


def get_features_and_labels(
    dataset: Dataset, num_rows: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of NumPy arrays, containing the features and labels from `dataset`.
    If `num_rows` is provided, only the first `num_rows` of data are included.
    """
    all_features = []
    all_labels = []

    for features, labels in iter(dataset):
        features = features.numpy()
        labels = labels.numpy()

        # Flatten the batch dimension
        all_features.append(features.reshape(-1, features.shape[-1]))
        all_labels.append(labels.reshape(-1))

        if num_rows is not None and len(all_features) >= num_rows:
            break

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def get_classifier_bias_init(labels: np.ndarray) -> float:
    """Computes a more sensible initial value for the classifiers bias based on the ratio
    of negative to positive class instances in labels. For more details,
    see: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
    """
    pos = labels.sum()
    neg = labels.shape[0] - pos
    classifier_bias = np.log(pos / neg)
    return classifier_bias


def get_normalizer(features: np.ndarray):
    """Returns a Keras compatible normalization layer trained on `features`."""
    # Compute the sum and norm of our accelerometer data.
    # Train a normalization layer on all features.
    sum_ = np.sum(features, axis=-1, keepdims=True)
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    features = np.concatenate((features, sum_, norm), axis=-1)
    normalizer = preprocessing.Normalization(-1)
    normalizer.adapt(features)
    return normalizer
