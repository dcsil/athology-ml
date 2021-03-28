from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import Dataset
import copy

BUFFER_SIZE = 500

TRAIN_FILE_PATTERN = "train/*.csv"
VALID_FILE_PATTERN = "valid/*.csv"
TEST_FILE_PATTERN = "test/*.csv"

DATASET_KWARGS = {
    "batch_size": 1,  # We will set up timesteps and batching below
    "label_name": "is_air",
    "select_columns": ["x-axis (g)", "y-axis (g)", "z-axis (g)", "is_air"],
    "header": True,
    "num_epochs": 1,  # Will set num_epochs within model.fit()
    "shuffle": False,  # False to sample windows as they appear in the input
}


def pack_features_vector(features, labels):
    """Pack the features into a single array. Adapted from:
    https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#create_a_tfdatadataset
    """
    features = tf.stack(list(features.values()), axis=1)
    # Remove all but the leading dimension.
    # We set up batching outside of this function.
    features = tf.reshape(features, shape=(-1,))
    labels = tf.reshape(labels, shape=(-1,))

    return features, labels


def squeeze_labels_vector(features, labels):
    """Remove the trailing dimension of labels."""
    labels = tf.squeeze(labels, axis=-1)
    return features, labels


def cache_shuffle_batch_prefetch(
    dataset: Dataset, batch_size: int, shuffle: bool = False, buffer_size: Optional[int] = None
):
    """Given a `dataset`, returns a new `dataset` which generates batches of size `batch_size`.
    If `shuffle`, the data is shuffled by randomly choosing items from `buffer_size` number of
    examples. Follows best practices for optimized data loading by caching and prefetching the data.

    See the individual tf.data.Dataset methods for more details:

    - [cache](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache)
    - [shuffle](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)
    - [batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch)
    - [prefetch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch)
    """
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_datasets(
    directory: str, batch_size: int, num_timesteps: int, **kwargs
) -> Tuple[Dataset, Dataset, Dataset]:
    """Return a tuple of tensorflow `Dataset`s corresponding to the CSV files at
    `directory/TRAIN_FILE_PATTERN`, `directory/VALID_FILE_PATTERN`, and `directory/TEST_FILE_PATTERN`.
    Extra `**kwargs` are passed to `tf.data.experimental.make_csv_dataset`, overwriting sensible
    defaults set in this function. For details on these arguments,
    see: https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset
    """

    dataset_kwargs = copy.deepcopy(DATASET_KWARGS)
    dataset_kwargs.update(kwargs)

    train_file_pattern = str(Path(directory) / TRAIN_FILE_PATTERN)
    valid_file_pattern = str(Path(directory) / VALID_FILE_PATTERN)
    test_file_pattern = str(Path(directory) / TEST_FILE_PATTERN)

    train_dataset = tf.data.experimental.make_csv_dataset(train_file_pattern, **dataset_kwargs)
    valid_dataset = tf.data.experimental.make_csv_dataset(valid_file_pattern, **dataset_kwargs)
    test_dataset = tf.data.experimental.make_csv_dataset(test_file_pattern, **dataset_kwargs)

    # Stack the features and labels of the dataset into tensors
    train_dataset = train_dataset.map(pack_features_vector, num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.map(pack_features_vector, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(pack_features_vector, num_parallel_calls=tf.data.AUTOTUNE)

    # Create the first batch dimension, which corresponds to timesteps
    train_dataset = train_dataset.batch(num_timesteps, drop_remainder=True)
    valid_dataset = valid_dataset.batch(num_timesteps, drop_remainder=True)
    test_dataset = test_dataset.batch(num_timesteps, drop_remainder=True)

    # Batching introduces an unnecessary trailing dimension in the labels, drop it
    train_dataset = train_dataset.map(squeeze_labels_vector, num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.map(squeeze_labels_vector, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(squeeze_labels_vector, num_parallel_calls=tf.data.AUTOTUNE)

    # Finally, we setup traditional batching, using a helper function that also implements best
    # practices (e.g. caching, prefetching)
    train_dataset = cache_shuffle_batch_prefetch(
        train_dataset, batch_size=batch_size, shuffle=True, buffer_size=BUFFER_SIZE
    )
    valid_dataset = cache_shuffle_batch_prefetch(
        valid_dataset, batch_size=batch_size, shuffle=False, buffer_size=BUFFER_SIZE
    )
    test_dataset = cache_shuffle_batch_prefetch(
        test_dataset, batch_size=batch_size, shuffle=False, buffer_size=BUFFER_SIZE
    )

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

        if num_rows is not None and len(all_features) == num_rows:
            break

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def get_classifier_bias_init(labels: np.ndarray) -> float:
    """Computes a more sensible initial value for the classifiers bias based on the ratio
    of negative to positive class instances in labels. For more details,
    see: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias

    Returns None if there is no support for either the positive or negative class.
    Raises a value error if labels is an empty array.
    """
    num_labels = labels.size
    if not num_labels:
        raise ValueError(f"Labels must be a non-empty array. Got shape: {labels.shape}")
    pos = labels.sum()
    neg = labels.size - pos
    classifier_bias = np.log(pos / neg) if pos > 0 and neg > 0 else None
    return classifier_bias


def get_normalizer(features: np.ndarray):
    """Returns a Keras compatible normalization layer trained on `features`."""
    # Compute the sum and norm of our accelerometer data.
    # Train a normalization layer on all features.
    if any(not dim for dim in features.shape):
        raise ValueError(
            f"All dimensions in features must be non-zero. Got shape: {features.shape}"
        )
    sum_ = np.sum(features, axis=-1, keepdims=True)
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    features = np.concatenate((features, sum_, norm), axis=-1)
    normalizer = preprocessing.Normalization(-1)
    normalizer.adapt(features)
    return normalizer
