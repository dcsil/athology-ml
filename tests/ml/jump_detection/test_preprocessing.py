from typing import Tuple

import hypothesis.strategies as st
import numpy as np
import pytest
from athology_ml.ml.jump_detection import preprocessing
from hypothesis import given
from hypothesis.extra import numpy
from tensorflow.data import Dataset
from tensorflow.keras.layers.experimental.preprocessing import Normalization


def test_pack_features_vector_correct_shape(
    dummy_jump_detection_dataset: Tuple[Dataset, Dataset, Dataset]
) -> None:
    """Sanity check that feature and label vector packing produce tensors of the correct shape."""
    train_dataset, _, _ = dummy_jump_detection_dataset
    train_dataset = train_dataset.map(preprocessing.pack_features_vector)
    num_features = 3
    for features, labels in train_dataset.as_numpy_iterator():
        assert features.shape == (num_features,)
        assert labels.shape == (1,)


def test_get_features_and_labels(
    dummy_jump_detection_dataset: Tuple[Dataset, Dataset, Dataset]
) -> None:
    """Sanity check that get_features_and_labels returns tensors of the correct shape."""
    train_dataset, _, _ = dummy_jump_detection_dataset
    # get_features_and_labels expects pack_features_vector to have been called
    train_dataset = train_dataset.map(preprocessing.pack_features_vector)
    num_rows, num_features = 100, 3
    all_features, all_labels = preprocessing.get_features_and_labels(
        dataset=train_dataset, num_rows=num_rows
    )
    assert all_features.shape == (num_rows, num_features)
    assert all_labels.shape == (num_rows,)


@given(
    numpy.arrays(
        np.dtype("bool"),
        shape=st.tuples(
            st.integers(min_value=0, max_value=16), st.integers(min_value=0, max_value=16)
        ),
    )
)
def test_get_classifier_bias_init(labels: np.ndarray) -> None:
    """Tests get_classifier_bias_init over a grid of random, two-dimensional bool arrays."""
    pos = labels.sum()
    neg = labels.size - pos

    # Assert that a ValueError is raised if labels is an empty array.
    if not labels.size:
        with pytest.raises(ValueError):
            _ = preprocessing.get_classifier_bias_init(labels)
    else:
        actual = preprocessing.get_classifier_bias_init(labels)
        # If there is at least one positive and one negative, we should be returning
        # the log ratio of the support. Otherwise we should be returning None.
        if pos > 0 and neg > 0:
            expected = np.log(pos / neg)
        else:
            expected = None

        assert actual == expected


def test_get_normalizer() -> None:
    """Sanity check that the normalization layer can be created with some valid training data."""
    # batch_size and timesteps are tuneable. Here we use similar values as used during training.
    # input_dim and output_dim are decided by the data and the feature_extractor layer respectively.
    batch_size, timesteps, input_dim, output_dim = 1, 128, 3, 5

    features = np.random.rand(batch_size, timesteps, input_dim)
    sum_ = np.sum(features, axis=-1, keepdims=True)
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    inputs = np.concatenate((features, sum_, norm), axis=-1)

    normalizer = preprocessing.get_normalizer(features)
    assert isinstance(normalizer, Normalization)
    outputs = normalizer(inputs).numpy()
    assert outputs.shape == (batch_size, timesteps, output_dim)


def test_get_normalizer_value_error() -> None:
    """Check that the get_normalizer will return a ValueError for bad data."""
    # batch_size and timesteps are tuneable. Here we use similar values as used during training.
    # input_dim and output_dim are decided by the data and the feature_extractor layer respectively.
    batch_size, timesteps, input_dim = 1, 128, 3

    features = np.random.rand(0, timesteps, input_dim)
    with pytest.raises(ValueError):
        _ = preprocessing.get_normalizer(features)

    features = np.random.rand(batch_size, 0, input_dim)
    with pytest.raises(ValueError):
        _ = preprocessing.get_normalizer(features)
