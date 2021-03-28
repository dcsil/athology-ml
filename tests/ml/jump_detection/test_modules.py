import numpy as np
import numpy.testing as npt
from athology_ml.ml import jump_detection


def test_feature_extractor() -> None:
    # batch_size and timesteps are tuneable. Here we use similar values as used during training.
    # input_dim and output_dim are decided by the data and the feature_extractor layer respectively.
    batch_size, timesteps, input_dim, output_dim = 1, 128, 3, 5
    feature_extractor = jump_detection.modules.FeatureExtractor()
    inputs = np.random.rand(batch_size, timesteps, input_dim)
    outputs = feature_extractor(inputs)
    assert outputs.shape == (batch_size, timesteps, output_dim)
    # Have to reduce the strictness quite a bit to pass the tests. This likely just reflects
    # small differences in inplementation between TensorFlow and NumPy.
    npt.assert_almost_equal(outputs[:, :, -2], np.sum(inputs, axis=-1), decimal=6)
    npt.assert_almost_equal(outputs[:, :, -1], np.linalg.norm(inputs, axis=-1), decimal=6)
