import kerastuner as kt
from athology_ml.ml.jump_detection import JumpDetector
from tensorflow.keras.layers.experimental import preprocessing


def test_can_build_model():
    """Sanity check that the model can be built with default hyperparameters."""
    hp = kt.HyperParameters()
    model = JumpDetector()
    _ = model.build(hp)


def test_can_build_model_with_normalizer():
    """Sanity check that the model can be built with default hyperparameters and a
    normalization layer."""
    hp = kt.HyperParameters()
    normalizer = preprocessing.Normalization()
    model = JumpDetector(normalizer=normalizer)
    _ = model.build(hp)
