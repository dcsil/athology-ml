import tensorflow as tf

from pathlib import Path
from athology_ml.ml.jump_detection.modules import FeatureExtractor
from functools import lru_cache
from tensorflow.keras import Model


PRETRAINED_MODELS = Path(__file__).parent.parent / "pretrained_models"


@lru_cache()
def load_jump_detection_model() -> Model:
    model = tf.keras.models.load_model(
        PRETRAINED_MODELS / "jump_detection.h5",
        custom_objects={"FeatureExtractor": FeatureExtractor()},
        compile=False,
    )
    return model
