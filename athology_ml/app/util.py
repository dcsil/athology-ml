import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from athology_ml.ml.jump_detection.modules import FeatureExtractor
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


def salt_password(password: str, salt: Optional[bytes] = None, iterations: int = 100000):
    """Hashes passwords with pbkdf2_hmac. If `salt` is not provided, a new salt is generated.
    See: https://nitratine.net/blog/post/how-to-hash-passwords-in-python/ for details.
    """
    salt = os.urandom(32) if salt is None else salt
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return salt, key


def filter_predictions(predictions: List[bool], window_size: int = 25) -> List[bool]:
    predictions = predictions[:]
    for i in range(len(predictions) - 1):
        start = min(0, i - window_size)
        if predictions[i] and (not predictions[i + 1] or i == len(predictions) - 1):
            start = max(0, i - window_size)
            end = min(start + window_size, len(predictions))
            if sum(predictions[start:end]) < window_size:
                predictions[start:end] = [False] * len(predictions[start:end])
    return predictions
