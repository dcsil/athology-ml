import random

import numpy as np
import tensorflow as tf
import typer
from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras import layers

SEED = 13370
NUMPY_SEED = 1337
TF_SEED = 133

random.seed(SEED)
np.random.seed(NUMPY_SEED)
tf.random.set_seed(TF_SEED)

BATCH_SIZE = 1
NUM_TIMESTEPS = 128
BUFFER_SIZE = 10000

METRICS = [
    #   keras.metrics.TruePositives(name='tp'),
    #   keras.metrics.FalsePositives(name='fp'),
    #   keras.metrics.TrueNegatives(name='tn'),
    #   keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]

app = typer.Typer()


class FeatureExtractor(layers.Layer):
    """Extracts additional features from our inputs, like sum, and norm."""

    def call(self, inputs):
        sum = tf.math.reduce_sum(inputs, axis=-1, keepdims=True, name="sum")
        norm = tf.norm(inputs, axis=-1, keepdims=True, name="norm")
        return tf.concat([inputs, sum, norm], axis=-1)


class JumpPredictor(HyperModel):
    def __init__(self, normalizer=None, classifier_bias_init=None):
        self._normalizer = normalizer
        self._classifier_bias_initializer = (
            tf.keras.initializers.Constant(value=classifier_bias_init)
            if classifier_bias_init
            else None
        )

    def build(self, hp):
        inputs = keras.Input(shape=(None, 3))
        x = FeatureExtractor()(inputs)
        if self._normalizer is not None:
            x = self._normalizer(x)
        x = layers.Conv1D(
            filters=hp.Choice("filters", values=[8, 16, 32], default=32),
            # TODO: Try a bigger kernel size!
            kernel_size=hp.Choice("kernel_size", values=[3, 5, 10], default=5),
            strides=1,
            padding="causal",
            activation="relu",
        )(x)

        x = layers.Bidirectional(
            layers.LSTM(
                hp.Choice("units_0", values=[32, 64, 128], default=128),
                return_sequences=True,
                recurrent_dropout=hp.Choice(
                    "recurrent_dropout_0", values=[0.0, 0.1, 0.25], default=0.1
                ),
            )
        )(x)

        x = layers.Bidirectional(
            layers.LSTM(
                hp.Choice("units_1", values=[32, 64, 128], default=128),
                return_sequences=True,
                recurrent_dropout=hp.Choice(
                    "recurrent_dropout_1", values=[0.0, 0.1, 0.25], default=0.1
                ),
            )
        )(x)

        x = layers.Dense(
            1, activation="sigmoid", bias_initializer=self._classifier_bias_initializer
        )(x)
        outputs = layers.Reshape((-1,))(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice(
                    "learning_rate", values=[5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5], default=1e-4
                )
            ),
            loss="binary_crossentropy",
            metrics=METRICS,
        )

        return model
