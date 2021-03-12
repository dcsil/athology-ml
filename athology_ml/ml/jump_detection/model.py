import tensorflow as tf
import typer
from athology_ml.ml.jump_detection.modules import FeatureExtractor
from athology_ml.ml.jump_detection.util import set_seeds
from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

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

app = typer.Typer(callback=set_seeds)


class JumpPredictor(HyperModel):
    """A HyperModel that implements the jump detection model. Can be used with Keras Tuner for
    hyperparameter tuning. See the [Keras Tuner](https://keras-team.github.io/keras-tuner/) docs
    for more details.

    # Parameters

    normalizer : Normalization
        A normalization layer that will be used to normalize the input features to the model.
        It is expected that this layer has been initialized and adapted to the training data.

    classifier_bias_init : float
        A value to initialize the bias unit of the classification layer with. when the data is
        highly unbalanced, it can be useful to set this such that the model is heavily biased
        towards the most popular class during initialization. See
        [here](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias)
        for more details.
    """

    def __init__(
        self, normalizer: Normalization = None, classifier_bias_init: float = None
    ) -> None:
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
