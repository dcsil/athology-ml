import tensorflow as tf
from tensorflow.keras import layers


class FeatureExtractor(layers.Layer):
    """A custom Keras layer that extracts additional features from our inputs, like sum, and norm."""

    def call(self, inputs):
        sum = tf.math.reduce_sum(inputs, axis=-1, keepdims=True, name="sum")
        norm = tf.norm(inputs, axis=-1, keepdims=True, name="norm")
        return tf.concat([inputs, sum, norm], axis=-1)
