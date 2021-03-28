import random

import numpy as np
import tensorflow as tf
import typer

SEED = 13370
NUMPY_SEED = 1337
TF_SEED = 133
KERAS_TUNER_SEED = 13


def set_seeds():
    """Sets the random seeds of python, numpy and tensorflow for reproducible experiements."""
    random.seed(SEED)
    np.random.seed(NUMPY_SEED)
    tf.random.set_seed(TF_SEED)


def print_baselines(labels: np.ndarray) -> None:
    """Prints the precision and recall values of a few simple baselines on the given `labels`."""
    random_baseline_precision = tf.keras.metrics.Precision()
    random_baseline_recall = tf.keras.metrics.Recall()
    never_jumping_precision = tf.keras.metrics.Precision()
    never_jumping_recall = tf.keras.metrics.Recall()
    always_jumping_precision = tf.keras.metrics.Precision()
    always_jumping_recall = tf.keras.metrics.Recall()

    # Compute baseline precision and recalls to compare against
    random_prediction = np.random.randint(2, size=labels.shape)
    all_zeros_prediction = np.zeros_like(labels)
    all_ones_prediction = np.ones_like(labels)

    random_baseline_precision.update_state(labels, random_prediction)
    random_baseline_recall.update_state(labels, random_prediction)
    never_jumping_precision.update_state(labels, all_zeros_prediction)
    never_jumping_recall.update_state(labels, all_zeros_prediction)
    always_jumping_precision.update_state(labels, all_ones_prediction)
    always_jumping_recall.update_state(labels, all_ones_prediction)

    typer.secho("\nA random baseline (predict 0/1 at random, uniformly) achieves:", bold=True)
    typer.secho(f"  * precision: {random_baseline_precision.result().numpy():.4f}")
    typer.secho(f"  * recall:    {random_baseline_recall.result().numpy():.4f}")
    typer.secho("A never jumping baseline (always predict 0) achieves:", bold=True)
    typer.secho(f"  * precision: {never_jumping_precision.result().numpy():.4f}")
    typer.secho(f"  * recall:    {never_jumping_recall.result().numpy():.4f}")
    typer.secho("An always jumping baseline (always predict 1) achieves:", bold=True)
    typer.secho(f"  * precision: {always_jumping_precision.result().numpy():.4f}")
    typer.secho(f"  * recall:    {always_jumping_recall.result().numpy():.4f}\n")
