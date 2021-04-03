from athology_ml.ml.jump_detection import util
import numpy as np


def test_set_seeds() -> None:
    # As it turns out, it is quite hard to determine the current random
    # seed. I was able to determine how for numpy, but not python or tensorflow.
    # See: https://stackoverflow.com/a/49749486/6578628
    util.set_seeds()
    assert np.random.get_state()[1][0] == util.NUMPY_SEED


def test_print_baselines_runs() -> None:
    """A sanity check that print_baselines runs."""
    labels = np.random.randint(2, size=(16, 128))
    util.print_baselines(labels)
