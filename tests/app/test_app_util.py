import hypothesis.strategies as st
from athology_ml.app import util
from hypothesis import given
from tensorflow.keras import Model


def test_load_jump_detection_model():
    model = util.load_jump_detection_model()
    assert isinstance(model, Model)
    # This return value of this function is cached, so a subsequent
    # call should return the same object.
    assert model == util.load_jump_detection_model()


@given(expected_password=st.text())
def test_salt_password_without_salt(expected_password: str) -> None:
    actual_salt, actual_password = util.salt_password(password=expected_password)
    assert isinstance(actual_salt, bytes)
    assert isinstance(actual_password, bytes)


@given(expected_salt=st.binary(), expected_password=st.text())
def test_salt_password_with_salt(expected_salt: bytes, expected_password: str) -> None:
    actual_salt, actual_password = util.salt_password(
        salt=expected_salt, password=expected_password
    )
    assert actual_salt == expected_salt
    assert isinstance(actual_password, bytes)


def test_filter_predictions() -> None:
    window_size = 5
    predictions = [False, True, True, True, True, False]
    expected = [False, False, False, False, False, False]
    actual = util.filter_predictions(predictions, window_size=window_size)
    assert actual == expected

    predictions = [False, True, True, True, True, True]
    expected = [False, True, True, True, True, True]
    actual = util.filter_predictions(predictions, window_size=window_size)
    assert actual == expected

    predictions = [True, True, True, True, True, False]
    expected = [True, True, True, True, True, False]
    actual = util.filter_predictions(predictions, window_size=window_size)
    assert actual == expected
