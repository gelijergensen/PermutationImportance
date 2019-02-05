

import numpy as np
import pytest


from PermutationImportance.metrics import gerrity_score, heidke_skill_score, peirce_skill_score, _get_contingency_table
from PermutationImportance.error_handling import AmbiguousProbabilisticForecastsException, UnmatchingProbabilisticForecastsException, UnmatchedLengthPredictionsException


def test__get_contingency_table():
    truths = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"])
    predictions = np.array(["a", "b", "c", "a", "c", "b", "a", "b", "b"])

    expected = np.array([[3, 0, 0], [0, 2, 2], [0, 1, 1]])
    assert (expected == _get_contingency_table(truths, predictions)).all()

    classes = ["a", "b", "c", "d"]
    expected = np.array([[3, 0, 0, 0], [0, 2, 2, 0],
                         [0, 1, 1, 0], [0, 0, 0, 0]])
    assert (expected == _get_contingency_table(
        truths, predictions, classes)).all()

    classes = ["a", "b", "c"]
    predictions = np.array([[1, 0, 0], [0, 0.9, 0.1], [0.05, 0.05, 0.9], [1, 0, 0], [0.05, 0.05, 0.9], [
                           0, 0.9, 0.1], [1, 0, 0], [0, 0.9, 0.1], [0, 0.9, 0.1]])
    expected = np.array([[3, 0, 0], [0, 2, 2], [0, 1, 1]])
    assert (expected == _get_contingency_table(
        truths, predictions, classes)).all()

    with pytest.raises(AmbiguousProbabilisticForecastsException):
        _get_contingency_table(truths, predictions)

    truths = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [
                      0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert (expected == _get_contingency_table(truths, predictions)).all()

    predictions = np.zeros((9,))
    with pytest.raises(UnmatchingProbabilisticForecastsException):
        _get_contingency_table(truths, predictions)

    predictions = np.zeros((9, 4))
    with pytest.raises(UnmatchingProbabilisticForecastsException):
        _get_contingency_table(truths, predictions)

    truths = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"])
    predictions = np.array(["a", "b", "c", "a", "c"])
    with pytest.raises(UnmatchedLengthPredictionsException):
        _get_contingency_table(truths, predictions)


def test_gerrity_score():
    truths = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"])
    predictions = np.array(["a", "a", "a", "a", "a", "a", "a", "a", "a"])

    assert gerrity_score(truths, predictions) < 1e-6
    assert abs(1 - gerrity_score(truths, truths)) < 1e-6

    predictions = np.array(["a", "b", "c", "a", "b", "c", "a", "a", "a"])
    assert 0.5 < gerrity_score(truths, predictions) < 1


def test_heidke_skill_score():
    truths = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"])
    predictions = np.array(["a", "a", "a", "a", "a", "a", "a", "a", "a"])

    assert 0 == heidke_skill_score(truths, predictions)
    assert 1 == heidke_skill_score(truths, truths)

    predictions = np.array(["a", "b", "c", "a", "b", "c", "a", "a", "a"])
    assert 0.5 < heidke_skill_score(truths, predictions) < 1


def test_peirce_skill_score():
    truths = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"])
    predictions = np.array(["a", "a", "a", "a", "a", "a", "a", "a", "a"])

    assert 0 == peirce_skill_score(truths, predictions)
    assert 1 == peirce_skill_score(truths, truths)

    predictions = np.array(["a", "b", "c", "a", "b", "c", "a", "a", "a"])
    assert 0.5 < peirce_skill_score(truths, predictions) < 1
