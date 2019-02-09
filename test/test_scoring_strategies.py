import numpy as np
import pytest

from PermutationImportance.scoring_strategies import verify_scoring_strategy, VALID_SCORING_STRATEGIES, argmin_of_mean, indexer_of_converter
from PermutationImportance.error_handling import InvalidStrategyException


def test_valid_callable():
    assert np.argmin == verify_scoring_strategy(np.argmin)


def test_invalid_strategy():
    with pytest.raises(InvalidStrategyException):
        verify_scoring_strategy("asdfasdfa")


def test_valid_string_strategy():
    for key, value in VALID_SCORING_STRATEGIES.items():
        assert value == verify_scoring_strategy(key)


def test_composed():
    assert 2 == argmin_of_mean([np.array([1, 2]), np.array(
        [2, 4]), np.array([0, 1]), np.array([10, 12])])

    assert 3 == VALID_SCORING_STRATEGIES['argmax_of_mean']([np.array([1, 2]), np.array(
        [2, 4]), np.array([0, 1]), np.array([10, 12])])

    assert 2 == indexer_of_converter(np.argmin, np.mean)([np.array([1, 2]), np.array(
        [2, 4]), np.array([0, 1]), np.array([10, 12])])
