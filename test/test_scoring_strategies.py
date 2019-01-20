import numpy as np
import pytest

from src.scoring_strategies import verify_scoring_strategy, VALID_SCORING_STRATEGIES
from src.error_handling import InvalidStrategyException


def test_valid_callable():
    assert np.argmin == verify_scoring_strategy(np.argmin)


def test_invalid_strategy():
    with pytest.raises(InvalidStrategyException):
        verify_scoring_strategy("asdfasdfa")


def test_valid_string_strategy():
    for key, value in VALID_SCORING_STRATEGIES.items():
        assert value == verify_scoring_strategy(key)
