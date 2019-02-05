"""A scoring strategy is a function which takes a list of floats and returns
the index of the one which should be considered most "optimal". For instance, if
we want to maximize, then we should return the argmax"""

import numpy as np

from src.error_handling import InvalidStrategyException

__all__ = ["verify_scoring_strategy", "VALID_SCORING_STRATEGIES"]


VALID_SCORING_STRATEGIES = {
    'max': np.argmax,
    'maximize': np.argmax,
    'argmax': np.argmax,
    'min': np.argmin,
    'minimize': np.argmin,
    'argmin': np.argmin,
}


def verify_scoring_strategy(scoring_strategy):
    """Asserts that the scoring strategy is valid and interprets various strings

    :param scoring_strategy: a function to be used for determining optimal
        variables or a string. If a function, should be of the form 
            ([floats]) -> index. If a string, must be one of the options in 
        VALID_SCORING_STRATEGIES
    :returns: a function to be used for determining optimal variables
    """
    if callable(scoring_strategy):
        return scoring_strategy
    elif scoring_strategy in VALID_SCORING_STRATEGIES:
        return VALID_SCORING_STRATEGIES[scoring_strategy]
    else:
        raise InvalidStrategyException(
            scoring_strategy, options=VALID_SCORING_STRATEGIES.keys())
