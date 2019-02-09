"""A scoring strategy is a function which takes a list of floats and returns
the index of the one which should be considered most "optimal". For instance, if
we want to maximize, then we should return the argmax"""

import numpy as np

from .error_handling import InvalidStrategyException

__all__ = ["verify_scoring_strategy", "VALID_SCORING_STRATEGIES",
           "argmin_of_mean", "argmax_of_mean", "indexer_of_converter"]


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


class indexer_of_converter(object):

    def __init__(self, indexer, converter):
        """Constructs a function which first converts all objects in a list to
        something simpler and then uses the indexer to determine the index of 
        the most "optimal" one

        :param indexer: a function which converts a list of probably simply
            values (like numbers) to a single index
        :param converter: a function which converts a single more complex object
            to a simpler one (like a single number)
        """
        self.indexer = indexer
        self.converter = converter

    def __call__(self, scores):
        """Finds the index of the most "optimal" score in a list"""
        return self.indexer([self.converter(score) for score in scores])


argmin_of_mean = indexer_of_converter(np.argmin, np.mean)
argmax_of_mean = indexer_of_converter(np.argmax, np.mean)


VALID_SCORING_STRATEGIES = {
    'max': np.argmax,
    'maximize': np.argmax,
    'argmax': np.argmax,
    'min': np.argmin,
    'minimize': np.argmin,
    'argmin': np.argmin,
    'argmin_of_mean': argmin_of_mean,
    'argmax_of_mean': argmax_of_mean,
}
