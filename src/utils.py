"""Various and sundry useful functions which handy for different types of
variable importance"""

import numpy as np
import pandas as pd

from src.error_handling import InvalidDataException

__all__ = ["add_ranks_to_dict", "get_data_subset"]


def add_ranks_to_dict(result, variable_names, scoring_strategy):
    """Takes a list of (var, score) and converts to a dictionary

    :param result: a dict of {var_index: score}
    :param variable_names: a list of variable names
    :param scoring_strategy: a function to be used for determining optimal
        variables. Should be of the form ([floats]) -> index
    """
    if len(result) == 0:
        return dict()

    result_dict = dict()
    rank = 0
    while len(result) > 1:
        best_var = result.keys()[scoring_strategy(result.values())]
        score = result.pop(best_var)
        result_dict[variable_names[best_var]] = (rank, score)
        rank += 1
    var, score = result.items()[0]
    result_dict[variable_names[var]] = (rank, score)
    return result_dict


def get_data_subset(data, rows, columns=None):
    """Returns a subset of the data corresponding to the desired columns

    :param data: either a pandas dataframe or a numpy array
    :param rows: a list of row indices
    :param columns: a list of column indices
    :returns: data_subset (same type as data)
    """
    if isinstance(data, pd.DataFrame):
        if columns is None:
            return data.iloc[rows]
        else:
            return data.iloc[rows, columns]
    elif isinstance(data, np.ndarray):
        if columns is None:
            return data[rows]
        else:
            return data[np.ix_(rows, columns)]
    else:
        raise InvalidDataException(
            data, "Data must be a pandas dataframe or numpy array")
