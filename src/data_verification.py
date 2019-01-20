"""We can accept either a pandas dataframe with the target column indicated,
two different dataframes, or two numpy arrays"""

import numpy as np
import pandas as pd

from src.error_handling import InvalidDataException

__all__ = ["verify_data"]


def verify_data(data):
    """Verifies that the data tuple is of the right format and coerces it to
    numpy arrays for the code under the hood

    :param data: one of the following:
        (pandas dataframe, string for target column)
        (pandas dataframe for inputs, pandas dataframe for outputs)
        (numpy array for inputs, numpy array for outputs)
    :returns: (numpy array for input, numpy array for output) or 
        (pandas dataframe for input, pandas dataframe for output)
    """
    try:
        iter(data)
    except TypeError:
        raise InvalidDataException(data, "Data must be iterable")
    else:
        if len(data) != 2:
            raise InvalidDataException(data, "Data must contain 2 elements")
        else:
            # check if the first element is pandas dataframe or numpy array
            if isinstance(data[0], pd.DataFrame):
                # check if the second element is string or pandas dataframe
                if isinstance(data[1], basestring):
                    return data[0].loc[:, data[0].columns != data[1]], data[0][[data[1]]]
                elif isinstance(data[1], pd.DataFrame):
                    return data[0], data[1]
                else:
                    raise InvalidDataException(
                        data, "Second element of data must be a string for the target column or a pandas dataframe")

            elif isinstance(data[0], np.ndarray):
                if isinstance(data[1], np.ndarray):
                    return data[0], data[1]
                else:
                    raise InvalidDataException(
                        data, "Second element of data must also be a numpy array")
            else:
                raise InvalidDataException(
                    data, "First element of data must be a numpy array or pandas dataframe")
