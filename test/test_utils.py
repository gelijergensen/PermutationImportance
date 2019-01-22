
import numpy as np
import pandas as pd
import pytest

from src.error_handling import InvalidDataException
from src.utils import convert_result_list_to_dict, get_data_subset


def test_convert_result_list_to_dict():
    result = [(10, 0.5), (9, 0.4), (4, 0.6)]
    variable_names = list(range(11))
    scoring_strategy = np.argmin
    expected = {
        9: (0, 0.4),
        10: (1, 0.5),
        4: (2, 0.6),
    }
    assert expected == convert_result_list_to_dict(
        result, variable_names, scoring_strategy)


def test_get_data_subset():
    data = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    columns = [1, 0]
    expected = np.array([[1, 0], [2, 1]])
    assert (expected == get_data_subset(data, columns)).all()

    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    data = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})
    expected = pd.DataFrame({'B': B, 'A': A}).loc[:, ['B', 'A']]
    assert expected.equals(get_data_subset(data, columns))

    data = [[0, 1, 2], [1, 2, 3]]
    with pytest.raises(InvalidDataException):
        get_data_subset(data, columns)
