
import numpy as np
import pandas as pd
import pytest

from PermutationImportance.error_handling import InvalidDataException
from PermutationImportance.utils import add_ranks_to_dict, get_data_subset, make_data_from_columns


def test_add_ranks_to_dict():
    result = {10: 0.5, 9: 0.4, 4: 0.6}
    variable_names = list(range(11))
    scoring_strategy = np.argmin
    expected = {
        9: (0, 0.4),
        10: (1, 0.5),
        4: (2, 0.6),
    }
    assert expected == add_ranks_to_dict(
        result, variable_names, scoring_strategy)


def test_get_data_subset():
    data = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    rows = np.arange(data.shape[0])
    columns = [1, 0]
    expected = np.array([[1, 0], [2, 1]])
    assert (expected == get_data_subset(data, rows, columns)).all()
    assert (expected == get_data_subset(data, None, columns)).all()
    expected = data
    assert (expected == get_data_subset(data, rows)).all()

    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    data = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})
    expected = pd.DataFrame({'B': B, 'A': A}).loc[:, ['B', 'A']]
    assert expected.equals(get_data_subset(
        data, rows, columns))
    assert expected.equals(get_data_subset(data, None, columns))
    expected = data
    assert expected.equals(get_data_subset(data, rows))

    data = [[0, 1, 2], [1, 2, 3]]
    with pytest.raises(InvalidDataException):
        get_data_subset(data, rows, columns)


def test_make_data_from_columns():
    columns = [np.array([0, 1, 2]), np.array([10, 11, 12])]
    expected = np.array([[0, 10], [1, 11], [2, 12]])
    assert (expected == make_data_from_columns(columns)).all()

    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    data = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})
    get_data_subset(data, None, [0])
    columns = [get_data_subset(data, None, [0]), get_data_subset(
        data, None, [2]), get_data_subset(data, None, [3])]
    expected = data[['A', 'C', 'D']]
    assert expected.equals(make_data_from_columns(columns))

    assert (np.array([]) == make_data_from_columns([np.array([])])).all()
