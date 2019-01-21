
import numpy as np
import pandas as pd
import pytest

from src.data_verification import verify_data, determine_variable_names
from src.error_handling import InvalidDataException, InvalidInputException


def test_numpy_arrays():
    inputs = np.array([[1, 2, 3], [2, 4, 6]])
    outputs = np.array([1, 0])
    data = (inputs, outputs)
    result = verify_data(data)
    for exp, res in zip(data, result):
        assert (exp == res).all()


def test_pandas_dataframes():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})
    data = (inputs, outputs)
    result = verify_data(data)
    for exp, res in zip(data, result):
        assert exp.equals(res)


def test_pandas_with_string():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})
    outputs = "D"
    data = (inputs, outputs)
    expected_inputs = inputs[['A', 'B', 'C']]
    expected_outputs = inputs[['D']]
    expected = (expected_inputs, expected_outputs)
    result = verify_data(data)
    for exp, res in zip(expected, result):
        assert exp.equals(res)


def test_invalid_numpy_string():
    inputs = np.array([[1, 2, 3], [2, 4, 6]])
    outputs = "Hi"
    data = (inputs, outputs)
    with pytest.raises(InvalidDataException):
        verify_data(data)


def test_variable_names():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})
    data = (inputs, outputs)
    variable_names = np.array(["Fred", "George", "Bob"])
    assert (variable_names == determine_variable_names(
        data, variable_names)).all()


def test_pandas_variable_names():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})
    data = (inputs, outputs)
    variable_names = np.array(["A", "B", "C"])
    assert (variable_names == determine_variable_names(
        data, variable_names=None)).all()


def test_range_variable_names():
    inputs = np.array([[1, 2, 3], [2, 4, 6]])
    outputs = np.array([1, 0])
    data = (inputs, outputs)
    variable_names = np.array([0, 1, 2])
    assert (variable_names == determine_variable_names(
        data, variable_names=None)).all()


def test_invalid_variable_names():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})
    data = (inputs, outputs)
    variable_names = ["A", "B", "C", "D"]
    with pytest.raises(InvalidInputException):
        determine_variable_names(data, variable_names)
