
import numpy as np
import pandas as pd
import pytest

from PermutationImportance.data_verification import verify_data, determine_variable_names
from PermutationImportance.error_handling import InvalidDataException, InvalidInputException


def test_pandas_dataframes():
    inputs = np.array([[1, 2, 3], [2, 4, 6]])
    outputs = np.array([1, 0])
    data = (inputs, outputs)
    result = verify_data(data)
    for exp, res in zip(data, result):
        assert (exp == res).all()

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

    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})
    outputs = "D"
    data = (inputs, outputs)
    expected_inputs = inputs[['A', 'B', 'C']]
    expected_outputs = inputs[['D']]
    expected = (expected_inputs, expected_outputs)
    result = verify_data(data)
    for exp, res in zip(expected, result):
        assert exp.equals(res)

    outputs = pd.DataFrame({'D': D}).values
    data = (inputs, outputs)
    with pytest.raises(InvalidDataException):
        verify_data(data)

    inputs = np.array([[1, 2, 3], [2, 4, 6]])
    outputs = "Hi"
    data = (inputs, outputs)
    with pytest.raises(InvalidDataException):
        verify_data(data)

    data = "Hi"
    with pytest.raises(InvalidDataException):
        verify_data(data)

    inputs = "Hi"
    outputs = "Bye"
    data = (inputs, outputs)
    with pytest.raises(InvalidDataException):
        verify_data(data)

    inputs = np.array([[1, 2, 3], [2, 4, 6]])
    outputs = np.array([1, 0])
    other = np.array(["A", "B"])
    data = (inputs, outputs, other)
    with pytest.raises(InvalidDataException):
        verify_data(data)


def test_variable_names():
    inputs = np.array([[1, 2, 3], [2, 4, 6]])
    outputs = np.array([1, 0])
    data = (inputs, outputs)
    variable_names = np.array([0, 1, 2])
    assert (variable_names == determine_variable_names(
        data, variable_names=None)).all()

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

    variable_names = np.array(["A", "B", "C"])
    assert (variable_names == determine_variable_names(
        data, variable_names=None)).all()

    variable_names = ["A", "B", "C", "D"]
    with pytest.raises(InvalidInputException):
        determine_variable_names(data, variable_names)

    variable_names = 0
    with pytest.raises(InvalidInputException):
        determine_variable_names(data, variable_names)
