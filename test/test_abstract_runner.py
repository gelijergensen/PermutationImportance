
import numpy as np
import pandas as pd
import pytest

from PermutationImportance.abstract_runner import _singlethread_iteration, _multithread_iteration


def test__singlethread_iteration():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})

    def scoring_fn(training_data, scoring_data):
        return scoring_data[0].iloc[0, 1]
    selection_iterator = [(0, (inputs.iloc[:, [1, 0]], outputs), (inputs.iloc[:, [
                           1, 0]], outputs)), (2, (inputs.iloc[:, [1, 2]], outputs), (inputs.iloc[:, [1, 2]], outputs))]

    expected = {0: 1, 2: 3}
    assert expected == _singlethread_iteration(selection_iterator,
                                               scoring_fn)


def test__multithread_iteration():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})

    def scoring_fn(training_data, scoring_data):
        return scoring_data[0].iloc[0, 1]
    selection_iterator = [(0, (inputs.iloc[:, [1, 0]], outputs), (inputs.iloc[:, [
                           1, 0]], outputs)), (2, (inputs.iloc[:, [1, 2]], outputs), (inputs.iloc[:, [1, 2]], outputs))]

    expected = {0: 1, 2: 3}
    assert expected == _multithread_iteration(selection_iterator,
                                              scoring_fn, njobs=2)


# needs to run in 20 seconds or it probably hung in pool.join()
@pytest.mark.timeout(20)
def test__multithread_deadlock():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})

    def scoring_fn(training_data, scoring_data):
        return np.random.random((100000,))
    selection_iterator = ((i, inputs, outputs) for i in range(1000))

    _multithread_iteration(selection_iterator, scoring_fn, njobs=2)
    # If we get here at all, the test passes
    assert True
