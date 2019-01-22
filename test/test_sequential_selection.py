

import pandas as pd

from src.sequential_selection import _singlethread_iteration


def test__singlethread_iteration():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})

    training_data = (inputs, outputs)
    scoring_data = (inputs, outputs)

    def scoring_fn(training_data, scoring_data):
        return scoring_data[0].iloc[0, 1]
    selection_iterator = [(0, [1, 0]), (2, [1, 2])]

    expected = [(0, 1), (2, 3)]
    assert expected == _singlethread_iteration(training_data, scoring_data,
                                               scoring_fn, selection_iterator)
