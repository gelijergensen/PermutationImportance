

import pandas as pd

from src.result import ImportanceResult
from src.sequential_selection import _singlethread_iteration, sequential_forward_selection


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


def test_sequential_forward_selection():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})

    training_data = (inputs, outputs)
    scoring_data = (inputs, outputs)

    def scoring_fn(training_data, scoring_data):
        if 'A' in training_data[0].columns:
            return scoring_data[0].iloc[1, -1]
        else:
            return scoring_data[0].iloc[1, -1] / 2

    expected = ImportanceResult(
        "Sequential Forward Selection", ["A", "B", "C"])
    expected.add_new_results({'A': (0, 2), 'B': (1, 2), 'C': (2, 3)})
    expected.add_new_results({'B': (0, 4), 'C': (1, 6)})
    expected.add_new_results({'C': (0, 6)})

    result = sequential_forward_selection(
        training_data, scoring_data, scoring_fn, "argmin", nbootstrap=2)

    assert expected.method == result.method
    assert (expected.variable_names == result.variable_names).all()
    assert expected.retrieve_breiman() == result.retrieve_breiman()
    assert expected.retrieve_laks() == result.retrieve_laks()
    for (exp_context, exp_result), (true_context, true_result) in zip(expected, result):
        assert exp_context == true_context
        assert exp_result == true_result
