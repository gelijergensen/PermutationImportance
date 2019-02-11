

import pandas as pd

from PermutationImportance.result import ImportanceResult
from PermutationImportance.sequential_selection import sequential_forward_selection, sequential_backward_selection


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
        if len(training_data[0].columns) == 0:
            return 0
        if 'A' in training_data[0].columns:
            return scoring_data[0].iloc[1, -1]
        else:
            return scoring_data[0].iloc[1, -1] / 2

    expected = ImportanceResult(
        "Sequential Forward Selection", ["A", "B", "C"], 6)
    expected.add_new_results({'A': (0, 2), 'B': (1, 2), 'C': (2, 3)})
    expected.add_new_results({'B': (0, 4), 'C': (1, 6)})
    expected.add_new_results({'C': (0, 6)})

    result = sequential_forward_selection(
        training_data, scoring_data, scoring_fn, "argmin", njobs=2)

    assert expected.method == result.method
    assert expected.original_score == result.original_score
    assert (expected.variable_names == result.variable_names).all()
    assert expected.retrieve_singlepass() == result.retrieve_singlepass()
    assert expected.retrieve_multipass() == result.retrieve_multipass()
    for (exp_context, exp_result), (true_context, true_result) in zip(expected, result):
        assert exp_context == true_context
        assert exp_result == true_result


def test_sequential_backward_selection():
    A = [1, 2]
    B = [2, 4]
    C = [3, 6]
    D = [1, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})

    training_data = (inputs, outputs)
    scoring_data = (inputs, outputs)

    def scoring_fn(training_data, scoring_data):
        if len(training_data[0].columns) == 0:
            return 0
        elif 'A' not in training_data[0].columns:
            return scoring_data[0].iloc[1, 0]
        else:
            return scoring_data[0].iloc[1, 0] / 2

    expected = ImportanceResult(
        "Sequential Backward Selection", ["A", "B", "C"], 1)
    expected.add_new_results({'A': (2, 4), 'B': (0, 1), 'C': (1, 1)})
    expected.add_new_results({'A': (1, 6), 'C': (0, 1)})
    expected.add_new_results({'A': (0, 0)})

    result = sequential_backward_selection(
        training_data, scoring_data, scoring_fn, "argmin", njobs=2)

    assert expected.method == result.method
    assert expected.original_score == result.original_score
    assert (expected.variable_names == result.variable_names).all()
    assert expected.retrieve_singlepass() == result.retrieve_singlepass()
    assert expected.retrieve_multipass() == result.retrieve_multipass()
    for (exp_context, exp_result), (true_context, true_result) in zip(expected, result):
        assert exp_context == true_context
        assert exp_result == true_result
