
import pandas as pd

from PermutationImportance.result import ImportanceResult
from PermutationImportance.permutation_importance import permutation_importance


def test_permutation_importance():
    A = [1, 0, 0]
    B = [0, 1, 0]
    C = [0, 0, 1]
    D = [0, 0, 0]
    inputs = pd.DataFrame({'A': A, 'B': B, 'C': C})
    outputs = pd.DataFrame({'D': D})

    scoring_data = (inputs, outputs)

    # This score is degenerate because I can't really guarantee any shuffling
    def scoring_fn(training_data, scoring_data):
        return scoring_data[0].values.sum()

    expected = ImportanceResult(
        "Permutation Importance", ["A", "B", "C"], 3)
    expected.add_new_results(
        {'A': (0, 3), 'B': (1, 3), 'C': (2, 3)})
    expected.add_new_results({'B': (0, 3), 'C': (1, 3)})
    expected.add_new_results({'C': (0, 3)})

    result = permutation_importance(
        scoring_data, scoring_fn, "argmin", njobs=2)

    assert expected.method == result.method
    assert expected.original_score == result.original_score
    assert (expected.variable_names == result.variable_names).all()
    assert expected.retrieve_singlepass() == result.retrieve_singlepass()
    assert expected.retrieve_multipass() == result.retrieve_multipass()
    for (exp_context, exp_result), (true_context, true_result) in zip(expected, result):
        assert exp_context == true_context
        assert exp_result == true_result
