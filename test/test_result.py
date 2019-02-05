
import pytest

from PermutationImportance.error_handling import FullImportanceResultWarning
from PermutationImportance.result import ImportanceResult


def test_result():
    method = "test"
    variable_names = ["A", "B", "C", "D"]

    imp_result = ImportanceResult(method, variable_names, 0)

    imp_result.add_new_results({
        "A": (0, 0.5),
        "B": (2, 0.7),
        "C": (1, 0.6),
        "D": (3, 0.8),
    }, "A")
    imp_result.add_new_results({
        "B": (0, 0.3),
        "C": (2, 0.5),
        "D": (1, 0.4),
    })
    imp_result.add_new_results({
        "C": (1, 0.3),
        "D": (0, 0.1),
    })
    assert not imp_result.complete
    assert len(imp_result) == 3
    imp_result.add_new_results({
        "C": (0, 0.0),
    })
    assert imp_result.original_score == 0
    assert imp_result.complete
    assert len(imp_result) == 5
    expected_breiman = {
        "A": (0, 0.5),
        "B": (2, 0.7),
        "C": (1, 0.6),
        "D": (3, 0.8),
    }
    assert expected_breiman == imp_result.retrieve_singlepass()
    expected_laks = {
        "A": (0, 0.5),
        "B": (1, 0.3),
        "C": (3, 0),
        "D": (2, 0.1),
    }
    assert expected_laks == imp_result.retrieve_multipass()
    expected_1 = ({
        "A": (0, 0.5),
    }, {
        "B": (0, 0.3),
        "C": (2, 0.5),
        "D": (1, 0.4),
    })
    assert expected_1 == imp_result[1]
    assert imp_result[4] == imp_result[-1]
    for context, result in imp_result:
        assert len(context) + len(result) == 4
    with pytest.warns(FullImportanceResultWarning):
        imp_result.add_new_results({"E": (0, 9000)})
