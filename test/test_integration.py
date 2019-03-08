"""These are complete tests which hits multiple parts of the codebase"""

from sklearn.neural_network import MLPClassifier


from PermutationImportance.metrics import gerrity_score, peirce_skill_score, heidke_skill_score
from PermutationImportance.permutation_importance import sklearn_permutation_importance
from PermutationImportance.sequential_selection import sklearn_sequential_forward_selection, sklearn_sequential_backward_selection
from test.utils import make_test_data, make_proba_test_data


def validate_result(result):
    singlepass = list(result.retrieve_singlepass().values())
    singlepass.sort(key=lambda x: x[0])
    last = singlepass[0][1]
    for val in singlepass:
        assert last <= val[1]
        last = val[1]

    multipass = list(result.retrieve_multipass().values())
    assert len(singlepass) == len(multipass)


def test_deterministic():
    training_data, scoring_data = make_test_data()

    model = MLPClassifier(solver='lbfgs')

    # SFS
    result = sklearn_sequential_forward_selection(
        model, training_data, scoring_data, peirce_skill_score, "argmin")
    validate_result(result)

    # SBS
    result = sklearn_sequential_backward_selection(
        model, training_data, scoring_data, gerrity_score, "argmin")
    validate_result(result)

    # Permutation
    trained_model = model.fit(*training_data)
    result = sklearn_permutation_importance(trained_model,
                                            scoring_data, heidke_skill_score, "argmin")
    validate_result(result)


def test_probabilistic():
    training_data, scoring_data = make_proba_test_data()

    model = MLPClassifier(solver='lbfgs')

    # SFS
    result = sklearn_sequential_forward_selection(
        model, training_data, scoring_data, heidke_skill_score, "argmin")
    validate_result(result)

    # SBS
    result = sklearn_sequential_backward_selection(
        model, training_data, scoring_data, peirce_skill_score, "argmin")
    validate_result(result)

    # Permutation
    trained_model = model.fit(*training_data)
    result = sklearn_permutation_importance(
        trained_model, scoring_data, gerrity_score, "argmin")
    validate_result(result)
