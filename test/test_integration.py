"""These are complete tests which hits multiple parts of the codebase"""

from sklearn.neural_network import MLPClassifier


from src.metrics import gerrity_score, peirce_skill_score, heidke_skill_score
from src.permutation_importance import permutation_importance
from src.sequential_selection import sequential_forward_selection, sequential_backward_selection
from src.sklearn_api import score_trained_sklearn_model, score_trained_sklearn_model_with_probabilities, score_untrained_sklearn_model, score_untrained_sklearn_model_with_probabilities
from test.utils import make_test_data, make_proba_test_data


def validate_result(result):
    print(result.retrieve_singlepass())
    singlepass = result.retrieve_singlepass().values()
    singlepass.sort(key=lambda x: x[0])
    last = 0
    for val in singlepass:
        print(val[1])
        assert last <= val[1]
        last = val[1]

    multipass = result.retrieve_multipass().values()
    assert len(singlepass) == len(multipass)


def test_deterministic():
    training_data, scoring_data = make_test_data()

    model = MLPClassifier(solver='lbfgs')

    # SFS
    scoring_fn = score_untrained_sklearn_model(model, peirce_skill_score)
    result = sequential_forward_selection(
        training_data, scoring_data, scoring_fn, "argmin")
    validate_result(result)

    # SBS
    scoring_fn = score_untrained_sklearn_model(model, gerrity_score)
    result = sequential_backward_selection(
        training_data, scoring_data, scoring_fn, "argmin")
    validate_result(result)

    # Permutation
    trained_model = model.fit(*training_data)
    scoring_fn = score_trained_sklearn_model(trained_model, heidke_skill_score)
    result = permutation_importance(
        scoring_data, scoring_fn, "argmin")
    validate_result(result)


def test_probabilistic():
    training_data, scoring_data = make_proba_test_data()
    print(scoring_data)

    model = MLPClassifier(solver='lbfgs')

    # SFS
    scoring_fn = score_untrained_sklearn_model_with_probabilities(
        model, peirce_skill_score)
    result = sequential_forward_selection(
        training_data, scoring_data, scoring_fn, "argmin")
    print(result.retrieve_singlepass())
    print(result.retrieve_multipass())
    validate_result(result)

    # SBS
    scoring_fn = score_untrained_sklearn_model_with_probabilities(
        model, peirce_skill_score)
    result = sequential_backward_selection(
        training_data, scoring_data, scoring_fn, "argmin")
    validate_result(result)

    # Permutation
    trained_model = model.fit(*training_data)
    scoring_fn = score_trained_sklearn_model_with_probabilities(
        trained_model, gerrity_score)
    result = permutation_importance(
        scoring_data, scoring_fn, "argmin")
    validate_result(result)
