"""These are complete tests which hits multiple parts of the codebase"""

from sklearn.neural_network import MLPClassifier


from src.metrics import gerrity_score, peirce_skill_score
from src.sequential_selection import sequential_forward_selection, sequential_backward_selection
from src.sklearn_api import score_sklearn_model, score_sklearn_model_with_probabilities
from test.utils import make_test_data, make_proba_test_data


def test_deterministic():
    training_data, scoring_data = make_test_data()

    model = MLPClassifier(solver='lbfgs')

    scoring_fn = score_sklearn_model(model, peirce_skill_score)

    result = sequential_forward_selection(
        training_data, scoring_data, scoring_fn, "argmax")

    singlepass = result.retrieve_singlepass().values()
    singlepass.sort(key=lambda x: x[0])
    last = 1
    for val in singlepass:
        assert last >= val[1]
        last = val[1]

    multipass = result.retrieve_multipass().values()
    assert len(singlepass) == len(multipass)


def test_probabilistic():
    training_data, scoring_data = make_proba_test_data()

    model = MLPClassifier(solver='lbfgs')

    scoring_fn = score_sklearn_model_with_probabilities(
        model, peirce_skill_score)

    result = sequential_backward_selection(
        training_data, scoring_data, scoring_fn, "argmax")

    singlepass = result.retrieve_singlepass().values()
    singlepass.sort(key=lambda x: x[0])
    last = 1
    for val in singlepass:
        assert last >= val[1]
        last = val[1]

    multipass = result.retrieve_multipass().values()
    assert len(singlepass) == len(multipass)
