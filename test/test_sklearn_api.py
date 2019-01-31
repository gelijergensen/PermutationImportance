
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from src.sklearn_api import train_model, predict_model, predict_proba_model, model_scorer, score_sklearn_model, score_sklearn_model_with_probabilities
from test.utils import make_test_data, make_proba_test_data


def test_train_model():

    (training_inputs, training_outputs), (scoring_inputs,
                                          scoring_outputs) = make_test_data()
    model = SVC(gamma='auto')

    assert train_model(model, training_inputs,
                       training_outputs).fit_status_ == 0


def test_predict_model():

    (training_inputs, training_outputs), (scoring_inputs,
                                          scoring_outputs) = make_test_data()
    model = SVC(gamma='auto')

    assert predict_model(train_model(model, training_inputs, training_outputs),
                         scoring_inputs).shape == scoring_outputs.shape


def test_predict_proba_model():

    (training_inputs, training_outputs), (scoring_inputs,
                                          scoring_outputs) = make_proba_test_data()
    model = SVC(gamma='auto', probability=True)

    assert predict_proba_model(train_model(model, training_inputs, training_outputs),
                               scoring_inputs).shape == scoring_outputs.shape


def test_model_scorer():

    training_data, scoring_data = make_test_data()
    model = SVC(gamma='auto', probability=True)

    score_fn = model_scorer(model, train_model, predict_model, accuracy_score)

    assert callable(score_fn)

    score = score_fn(training_data, scoring_data)

    assert 0 <= score
    assert score <= 1


def test_score_sklearn_models():
    model = SVC(gamma='auto', probability=True)

    expected = score_sklearn_model(model, accuracy_score)
    result = model_scorer(model, train_model, predict_model, accuracy_score)
    for key in expected.__dict__:
        assert getattr(expected, key) == getattr(result, key)

    expected = score_sklearn_model_with_probabilities(model, accuracy_score)
    result = model_scorer(model, train_model,
                          predict_proba_model, accuracy_score)
    for key in expected.__dict__:
        assert getattr(expected, key) == getattr(result, key)
