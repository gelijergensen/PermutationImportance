"""These tools are useful to assist the training and evaluation of sklearn 
models as components of a scoring function"""

import numpy as np
from sklearn.base import clone

from src.utils import get_data_subset


__all__ = ["model_scorer", "score_untrained_sklearn_model",
           "score_untrained_sklearn_model_with_probabilities",
           "score_trained_sklearn_model",
           "score_trained_sklearn_model_with_probabilities"]


def train_model(model, training_inputs, training_outputs):
    """Trains a scikit-learn model and returns the trained model"""
    if training_inputs.shape[1] == 0:
        # No data to train over, so don't bother
        return None
    cloned_model = clone(model)
    return cloned_model.fit(training_inputs, training_outputs)


def get_model(model, training_inputs, training_outputs):
    """Just return the trained model"""
    return model


def predict_model(model, scoring_inputs):
    """Uses a trained model to predict over the scoring data"""
    return model.predict(scoring_inputs)


def predict_proba_model(model, scoring_inputs):
    """Uses a trained model to predict class probabilities for the scoring data"""
    return model.predict_proba(scoring_inputs)


class model_scorer(object):
    """General purpose scoring method which trains a model, uses the model to
    predict, and evaluates the predictions with some metric
    """

    def __init__(self, model, training_fn, prediction_fn, evaluation_fn, default_score=0.0, nbootstrap=1, subsample=1):
        """Initializes the scoring object by storing the training, predicting,
        and evaluation functions

        :param model: a scikit-learn model
        :param training_fn: a function for training a scikit-learn model. Must 
            be of the form (model, training_inputs, training_outputs) -> 
                trained_model | None. If the function returns None, then it is
            assumed that the model training failed.
            Probably sklearn_api.train_model or sklearn_api.get_model
        :param predicting_fn: a function for predicting on scoring data using a 
            scikit-learn model. Must be of the form (model, scoring_inputs) -> 
                predictions. Predictions may be either deterministic or 
            probabilistic, depending on what the evaluation_fn accepts.
            Probably sklearn_api.predict_model or 
            sklearn_api.predict_proba_model
        :param evaluation_fn: a function which takes the deterministic or 
            probabilistic model predictions and scores them against the true 
            values. Must be of the form (truths, predictions) -> float
            Probably one of the metrics in src.metrics or sklearn.metrics
        :param default_score: value to return if the model cannot be trained
        :param nbootstrap: number of times to perform scoring on each variable.
            Results over different bootstrap iterations are averaged. Defaults to 1
        :param subsample: number of elements to sample (with replacement) per
            bootstrap round. If between 0 and 1, treated as a fraction of the number
            of total number of events (e.g. 0.5 means half the number of events).
            If not specified, subsampling will not be used and the entire data will
            be used (without replacement)
        """

        self.model = model
        self.training_fn = training_fn
        self.prediction_fn = prediction_fn
        self.evaluation_fn = evaluation_fn
        self.default_score = default_score
        self.nbootstrap = nbootstrap
        self.subsample = subsample

    def __call__(self, training_data, scoring_data):
        """Uses the training, predicting, and evaluation functions to score the
        model given the training and scoring data

        :param training_data: (training_input, training_output)
        :param scoring_data: (scoring_input, scoring_output)
        :returns: a float for the score
        """
        (training_inputs, training_outputs) = training_data
        (scoring_inputs, scoring_outputs) = scoring_data

        subsample = int(len(scoring_data[0]) *
                        self.subsample) if self.subsample <= 1 else self.subsample

        # Try to train model
        trained_model = self.training_fn(
            self.model, training_inputs, training_outputs)
        # If we didn't succeed in training (probably because there weren't any
        # training predictors), return the default_score
        if trained_model is None:
            return self.default_score

        # Score possibly multiple times
        scores = list()
        for _ in range(self.nbootstrap):
            if subsample == scoring_inputs.shape[0]:
                rows = np.arange(scoring_inputs.shape[0])
            else:
                rows = np.random.choice(
                    scoring_inputs.shape[0], subsample)
            subset_inputs = get_data_subset(scoring_inputs, rows)
            subset_outputs = get_data_subset(scoring_outputs, rows)

            predictions = self.prediction_fn(trained_model, subset_inputs)
            scores.append(self.evaluation_fn(predictions, subset_outputs))
        return np.average(scores)


def score_untrained_sklearn_model(model, evaluation_fn, nbootstrap=1, subsample=1):
    """A convenience method which uses the default training and the 
    deterministic prediction methods for scikit-learn to evaluate a model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or 
        probabilistic model predictions and scores them against the true 
        values. Must be of the form (truths, predictions) -> float
        Probably one of the metrics in src.metrics or sklearn.metrics
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :returns: a callable which accepts (training_data, scoring_data) and returns
        a float
    """
    return model_scorer(model, training_fn=train_model, prediction_fn=predict_model, evaluation_fn=evaluation_fn, nbootstrap=nbootstrap, subsample=subsample)


def score_untrained_sklearn_model_with_probabilities(model, evaluation_fn, nbootstrap=1, subsample=1):
    """A convenience method which uses the default training and the 
    probabilistic prediction methods for scikit-learn to evaluate a model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or 
        probabilistic model predictions and scores them against the true 
        values. Must be of the form (truths, predictions) -> float
        Probably one of the metrics in src.metrics or sklearn.metrics
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :returns: a callable which accepts (training_data, scoring_data) and returns
        a float
    """
    return model_scorer(model, training_fn=train_model, prediction_fn=predict_proba_model, evaluation_fn=evaluation_fn, nbootstrap=nbootstrap, subsample=subsample)


def score_trained_sklearn_model(model, evaluation_fn, nbootstrap=1, subsample=1):
    """A convenience method which uses the default training and the 
    deterministic prediction methods for scikit-learn to evaluate a model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or 
        probabilistic model predictions and scores them against the true 
        values. Must be of the form (truths, predictions) -> float
        Probably one of the metrics in src.metrics or sklearn.metrics
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :returns: a callable which accepts (training_data, scoring_data) and returns
        a float
    """
    return model_scorer(model, training_fn=get_model, prediction_fn=predict_model, evaluation_fn=evaluation_fn, nbootstrap=nbootstrap, subsample=subsample)


def score_trained_sklearn_model_with_probabilities(model, evaluation_fn, nbootstrap=1, subsample=1):
    """A convenience method which uses the default training and the 
    probabilistic prediction methods for scikit-learn to evaluate a model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or 
        probabilistic model predictions and scores them against the true 
        values. Must be of the form (truths, predictions) -> float
        Probably one of the metrics in src.metrics or sklearn.metrics
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :returns: a callable which accepts (training_data, scoring_data) and returns
        a float
    """
    return model_scorer(model, training_fn=get_model, prediction_fn=predict_proba_model, evaluation_fn=evaluation_fn, nbootstrap=nbootstrap, subsample=subsample)
