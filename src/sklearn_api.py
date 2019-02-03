"""These tools are useful to assist the training and evaluation of sklearn 
models as components of a scoring function"""

from sklearn.base import clone


__all__ = ["model_scorer", "score_sklearn_model",
           "score_sklearn_model_with_probabilities"]


def train_model(model, training_inputs, training_outputs):
    """Trains a scikit-learn model and returns the trained model"""
    cloned_model = clone(model)
    return cloned_model.fit(training_inputs, training_outputs)


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

    def __init__(self, model, training_fn, prediction_fn, evaluation_fn):
        """Initializes the scoring object by storing the training, predicting,
        and evaluation functions

        :param model: a scikit-learn model
        :param training_fn: a function for training a scikit-learn model. Must 
            be of the form (model, training_inputs, training_outputs) -> 
                trained_model
            Probably sklearn_api.train_model
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
        """
        self.model = model
        self.training_fn = training_fn
        self.prediction_fn = prediction_fn
        self.evaluation_fn = evaluation_fn

    def __call__(self, training_data, scoring_data):
        """Uses the training, predicting, and evaluation functions to score the
        model given the training and scoring data

        :param training_data: (training_input, training_output)
        :param scoring_data: (scoring_input, scoring_output)
        :returns: a float for the score
        """
        (training_inputs, training_outputs) = training_data
        (scoring_inputs, scoring_outputs) = scoring_data
        trained_model = self.training_fn(
            self.model, training_inputs, training_outputs)
        predictions = self.prediction_fn(trained_model, scoring_inputs)
        return self.evaluation_fn(predictions, scoring_outputs)


def score_sklearn_model(model, evaluation_fn):
    """A convenience method which uses the default training and the 
    deterministic prediction methods for scikit-learn to evaluate a model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or 
        probabilistic model predictions and scores them against the true 
        values. Must be of the form (truths, predictions) -> float
        Probably one of the metrics in src.metrics or sklearn.metrics
    :returns: a callable which accepts (training_data, scoring_data) and returns
        a float
    """
    return model_scorer(model, training_fn=train_model, prediction_fn=predict_model, evaluation_fn=evaluation_fn)


def score_sklearn_model_with_probabilities(model, evaluation_fn):
    """A convenience method which uses the default training and the 
    probabilistic prediction methods for scikit-learn to evaluate a model

    :param model: a scikit-learn model
    :param evaluation_fn: a function which takes the deterministic or 
        probabilistic model predictions and scores them against the true 
        values. Must be of the form (truths, predictions) -> float
        Probably one of the metrics in src.metrics or sklearn.metrics
    :returns: a callable which accepts (training_data, scoring_data) and returns
        a float
    """
    return model_scorer(model, training_fn=train_model, prediction_fn=predict_proba_model, evaluation_fn=evaluation_fn)
