"""Permutation Importance iteratively adds variables to the set of important 
variables by permuting the columns of variables considered important and 
evaluating a model on the partially permuted data"""

import numpy as np

from src.abstract_runner import abstract_variable_importance
from src.selection_strategies import PermutationImportanceSelectionStrategy
from src.sklearn_api import score_trained_sklearn_model, score_trained_sklearn_model_with_probabilities

__all__ = ["permutation_importance", "sklearn_permutation_importance"]


def permutation_importance(scoring_data, scoring_fn, scoring_strategy, variable_names=None, nimportant_vars=None, njobs=1):
    """Performs permutation importance over data given a particular
    set of functions for scoring and determining optimal variables

    :param scoring_data: a 2-tuple(inputs, outputs) for scoring in the
        scoring_fn
    :param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float, but should only use the 
        scoring_data to produce a score
    :param scoring_strategy: a function to be used for determining optimal
        variables. Should be of the form([floats]) -> index
    :param variable_names: an optional list for variable names. If not given,
        will use names of columns of data(if pandas dataframe) or column
        indices
    :param nimportant_vars: number of times to compute the next most important
        variable. Defaults to all
    :param njobs: an integer for the number of threads to use. If negative, will
        use the number of cpus + njobs. Defaults to 1
    :returns: ImportanceResult object which contains the results for each run
    """
    # We don't need the training data, so pass empty arrays to the abstract runner
    return abstract_variable_importance((np.array([]), np.array([])), scoring_data, scoring_fn, scoring_strategy, PermutationImportanceSelectionStrategy, variable_names=variable_names, nimportant_vars=nimportant_vars, njobs=njobs)


def sklearn_permutation_importance(model, scoring_data, evaluation_fn, scoring_strategy, variable_names=None, nimportant_vars=None, njobs=1, nbootstrap=1, subsample=1):
    """Performs permutation importance for a particular model, 
    scoring_data, evaluation_fn, and strategy for determining optimal variables

    :param model: a trained sklearn model
    :param scoring_data: a 2-tuple(inputs, outputs) for scoring in the
        scoring_fn
    :param evaluation_fn: a function which takes the deterministic or 
        probabilistic model predictions and scores them against the true 
        values. Must be of the form (truths, predictions) -> float
        Probably one of the metrics in src.metrics or sklearn.metrics
    :param scoring_strategy: a function to be used for determining optimal
        variables. Should be of the form([floats]) -> index
    :param variable_names: an optional list for variable names. If not given,
        will use names of columns of data(if pandas dataframe) or column
        indices
    :param nimportant_vars: number of times to compute the next most important
        variable. Defaults to all
    :param njobs: an integer for the number of threads to use. If negative, will
        use the number of cpus + njobs. Defaults to 1
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :returns: ImportanceResult object which contains the results for each run
    """
    # Check if the data is probabilistic
    if len(scoring_data[1].shape) > 1 and scoring_data[1].shape[1] > 1:
        scoring_fn = score_trained_sklearn_model_with_probabilities(
            model, evaluation_fn, nbootstrap=nbootstrap, subsample=subsample)
    else:
        scoring_fn = score_trained_sklearn_model(
            model, evaluation_fn, nbootstrap=nbootstrap, subsample=subsample)
    return permutation_importance(scoring_data, scoring_fn, scoring_strategy, variable_names=variable_names, nimportant_vars=nimportant_vars, njobs=njobs)
