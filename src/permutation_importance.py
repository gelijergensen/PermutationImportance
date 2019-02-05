"""Permutation Importance iteratively adds variables to the set of important 
variables by permuting the columns of variables considered important and 
evaluating a model on the partially permuted data"""

import numpy as np

from src.abstract_runner import abstract_variable_importance
from src.selection_strategies import PermutationImportanceSelectionStrategy

__all__ = ["permuation_importance"]


def permuation_importance(scoring_data, scoring_fn, scoring_strategy, variable_names=None, nimportant_vars=None, nbootstrap=1, subsample=1, njobs=1):
    """Performs sequential forward selection over data given a particular
    set of functions for scoring and determining optimal variables

    : param scoring_data: a 2-tuple(inputs, outputs) for scoring in the
        scoring_fn
    : param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float, but should only use the 
        scoring_data to produce a score
    : param scoring_strategy: a function to be used for determining optimal
        variables. Should be of the form([floats]) -> index
    : param variable_names: an optional list for variable names. If not given,
        will use names of columns of data(if pandas dataframe) or column
        indices
    : param nimportant_vars: number of times to compute the next most important
        variable. Defaults to all
    : param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    : param subsample: number of elements to sample(with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events(e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used(without replacement)
    : param njobs: an integer for the number of threads to use. If negative, will
        use the number of cpus + njobs. Defaults to 1
    : returns: ImportanceResult object which contains the results for each run
    """
    # We don't need the training data, so pass empty arrays to the abstract runner
    return abstract_variable_importance((np.array([]), np.array([])), scoring_data, scoring_fn, scoring_strategy, PermutationImportanceSelectionStrategy, variable_names=variable_names, nimportant_vars=nimportant_vars, nbootstrap=nbootstrap, subsample=subsample, njobs=njobs)
