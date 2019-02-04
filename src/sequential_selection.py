"""Sequential Selection methods iteratively add variables to the set of 
important variables by training a new model for each subset of the variables"""

from src.abstract_runner import abstract_variable_importance
from src.selection_strategies import SequentialForwardSelectionStrategy, SequentialBackwardSelectionStrategy

__all__ = ["sequential_forward_selection", "sequential_backward_selection"]


def sequential_forward_selection(training_data, scoring_data, scoring_fn, scoring_strategy, variable_names=None, nimportant_vars=None, nbootstrap=1, subsample=1, njobs=1):
    """Performs sequential forward selection over data given a particular
    set of functions for scoring and determining optimal variables

    : param training_data: a 2-tuple(inputs, outputs) for training in the
        scoring_fn
    : param scoring_data: a 2-tuple(inputs, outputs) for scoring in the
        scoring_fn
    : param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float
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
    return abstract_variable_importance(training_data, scoring_data, scoring_fn, scoring_strategy, SequentialForwardSelectionStrategy, variable_names=variable_names, nimportant_vars=nimportant_vars, nbootstrap=nbootstrap, subsample=subsample, njobs=njobs)


def sequential_backward_selection(training_data, scoring_data, scoring_fn, scoring_strategy, variable_names=None, nimportant_vars=None, nbootstrap=1, subsample=1, njobs=1):
    """Performs sequential backward selection over data given a particular
    set of functions for scoring and determining optimal variables

    : param training_data: a 2-tuple(inputs, outputs) for training in the
        scoring_fn
    : param scoring_data: a 2-tuple(inputs, outputs) for scoring in the
        scoring_fn
    : param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float
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
    return abstract_variable_importance(training_data, scoring_data, scoring_fn, scoring_strategy, SequentialBackwardSelectionStrategy, variable_names=variable_names, nimportant_vars=nimportant_vars, nbootstrap=nbootstrap, subsample=subsample, njobs=njobs)
