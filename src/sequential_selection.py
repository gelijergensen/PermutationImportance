"""While there are slightly different strategies for performing sequential
selection, they all use the same base idea, which is represented here"""

import numpy as np

from src.data_verification import verify_data, determine_variable_names
from src.result import ImportanceResult
from src.selection_strategies import SequentialForwardSelectionStrategy
from src.scoring_strategies import verify_scoring_strategy
from src.utils import convert_result_list_to_dict, get_data_subset


__all__ = ["sequential_forward_selection"]


def sequential_selection(training_data, scoring_data, scoring_fn, scoring_strategy, selection_strategy, variable_names=None, nimportant_vars=None, method=None, nbootstrap=1, subsample=None):
    """Performs an abstract sequential selection over data given a particular
    set of functions for scoring, determining optimal variables, and selecting
    data

    :param training_data: a 2-tuple (inputs, outputs) for training in the
        scoring_fn
    :param scoring_data: a 2-tuple (inputs, outputs) for scoring in the
        scoring_fn
    :param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float
    :param scoring_strategy: a function to be used for determining optimal
        variables or a string. If a function, should be of the form
            ([floats]) -> index. If a string, must be one of the options in
        scoring_strategies.VALID_SCORING_STRATEGIES
    :param selection_strategy: an object which, when iterated, produces triples
        (var, training_data, scoring_data). Typically a SelectionStrategy.
        Alternatively can be a function of the form (training_data, 
            scoring_data, num_vars, important_vars, bootstrap_iter, subsample)
            -> generator of (var, training_data, scoring_data)
    :param variable_names: an optional list for variable names. If not given,
        will use names of columns of data (if pandas dataframe) or column
        indices
    :param nimportant_vars: number of times to compute the next most important
        variable. Defaults to all
    :param method: a string for the name of the method used. Defaults to the
        name of the selection_strategy if not given
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :returns: ImportanceResult object which contains the results for each run
    """

    training_data = verify_data(training_data)
    scoring_data = verify_data(scoring_data)

    scoring_strategy = verify_scoring_strategy(scoring_strategy)

    variable_names = determine_variable_names(training_data, variable_names)

    nimportant_vars = len(
        variable_names) if nimportant_vars is None else nimportant_vars

    method = getattr(selection_strategy, "name", getattr(
        selection_strategy, "__name__")) if method is None else method

    subsample = 1 if subsample is None else subsample
    subsample = int(len(training_data[0]) *
                    subsample) if subsample <= 1 else subsample

    result_obj = ImportanceResult(method, variable_names)

    important_vars = list()
    num_vars = len(variable_names)
    for _ in range(nimportant_vars):
        result = None
        result_names = None
        for i in range(nbootstrap):
            # This must return in the same order each time
            selection_iter = selection_strategy(
                training_data, scoring_data, num_vars, important_vars, i, subsample)
            result_i = _singlethread_iteration(selection_iter, scoring_fn)
            if result is None:
                result = np.zeros((len(result_i), nbootstrap))
            if result_names is None:
                result_names = [res[0] for res in result_i]
            result[:, i] = [res[1] for res in result_i]
        avg_result = list(zip(result_names, np.average(result, axis=-1)))

        next_result = convert_result_list_to_dict(
            avg_result, variable_names, scoring_strategy)
        best_var = min(
            next_result.keys(), key=lambda key: next_result[key][0])
        best_index = np.flatnonzero(variable_names == best_var)[0]
        result_obj.add_new_results(
            next_result, next_important_variable=best_var)
        important_vars.append(best_index)

    return result_obj


def _singlethread_iteration(selection_iterator, scoring_fn):
    """Handles a single pass of the sequential selection algorithm, assuming a
    single worker thread

    :param selection_iterator: an object which, when iterated, produces triples
        (var, training_data, scoring_data). Typically a SelectionStrategy
    :param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float
    :returns: a list of (var, score)
    """

    result = list()
    for var, training_data, scoring_data in selection_iterator:
        score = scoring_fn(training_data, scoring_data)
        result.append((var, score))
    return result


def sequential_forward_selection(training_data, scoring_data, scoring_fn, scoring_strategy, variable_names=None, nimportant_vars=None, nbootstrap=1, subsample=None):
    """Performs an abstract sequential selection over data given a particular
    set of functions for scoring, determining optimal variables, and selecting
    data

    :param training_data: a 2-tuple (inputs, outputs) for training in the
        scoring_fn
    :param scoring_data: a 2-tuple (inputs, outputs) for scoring in the
        scoring_fn
    :param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float
    :param scoring_strategy: a function to be used for determining optimal
        variables. Should be of the form ([floats]) -> index
    :param variable_names: an optional list for variable names. If not given,
        will use names of columns of data (if pandas dataframe) or column
        indices
    :param nimportant_vars: number of times to compute the next most important
        variable. Defaults to all
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :returns: ImportanceResult object which contains the results for each run
    """
    return sequential_selection(training_data, scoring_data, scoring_fn, scoring_strategy, SequentialForwardSelectionStrategy, variable_names=variable_names, nimportant_vars=nimportant_vars)
