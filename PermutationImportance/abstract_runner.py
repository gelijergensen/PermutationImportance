"""The general algorithm for all of the data-based variable importance methods
is the same, regardless of whether the method is Sequential Selection or 
Permutation Importance or something else. This is represented in the 
`abstract_variable_importance` function. All of the different methods we provide
use this function under the hood and the only difference between them is the
`selection_strategy` object, which is detailed in 
PermutationImportance.selection_strategies. Typically, you will not need to use
this method but can instead us one of the methods imported directly into the 
top package of PermutationImportance.

If you wish to implement your own variable importance method, you will need to
devise your own `selection_strategy`. We recommend using
PermutationImportance.sequential_selection as a template for implementing your
own variable importance method."""

import numpy as np
import multiprocessing as mp

from .data_verification import verify_data, determine_variable_names
from .multiprocessing_utils import pool_imap_unordered
from .result import ImportanceResult
from .scoring_strategies import verify_scoring_strategy
from .utils import add_ranks_to_dict, get_data_subset


def abstract_variable_importance(training_data, scoring_data, scoring_fn, scoring_strategy, selection_strategy, variable_names=None, nimportant_vars=None, method=None, njobs=1):
    """Performs an abstract variable importance over data given a particular
    set of functions for scoring, determining optimal variables, and selecting
    data

    :param training_data: a 2-tuple (inputs, outputs) for training in the
        scoring_fn
    :param scoring_data: a 2-tuple (inputs, outputs) for scoring in the
        scoring_fn
    :param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> some value
    :param scoring_strategy: a function to be used for determining optimal
        variables or a string. If a function, should be of the form
            ([some value]) -> index. If a string, must be one of the options in
        scoring_strategies.VALID_SCORING_STRATEGIES
    :param selection_strategy: an object which, when iterated, produces triples
        (var, training_data, scoring_data). Almost certainly a SelectionStrategy
    :param variable_names: an optional list for variable names. If not given,
        will use names of columns of data (if pandas dataframe) or column
        indices
    :param nimportant_vars: number of times to compute the next most important
        variable. Defaults to all
    :param method: a string for the name of the method used. Defaults to the
        name of the selection_strategy if not given
    :param njobs: an integer for the number of threads to use. If negative, will
        use the number of cpus + njobs. Defaults to 1
    :returns: ImportanceResult object which contains the results for each run
    """

    training_data = verify_data(training_data)
    scoring_data = verify_data(scoring_data)
    scoring_strategy = verify_scoring_strategy(scoring_strategy)
    variable_names = determine_variable_names(scoring_data, variable_names)
    nimportant_vars = len(
        variable_names) if nimportant_vars is None else nimportant_vars
    method = getattr(selection_strategy, "name", getattr(
        selection_strategy, "__name__")) if method is None else method
    njobs = mp.cpu_count() + njobs if njobs <= 0 else njobs

    important_vars = list()
    num_vars = len(variable_names)

    # Compute the original score over all the data
    original_score = scoring_fn(training_data, scoring_data)
    result_obj = ImportanceResult(method, variable_names, original_score)
    for _ in range(nimportant_vars):
        selection_iter = selection_strategy(
            training_data, scoring_data, num_vars, important_vars)
        if njobs == 1:
            result = _singlethread_iteration(
                selection_iter, scoring_fn)
        else:
            result = _multithread_iteration(
                selection_iter, scoring_fn, njobs)
        next_result = add_ranks_to_dict(
            result, variable_names, scoring_strategy)
        best_var = min(
            next_result.keys(), key=lambda key: next_result[key][0])
        best_index = np.flatnonzero(variable_names == best_var)[0]
        result_obj.add_new_results(
            next_result, next_important_variable=best_var)
        important_vars.append(best_index)

    return result_obj


def _singlethread_iteration(selection_iterator, scoring_fn):
    """Handles a single pass of the abstract variable importance algorithm, 
    assuming a single worker thread

    :param selection_iterator: an object which, when iterated, produces triples
        (var, training_data, scoring_data). Typically a SelectionStrategy
    :param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float
    :returns: a dict of {var: score}
    """
    result = dict()
    for var, training_data, scoring_data in selection_iterator:
        score = scoring_fn(training_data, scoring_data)
        result[var] = score
    return result


def _multithread_iteration(selection_iterator, scoring_fn, njobs):
    """Handles a single pass of the abstract variable importance algorithm using
    multithreading

    :param selection_iterator: an object which, when iterated, produces triples
        (var, training_data, scoring_data). Typically a SelectionStrategy
    :param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float
    :param num_jobs: number of processes to use
    :returns: a dict of {var: score}
    """
    result = dict()
    for index, score in pool_imap_unordered(scoring_fn, selection_iterator, njobs):
        result[index] = score
    return result
