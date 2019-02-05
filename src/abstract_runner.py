"""While there are slightly different strategies for performing the various
importance methods, they all use the same base idea, which is represented here"""

from itertools import imap, tee
import numpy as np
import multiprocessing as mp

from src.data_verification import verify_data, determine_variable_names
from src.result import ImportanceResult
from src.scoring_strategies import verify_scoring_strategy
from src.utils import add_ranks_to_dict, get_data_subset


def abstract_variable_importance(training_data, scoring_data, scoring_fn, scoring_strategy, selection_strategy, variable_names=None, nimportant_vars=None, method=None, njobs=1):
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

    # Compute the original score (score over no variables considered important)
    original_score = scoring_fn(*selection_strategy(
        training_data, scoring_data, num_vars, important_vars).generate_datasets([]))
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
    """Handles a single pass of the sequential selection algorithm, assuming a
    single worker thread

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
    """Handles a single pass of the sequential selection algorithm, assuming a
    single worker thread

    :param selection_iterator: an object which, when iterated, produces triples
        (var, training_data, scoring_data). Typically a SelectionStrategy
    :param scoring_fn: a function to be used for scoring. Should be of the form
        (training_data, scoring_data) -> float
    :param num_jobs: number of processes to use
    :returns: a dict of {var: score}
    """
    in_queue = mp.Queue(maxsize=njobs)
    out_queue = mp.Queue()
    pool = mp.Pool(njobs, initializer=_multithreaded_runner,
                   initargs=(in_queue, out_queue, scoring_fn))
    for item in selection_iterator:
        in_queue.put(item)
    for _ in range(njobs):
        # tell the workers we are finished
        in_queue.put(None)
    pool.close()
    pool.join()
    result = dict()
    while not out_queue.empty():
        res = out_queue.get()
        result[res[0]] = res[1]
    return result


def _multithreaded_runner(in_queue, out_queue, func):
    """Actual process running in parallel. Accepts a function to perform on
    members of the queue. Only applies the function to all but the first item of
    each member

    :param in_queue: queue which is providing the inputs
    :param out_queue: queue which holds the outputs
    """
    while True:
        in_args = in_queue.get()
        if in_args is None:
            break
        else:
            out_queue.put((in_args[0], func(*in_args[1:])))
