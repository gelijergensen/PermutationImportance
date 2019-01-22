"""While there are slightly different strategies for performing sequential 
selection, they all use the same base idea, which is represented here"""


from src.scoring_strategies import verify_scoring_strategy


def sequential_selection(training_data, scoring_data, scoring_fn, scoring_strategy, selection_strategy, variable_names=None):
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
    :param selection_strategy: a function to be used for selecting a fraction of
        the data to be used at each iteration. Should be of the form 
        (num_vars, important_vars: [index], current: index) -> column_list
    :param variable_names: an optional list for variable names. If not given, 
        will use names of columns of data (if pandas dataframe) or column 
        indices
    :returns: ???? - object representing result
    """
    pass
