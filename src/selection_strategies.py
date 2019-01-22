

__all__ = ["sfs_strategy"]


def sfs_strategy(num_vars, important_vars, current):
    """Simply appends the current var to the list of important_vars. Has the
    effect of selecting one new variable at each round

    :param num_vars: integer for the total number of variables
    :param important_vars: a list of the indices of variables which are already
        considered important
    :param current: index for the current variable to consider
    """
    return important_vars + [current, ]
