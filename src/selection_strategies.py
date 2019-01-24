

__all__ = ["sfs_strategy"]


def sfs_strategy(num_vars, important_vars, bootstrap_iter, subsample):
    """Check each of the non-important variables. Dataset is the columns which
    are important plus the one being evaluated

    :param num_vars: integer for the total number of variables
    :param important_vars: a list of the indices of variables which are already
        considered important
    :param bootstrap_iter: ignored
    :param subsample: ignored
    :returns: a list of (variable being evaluated, columns to include)
    """

    to_test = list()
    for var in range(num_vars):
        if var not in important_vars:
            to_test.append((var, important_vars + [var, ]))
    return to_test
