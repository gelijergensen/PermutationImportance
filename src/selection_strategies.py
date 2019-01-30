
import numpy as np

from src.utils import get_data_subset

__all__ = ["SequentialForwardSelectionStrategy",
           "SequentialBackwardSelectionStrategy"]


class SelectionStrategy(object):
    """Selection strategies accept the data and know how to provide necessary
    subsets of it"""

    name = "Abstract Selection Strategy"

    def __init__(self, training_data, scoring_data, num_vars, important_vars, bootstrap_iter, subsample):
        """Initializes the object by storing the data and keeping track of other
        important information

        :param training_data: (training_inputs, training_outputs)
        :param scoring_data: (scoring_inputs, scoring_outputs)
        :param num_vars: integer for the total number of variables
        :param important_vars: a list of the indices of variables which are already
            considered important
        :param bootstrap_iter: number for which bootstrap iteration this is
        :param subsample: number of training examples to take
        """
        self.scoring_data = scoring_data
        self.num_vars = num_vars
        self.important_vars = important_vars
        self.bootstrap_iter = bootstrap_iter
        self.subsample = subsample

        # We need to subsample the training_data now
        if self.subsample == training_data[0].shape[0]:
            # Deterministic
            train_rows = np.arange(training_data[0].shape[0])
        else:
            # First we randomly pick the examples for training
            train_rows = np.random.choice(
                training_data[0].shape[0], self.subsample)
        self.training_data = (get_data_subset(
            training_data[0], train_rows), get_data_subset(training_data[1], train_rows))

    def generate_datasets(self, important_variables):
        """Generator which returns triples (variable, training_data, scoring_data)"""
        raise NotImplementedError(
            "Please implement a strategy for generating datasets on class %s" % self.name)

    def generate_all_datasets(self):
        """By default, loops over all variables not yet considered important"""
        for var in range(self.num_vars):
            if var not in self.important_vars:
                training_data, scoring_data = self.generate_datasets(
                    self.important_vars + [var, ])
                yield (var, training_data, scoring_data)

    def __iter__(self):
        return self.generate_all_datasets()


class SequentialForwardSelectionStrategy(SelectionStrategy):
    """Sequential Forward Selection tests each variable and attempts to add one
    new variable to the dataset on each iteration"""

    name = "Sequential Forward Selection"

    def generate_datasets(self, important_variables):
        """Check each of the non-important variables. Dataset is the columns which
        are important plus the one being evaluated

        :returns: (training_data, scoring_data)
        """
        training_inputs, training_outputs = self.training_data
        scoring_inputs, scoring_outputs = self.scoring_data

        columns = important_variables
        # Make a slice of the training inputs
        training_inputs_subset = get_data_subset(
            training_inputs, None, columns)
        # Make a slice of the scoring inputs
        scoring_inputs_subset = get_data_subset(
            scoring_inputs, None, columns)
        return (training_inputs_subset, training_outputs), (scoring_inputs_subset, scoring_outputs)


class SequentialBackwardSelectionStrategy(SelectionStrategy):
    """Sequential Backward Selection tests each variable and attempts to remove
    one new variable from the dataset on each iteration"""

    name = "Sequential Backward Selection"

    def generate_datasets(self, important_variables):
        """Check each of the non-important variables. Dataset is the columns which
        are not important minus the one being evaluated

        :yields: a sequence of (variable being evaluated, columns to include)
        """
        training_inputs, training_outputs = self.training_data
        scoring_inputs, scoring_outputs = self.scoring_data

        columns = [x for x in range(self.num_vars)
                   if x not in important_variables]
        # Make a slice of the training inputs
        training_inputs_subset = get_data_subset(
            training_inputs, None, columns)
        # Make a slice of the scoring inputs
        scoring_inputs_subset = get_data_subset(
            scoring_inputs, None, columns)
        return (training_inputs_subset, training_outputs), (scoring_inputs_subset, scoring_outputs)
