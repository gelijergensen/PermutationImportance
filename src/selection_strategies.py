
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
        self.training_data = training_data
        self.scoring_data = scoring_data
        self.num_vars = num_vars
        self.important_vars = important_vars
        self.bootstrap_iter = bootstrap_iter
        self.subsample = subsample

    def generate_datasets(self):
        """Generator which returns triples (variable, training_data, scoring_data)"""
        raise NotImplementedError(
            "Please implement a strategy for generating datasets on class %s" % self.name)

    def __iter__(self):
        return self.generate_datasets()


class SequentialForwardSelectionStrategy(SelectionStrategy):
    """Sequential Forward Selection tests each variable and attempts to add one
    new variable to the dataset on each iteration"""

    name = "Sequential Forward Selection"

    def generate_datasets(self):
        """Check each of the non-important variables. Dataset is the columns which
        are important plus the one being evaluated

        :yields: a sequence of (variable being evaluated, columns to include)
        """
        training_inputs, training_outputs = self.training_data
        scoring_inputs, scoring_outputs = self.scoring_data

        if self.subsample == training_inputs.shape[0]:
            # Deterministic
            train_rows = np.arange(training_inputs.shape[0])
        else:
            # First we randomly pick the examples for training
            train_rows = np.random.choice(
                training_inputs.shape[0], self.subsample)
        all_score_rows = np.arange(scoring_inputs.shape[0])

        for var in range(self.num_vars):
            if var not in self.important_vars:
                columns = self.important_vars + [var, ]
                # Make a slice of the training data
                training_inputs_subset = get_data_subset(
                    training_inputs, train_rows, columns)
                training_outputs_subset = get_data_subset(
                    training_outputs, train_rows)
                # Make a slice of the scoring data
                scoring_inputs_subset = get_data_subset(
                    scoring_inputs, all_score_rows, columns)
                yield((var, (training_inputs_subset, training_outputs_subset), (scoring_inputs_subset, scoring_outputs)))


class SequentialBackwardSelectionStrategy(SelectionStrategy):
    """Sequential Backward Selection tests each variable and attempts to remove
    one new variable from the dataset on each iteration"""

    name = "Sequential Backward Selection"

    def generate_datasets(self):
        """Check each of the non-important variables. Dataset is the columns which
        are not important minus the one being evaluated

        :yields: a sequence of (variable being evaluated, columns to include)
        """
        training_inputs, training_outputs = self.training_data
        scoring_inputs, scoring_outputs = self.scoring_data

        if self.subsample == training_inputs.shape[0]:
            # Deterministic
            train_rows = np.arange(training_inputs.shape[0])
        else:
            # First we randomly pick the examples for training
            train_rows = np.random.choice(
                training_inputs.shape[0], self.subsample)
        all_score_rows = np.arange(scoring_inputs.shape[0])

        for var in range(self.num_vars):
            if var not in self.important_vars:
                columns = [x for x in range(self.num_vars) if x not in (
                    self.important_vars + [var, ])]
                # Make a slice of the training data
                training_inputs_subset = get_data_subset(
                    training_inputs, train_rows, columns)
                training_outputs_subset = get_data_subset(
                    training_outputs, train_rows)
                # Make a slice of the scoring data
                scoring_inputs_subset = get_data_subset(
                    scoring_inputs, all_score_rows, columns)
                yield((var, (training_inputs_subset, training_outputs_subset), (scoring_inputs_subset, scoring_outputs)))
