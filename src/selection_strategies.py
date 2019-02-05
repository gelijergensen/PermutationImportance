
import numpy as np

from src.utils import get_data_subset, make_data_from_columns

__all__ = ["SequentialForwardSelectionStrategy",
           "SequentialBackwardSelectionStrategy"]


class SelectionStrategy(object):
    """Selection strategies accept the data and know how to provide necessary
    subsets of it"""

    name = "Abstract Selection Strategy"

    def __init__(self, training_data, scoring_data, num_vars, important_vars):
        """Initializes the object by storing the data and keeping track of other
        important information

        :param training_data: (training_inputs, training_outputs)
        :param scoring_data: (scoring_inputs, scoring_outputs)
        :param num_vars: integer for the total number of variables
        :param important_vars: a list of the indices of variables which are already
            considered important
        """
        self.training_data = training_data
        self.scoring_data = scoring_data
        self.num_vars = num_vars
        self.important_vars = important_vars

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
        are important

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
        are not important

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


class PermutationImportanceSelectionStrategy(SelectionStrategy):
    """Permutation Importance shuffles the columns which are important"""

    name = "Permutation Importance"

    def __init__(self, training_data, scoring_data, num_vars, important_vars):
        """Initializes the object by storing the data and keeping track of other
        important information

        :param training_data: (training_inputs, training_outputs)
        :param scoring_data: (scoring_inputs, scoring_outputs)
        :param num_vars: integer for the total number of variables
        :param important_vars: a list of the indices of variables which are already
            considered important
        """
        super(PermutationImportanceSelectionStrategy, self).__init__(
            training_data, scoring_data, num_vars, important_vars)
        # Also initialize the "shuffled data"
        scoring_inputs, __ = self.scoring_data
        indices = np.random.permutation(len(scoring_inputs))
        self.shuffled_scoring_inputs = get_data_subset(
            scoring_inputs, indices)  # This copies

    def generate_datasets(self, important_variables):
        """Check each of the non-important variables. Dataset has columns which
        are important shuffled

        :returns: (training_data, scoring_data)
        """
        scoring_inputs, scoring_outputs = self.scoring_data
        complete_scoring_inputs = make_data_from_columns(
            [get_data_subset(self.shuffled_scoring_inputs if i in important_variables else scoring_inputs, None, [i]) for i in range(self.num_vars)])

        return self.training_data, (complete_scoring_inputs, scoring_outputs)
