"""The result object keeps track of the "context" for each round of results as
well as the actual results. Additionally, it provides methods for the retrieval
of both the results without any context (singlepass, Breiman) and the most 
complete context (multipass, Lakshmanan)"""

import warnings
from itertools import izip

from src.error_handling import FullImportanceResultWarning


class ImportanceResult(object):
    """Houses the result of any importance method, which consists of a
    sequence of contexts and results. A individual result can only be truly
    interpreted correctly in light of the corresponding context. This object
    allows for indexing into the contexts and results and also provides
    convenience methods for retrieving the results with no context and the
    most complete context"""

    def __init__(self, method, variable_names, original_score):
        """Initializes the results object with the method used and a list of
        variable names

        :param method: string for the type of variable importance used
        :param variable_names: a list of names for variables
        :param original_score: the score of the model when no variables are 
            important
        """
        self.method = method
        self.variable_names = variable_names
        self.original_score = original_score
        # The initial context is "empty"
        self.contexts = [{}]
        self.results = list()

        self.complete = False

    def add_new_results(self, new_results, next_important_variable=None):
        """Adds a new round of results

        :param new_results: a dictionary with keys of variable names and values
            of (rank, score)
        :param next_important_variable: variable name of the next most important
            variable. If not given, will select the variable with the smallest
            rank
        """
        if not self.complete:
            if next_important_variable is None:
                next_important_variable = min(
                    new_results.keys(), key=lambda key: new_results[key][0])
            self.results.append(new_results)
            new_context = self.contexts[-1].copy()
            self.contexts.append(new_context)
            __, score = new_results[next_important_variable]
            self.contexts[-1][next_important_variable] = (
                len(self.results) - 1, score)
            # Check to see if this result could constitute the last possible one
            if len(self.results) == len(self.variable_names):
                self.results.append(dict())
                self.complete = True
        else:
            warnings.warn(
                "Cannot add new result to full ImportanceResult", FullImportanceResultWarning)

    def retrieve_singlepass(self):
        return self.results[0]

    def retrieve_multipass(self):
        return self.contexts[-1]

    def __iter__(self):
        return izip(self.contexts, self.results)

    def __getitem__(self, index):
        if index < 0:
            index = len(self.results) + index
        return (self.contexts[index], self.results[index])

    def __len__(self):
        return len(self.results)
