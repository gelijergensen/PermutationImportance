"""This is a standalone set of functions which compute the importances for each variable in a dataset, according to a
particular model using multiprocessing.

The main method is the permutation_selection_variable_importances function.

This file uses two different methods for computing the importance, one which tries to share memory between threads
and one which does not. The "windows" method does not share memory (this is a consequence of the fact that forking is a
Unix only thing). This method should work for any OS and will also likely work for Python 3.x. It is more memory
greedy, but may also be somewhat faster.

Additionally, the non-greedy method assumes that the input variables are doubles (c_double) and the outputs are either
strings (c_char_p) or doubles. If this is not the case, you can always use the memory greedy version or import the
correct type from the ctypes module.

@author G. Eli Jergensen <gelijergensen@ou.edu>
"""
import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocessing.sharedctypes import RawArray
from ctypes import c_double, c_char_p
from collections import deque
import sys

# These are global variables which are inherited by compute_score_for_column
inputs_share = None
outputs_share = None


def permutation_selection_variable_importances(model, classes, testing_input, testing_output, npermute, subsamples=None,
                                               score_fn=None, optimization='minimize', nimportant_variables=None,
                                               njobs=None, share_vars=True, diagnostics=False):
    """Determines the variable importances for an estimator for a particular testing set using permutation selection

    :param model: An object with a method named predict of the form (testing_input) -> class
    :param classes: A set of possible labels for (all) the data (required in case the testing_output happens to not have
     full coverage)
    :param testing_input: testing data inputs to the model
    :param testing_output: true labels of the testing data
    :param npermute: number of times to permute each column
    :param subsamples: number of elements to sample (with replacement) per bootstrap. If between 0 and 1, this is
     treated as a fraction of the number of total number of events (e.g. 0.5 means half the number of events). If not
     specified, subsampling will not be used and the entire data will be used (without replacement)
    :param score_fn: a function of the form (new_predictions, truths, classes) -> float. Defaults to
     an accuracy scorer.
    :param optimization: How to select the most important variable given a list of scores for each variable. If 
     'maximize' or 'minimize' then the maximal (minimal) score in the list is selected as most important. Can also be a
     function of the form (list of floats) -> index
     REMEMBER THAT IF YOU WISH TO MAXIMIZE THE SCORE, THE VARIABLE WITH THE LOWEST RESULTING SCORE IS THE MOST IMPORTANT
     e.g. with accuracy you should use 'minimize', but with some 'error' score function, you should use 'maximize'
    :param nimportant_variables: the number of most important variables to ask for. Defaults to as many variables as in
     input
    :param njobs: number of process (threads) to run this process in. If not specified, defaults to the number of cpus
     minus one.
    :param share_vars: Set to true to try to share memory between threads. Defaults to true. Will override to false on
     windows.
    :param diagnostics: Set to true to enable printouts
    :returns: A list of triples: (all_important_indices_ordered, score_before_permuting, all_scores_after_permuting)
        the length of the list of triples is nimportant_variables
        all_scores_after_permuting contains None values for the columns which weren't tested because they were already
            considered important
    """

    if sys.platform == "win32":
        if diagnostics:
            print("OS is windows. Resorting to memory greedy version")
        share_vars = False

    # If subsamples is less than 1, treat it as a fraction of the total number of events
    if subsamples is not None and (0 < subsamples <= 1):
        subsamples = int(len(testing_output) * subsamples)
    # Check if the score_fn is None
    if score_fn is None:
        score_fn = accuracy_scorer
    # Determine how many variables to return at the end
    if nimportant_variables is None:
        nimportant_variables = len(testing_input[0])
    elif nimportant_variables > len(testing_input[0]):
        # ensure we don't try to return more variables than we have
        nimportant_variables = len(testing_input[0])

    # REMEMBER THAT IF YOU WISH TO MAXIMIZE THE SCORE, THE VARIABLE WITH THE LOWEST RESULTING SCORE IS MOST IMPORTANT
    # e.g. with accuracy you should use 'minimize', but with some 'error' score function, you should use 'maximize'
    if 'max' in optimization:
        # when minimizing a score, the variable whose score is highest after permutation is most important
        optimization = np.argmax
    elif 'min' in optimization:
        # when ma = core, the variable whose score is lowest after permutation is most import
        optimization = np.argmin
    else:
        assert callable(
            optimization), "ERROR: optimization must be 'minimize', 'maximize', or a callable"

    # determine how to compute the score of a single column
    if share_vars:
        if diagnostics:
            print("Performing permutation selection")
        # we can use global variables to store things
        column_scorer = compute_score_for_column_unpack_share
        column_scorer_params = (model, classes, npermute, subsamples, score_fn)
    else:
        if diagnostics:
            print("Performing permutation selection in memory greedy mode")
        # We have to pass along the raw data
        column_scorer = compute_score_for_column_unpack
        column_scorer_params = (
            model, classes, npermute, subsamples, testing_input, testing_output, score_fn)

    # package things up if we can
    if share_vars:
        # ---- Prepackage the shared vars to the shared_variables_for_concurrency (shared_vars) module ----
        global inputs_share
        global outputs_share
        # Since the arrays are very large, we put them into shared memory
        # The multi-dimensional array is tricky
        # First, we flatten the array, then put it in memory, then grab it back out using numpy and then reshape it
        inputs_flattened = np.asarray(
            testing_input).flatten()  # flatten the inputs array
        inputs_shape = np.asarray(testing_input).shape
        # array of doubles (not write safe)
        inputs_share_base = RawArray(c_double, inputs_flattened)
        # shares memory with above, but is numpy array
        inputs_share_np = np.frombuffer(inputs_share_base)
        inputs_share = inputs_share_np.reshape(
            inputs_shape)  # reshape back to the way we expect
        # The single-dimensional arrays are simpler
        if isinstance(testing_output[0], str):
            # The outputs are strings
            # array of strings (not write safe)
            outputs_share = RawArray(c_char_p, testing_output)
        else:
            # Assume the outputs are doubles
            # array of doubles (not write safe)
            outputs_share = RawArray(c_double, testing_output)
        # ---- the two *_share arrays are now located in shared memory ----

    # ---- Construct the pool and parallelize! ----
    if njobs is None:
        njobs = cpu_count() - 1  # Minus 1 so that we don't completely overload everything
    if diagnostics:
        print("Number of workers: %i" % njobs)

    # ---- Actually perform the permutation selection ----
    num_cols = len(testing_input[0])
    original_score = score_fn(model.predict(
        testing_input), testing_output, classes)

    def find_most_import_var_rec(important_col_idxs, previous_score, num_times_to_test):
        """Recursively finds the next most important variables and returns the results as a deque

        :param important_col_idxs: list of indices which will always be permuted (because they have already been
            determined important)
        :param previous_score: score resulting from permuting all cols in important_col_idxs (and no others)
        :param num_times_to_test: number of times to recursively call this function
        :returns: each step returns a triple: 
            (ordered_important_vars_list, score_before_permuting, list_of_scores_after_permuting)
            so the entire result is a list (deque) of these triples"""
        cols_to_test = list()
        for var in range(num_cols):
            if var not in important_col_idxs:
                cols_to_test.append(var)

        pool = Pool(processes=njobs, maxtasksperchild=1)
        result = pool.map_async(column_scorer, [(j, important_col_idxs) + column_scorer_params
                                                for j in cols_to_test])
        pool.close()
        pool.join()
        scores = result.get()

        # Determine the next most important variable
        # the proxy column number (in the incomplete scores list)
        most_important_column = optimization(scores)
        # the true column number
        most_important_variable_index = cols_to_test[most_important_column]
        most_important_score = scores[most_important_column]
        # Package up all scores into a nice list (with Nones for untested values)
        all_scores = list()
        counter = 0
        for var in range(num_cols):
            if var not in cols_to_test:
                all_scores.append(None)
            else:
                all_scores.append(scores[counter])
                counter += 1

        important_col_idxs.append(most_important_variable_index)
        these_results = (list(important_col_idxs), previous_score,
                         all_scores)  # clone the list of indices

        if num_times_to_test == 1:
            results = deque()
        else:

            results = find_most_import_var_rec(
                important_col_idxs, most_important_score, num_times_to_test-1)
        results.appendleft(these_results)
        return results

    complete_results = find_most_import_var_rec(
        list(), original_score, nimportant_variables)  # recursive!
    return list(complete_results)  # convert from a deque back to a list


def compute_score_for_column_unpack(args):
    """A wrapper function around the compute_score_for_column function which allows us to pass multiple arguments.

    :param args: A tuple of arguments, as defined in compute_score_for_column below
    :return: compute_score_for_column(*args)
    """
    return compute_score_for_column(*args)


def compute_score_for_column_unpack_share(args):
    """A slightly different wrapper function around the compute_score_for_column which passes multiple arguments,
    including some global variables

    :param args: A tuple of arguments: (c, model, classes, npermute, subsamples, score_fn)
    :return: compute_score_for_column
    """
    # grab all the arguments and order them correctly
    c, important_columns, model, classes, npermute, subsamples, score_fn = args
    global inputs_share
    global outputs_share
    return compute_score_for_column(c, important_columns, model, classes, npermute, subsamples,
                                    inputs_share, outputs_share, score_fn)


def compute_score_for_column(c, important_columns, model, classes, npermute, subsamples, inputs, outputs, score_fn):
    """This function is mapped over all of the columns of the testing_data and a score computed for each

    :param c: The index of this column
    :param important_columns: A list of columns which are already considered important and also need to be permuted
    :param classes: A set of possible labels for (all) the data
    :param model: The model which is to be evaluated. Must have a predict method
    :param npermute: number of bootstrapping runs (number of times to permute each column)
    :param subsamples: number of elements to sample (with replacement) per bootstrap. If between 0 and 1, this is
     treated as a fraction of the number of total number of events (e.g. 0.5 means half the number of events). If not
     specified, subsampling will not be used and the entire data will be used (without replacement)
    :param inputs: testing data inputs to the model (after any preprocessing)
    :param outputs: true labels of the testing data
    :param score_fn: a function of the form (predictions, truths, classes) -> float.
    :returns: A float for the mean score of this column"""

    all_cols_to_permute = important_columns
    all_cols_to_permute.append(c)

    # npermute times per column
    scores_for_col = list()
    for b in range(npermute):
        # Permute the important columns, along with this column
        # Clone the array, making a new(!) one
        permuted_inputs = np.copy(inputs)
        for col in all_cols_to_permute:
            permuted_col = np.random.permutation(inputs[:, col])
            permuted_inputs[:, col] = permuted_col

        if subsamples is not None:

            # Subsample the inputs and outputs and permuted inputs (with replacement)
            subsampled_idxs = np.random.choice(range(len(outputs)), subsamples)
            subsampled_permuted_inputs = [
                permuted_inputs[idx] for idx in subsampled_idxs]
            subsampled_outputs = [outputs[idx] for idx in subsampled_idxs]

            # Predict on the subsampled, permuted inputs
            predictions = model.predict(subsampled_permuted_inputs)

            # Score the model on the permuted data
            score = score_fn(predictions, subsampled_outputs, classes)
        else:
            # Instead of subsampling, use the entire dataset without replacement
            predictions = model.predict(permuted_inputs)

            # Score the model on the permuted data
            score = score_fn(predictions, outputs, classes)
        scores_for_col.append(score)

    return np.average(scores_for_col, axis=0)


"""Each of these functions with 'scorer' at the end can be used for the score_fn parameter of 
permutation_selection_variable_importances. Or you can use your own"""


def accuracy_scorer(new_predictions, truths, classes):
    """Scores the model based on the accuracy of the new predictions

    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: NOT USED
    :returns: the accuracy of the new_predictions
    """

    new_accuracy = float([new_predictions[idx] == truths[idx]
                          for idx in range(len(truths))].count(True)) / float(len(truths))

    return new_accuracy


def gerrity_skill_scorer(new_predictions, truths, classes):
    """Determines the gerrity skill score, returning a scalar

    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a single value for the gerrity skill score
    """
    # get the confusion matrices
    new_cm = confusion_matrix(new_predictions, truths, classes)
    new_table = MulticlassContingencyTable(
        table=new_cm, n_classes=len(classes), class_names=classes)

    return new_table.gerrity_score()


def peirce_skill_scorer(new_predictions, truths, classes):
    """Determines the peirce skill score, returning a scalar

    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a single value for the peirce skill score
    """
    # get the confusion matrices
    new_cm = confusion_matrix(new_predictions, truths, classes)
    new_table = MulticlassContingencyTable(
        table=new_cm, n_classes=len(classes), class_names=classes)

    return new_table.peirce_skill_score()


def heidke_skill_scorer(new_predictions, truths, classes):
    """Determines the heidke skill score, returning a scalar

    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a single value for the heidke skill score
    """
    # get the confusion matrices
    new_cm = confusion_matrix(new_predictions, truths, classes)
    new_table = MulticlassContingencyTable(
        table=new_cm, n_classes=len(classes), class_names=classes)

    return new_table.heidke_skill_score()


def categorical_metascorer(scoring_fn, category=None, selection_strategy=None):
    """This is a meta-scorer which converts a binary-class scorer to a multi-class scorer for use above using a 
    one-versus-rest strategy or another specified strategy

    :param scoring_fn: function which is a binary-class scorer (like bias). Must be of the form:
        (new_predictions, truths, classes, particular class) -> float
    :param category: the identity of the class to consider (if doing one-versus-rest)
    :param selection_strategy: either "maximimum", "minimum", "average", or a callable
        NOTE: if neither category or selection_strategy is specified, prints a warning and defaults to average
        callable must be of the form (list of scores) -> float
        category is ignored if selection_strategy is specified
    :returns: scoring function which wraps correctly around scoring_fn
    """
    # First determine whether we are doing ovr or a specified strategy

    if category is None and selection_strategy is None:
        print("WARNING: categorical_metascorer defaulting to averaging")
        selection_strategy = 'average'

    if 'max' in selection_strategy:
        selection_strategy = np.max
    elif 'min' in selection_strategy:
        selection_strategy = np.min
    elif 'avg' in selection_strategy or 'average' in selection_strategy:
        selection_strategy = np.average
    else:
        assert callable(
            selection_strategy), "ERROR: strategy must be 'minimize', 'maximize', 'average' or a callable"

    def cat_scorer(new_predictions, truths, classes):
        if selection_strategy is None:  # then we know category isn't None
            return scoring_fn(new_predictions, truths, classes, category)
        else:
            all_scores = [scoring_fn(
                new_predictions, truths, classes, bin_class) for bin_class in classes]
            return selection_strategy(all_scores)

    return cat_scorer


"""This is a utility function for helping score things. You may also find it useful when 
implementing your own scoring functions"""


def confusion_matrix(predictions, truths, classes):
    """Computes the confusion matrix of the predictions vs truths

    :param predictions: model predictions
    :param truths: true labels
    :param classes: an ordered set for the label possibilities
    :returns: a numpy array for the confusion matrix
    """

    matrix = np.zeros((len(classes), len(classes)), dtype=np.float32)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            matrix[i, j] = [p == c1 and t == c2 for p,
                            t in zip(predictions, truths)].count(True)
    return matrix


"""This class is borrowed wholesale from the hagelslag repository as of 11 June 2018. It is reproduced here
with permission. Party responsible for copying: G. Eli Jergensen

@author David John Gagne <djgagne@ou.edu>"""


class MulticlassContingencyTable(object):
    """
    This class is a container for a contingency table containing more than 2 classes.
    The contingency table is stored in table as a numpy array with the rows corresponding to forecast categories,
    and the columns corresponding to observation categories.
    """

    def __init__(self, table=None, n_classes=2, class_names=("1", "0")):
        self.table = table
        self.n_classes = n_classes
        self.class_names = class_names
        if table is None:
            self.table = np.zeros((self.n_classes, self.n_classes), dtype=int)

    def __add__(self, other):
        assert self.n_classes == other.n_classes, "Number of classes does not match"
        return MulticlassContingencyTable(self.table + other.table,
                                          n_classes=self.n_classes,
                                          class_names=self.class_names)

    def peirce_skill_score(self):
        """
        Multiclass Peirce Skill Score (also Hanssen and Kuipers score, True Skill Score)
        """
        n = float(self.table.sum())
        nf = self.table.sum(axis=1)
        no = self.table.sum(axis=0)
        correct = float(self.table.trace())
        return (correct / n - (nf * no).sum() / n ** 2) / (1 - (no * no).sum() / n ** 2)

    def gerrity_score(self):
        """
        Gerrity Score, which weights each cell in the contingency table by its observed relative frequency.
        :return:
        """
        k = self.table.shape[0]
        n = float(self.table.sum())
        p_o = self.table.sum(axis=0) / n
        p_sum = np.cumsum(p_o)[:-1]
        a = (1.0 - p_sum) / p_sum
        s = np.zeros(self.table.shape, dtype=float)
        for (i, j) in np.ndindex(*s.shape):
            if i == j:
                s[i, j] = 1.0 / (k - 1.0) * \
                    (np.sum(1.0 / a[0:j]) + np.sum(a[j:k-1]))
            elif i < j:
                s[i, j] = 1.0 / (k - 1.0) * (np.sum(1.0 / a[0:i]
                                                    ) - (j - i) + np.sum(a[j:k-1]))
            else:
                s[i, j] = s[j, i]
        return np.sum(self.table / float(self.table.sum()) * s)

    def heidke_skill_score(self):
        n = float(self.table.sum())
        nf = self.table.sum(axis=1)
        no = self.table.sum(axis=0)
        correct = float(self.table.trace())
        return (correct / n - (nf * no).sum() / n ** 2) / (1 - (nf * no).sum() / n ** 2)


if __name__ == "__main__":
    """This is the debugger, which also provides an example for using the above"""

    class FakeModel(object):
        """Computes a particular linear combination of the first 3 input variables of a dataset"""

        def __init__(self, *coefs):
            """Store the coeficients to multiply the first 3 input variables by"""
            self.coefs = coefs

        def predict(self, input_data):
            return np.array([sum([self.coefs[i]*data[i] for i in range(len(data))]) > 20 for data in input_data])

    # For simplicity, our data has 4 input variables: one really, one kinda, and one slightly important, and one useless
    def population_def(x, y, z, w): return int(3*x + 2*y + w > 20)
    fake_model_input = np.random.randint(0, 10, size=(1000, 4))
    fake_model_output = [population_def(*data_point)
                         for data_point in fake_model_input]

    model = FakeModel(3, 2, 0, 1)

    classes = [0, 1]

    # other parameters to set
    npermute = 30  # definitely good enough. You could probably get away with 5
    subsamples = 0.5  # usually way more than sufficient. Often 0.2 is fine
    score_fn = None  # deliberately allow to become accuracy
    optimization = 'minimize'  # select the variable which most drastically affects accuracy
    # nimportant_variables = 3 # no need to set this
    share_vars = False  # not testing this right now
    njobs = 3  # just so I don't kill my machine entirely

    results = permutation_selection_variable_importances(model, classes, fake_model_input, fake_model_output, npermute,
                                                         subsamples=subsamples, score_fn=score_fn, optimization=optimization, share_vars=share_vars, njobs=njobs)
    statement = "Permutation selection will return an ordering of the most important variables along with their scores"
    if results[-1][0] != [0, 1, 3, 2]:
        print(
            statement, "Expected: results[-1][0] = [0, 1, 3, 2]", "Received:", results[-1][0])
        print("Entire result:", results)
    else:
        print("Tests passed!")
