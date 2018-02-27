"""This is a standalone set of functions which compute the importances for each variable in a dataset, according to a
particular model using multiprocessing.

The main method is the compute_variable_importances function.

This file contains two different versions for computing the importance, one which tries to share memory between threads
and one which does not. The "windows" version does not share memory (this is a consequence of the fact that forking is a
Unix only thing). This version should work for any OS and will also likely work for Python 3.x. It is more memory
greedy, but may also be somewhat faster.

Additionally, the non-greedy method assumes that the input variables are doubles (c_double) and the outputs are either
strings (c_char_p) or doubles. If this is not the case, you can always use the memory greedy version or import the
correct type from the ctypes module.
"""
import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocessing.sharedctypes import RawArray
from ctypes import c_double, c_char_p
import sys
from hagelslag.evaluation.MulticlassContingencyTable import MulticlassContingencyTable

# These are global variables which are inherited by compute_score_for_column
inputs_share = None
outputs_share = None
original_predictions_share = None


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
    c, model, classes, npermute, subsamples, score_fn = args
    global inputs_share
    global outputs_share
    global original_predictions_share
    return compute_score_for_column(c, model, classes, npermute, subsamples, inputs_share, outputs_share,
                                    original_predictions_share, score_fn)


def compute_score_for_column(c, model, classes, npermute, subsamples, inputs, outputs, original_predictions, score_fn):
    """This function is mapped over all of the columns of the testing_data and a score computed for each

    :param c: The index of this column
    :param classes: A set of possible labels for (all) the data
    :param model: The model which is to be evaluated. Must have a predict method
    :param npermute: number of bootstrapping runs (number of times to permute each column)
    :param subsamples: number of elements to sample (with replacement) per bootstrap. If between 0 and 1, this is
     treated as a fraction of the number of total number of events (e.g. 0.5 means half the number of events). If not
     specified, subsampling will not be used and the entire data will be used (without replacement)
    :param inputs: testing data inputs to the model (after any preprocessing)
    :param outputs: true labels of the testing data
    :param original_predictions: predictions of the model before any permutation
    :param score_fn: a function of the form (original_predictions, new_predictions, truths) -> float.
    :returns: A (possibly vector) score for this column as a pair (possibly of vectors) of mean and std of score"""

    # npermute times per column
    scores_for_col = list()
    for b in range(npermute):
        # Permute the column and reassemble the inputs using the permuted column
        permuted_col = np.random.permutation(inputs[:, c])
        permuted_inputs = np.copy(inputs)  # Clone the array, making a new(!) one
        permuted_inputs[:, c] = permuted_col

        if subsamples is not None:

            # Subsample the inputs and outputs and permuted inputs (with replacement)
            subsampled_idxs = np.random.choice(range(len(outputs)), subsamples)
            subsampled_permuted_inputs = [permuted_inputs[idx] for idx in subsampled_idxs]
            subsampled_outputs = [outputs[idx] for idx in subsampled_idxs]

            # Predict on the subsampled, permuted inputs
            predictions = model.predict(subsampled_permuted_inputs)
            subsampled_original_predictions = [original_predictions[idx] for idx in subsampled_idxs]

            # Score the model compared to the permuted model
            score = score_fn(subsampled_original_predictions, predictions, subsampled_outputs, classes)
        else:
            # Instead of subsampling, use the entire dataset without replacement
            predictions = model.predict(permuted_inputs)

            # Score the model compared to the permuted model
            score = score_fn(original_predictions, predictions, outputs, classes)
        scores_for_col.append(score)

    return np.average(scores_for_col, axis=0), np.std(scores_for_col, axis=0)


def compute_variable_importances(model, classes, testing_input, testing_output, npermute, subsamples=None,
                                 score_fn=None, zscores=False, njobs=None, share_vars=True):
    """Determines the variable importances for an estimator for a particular testing set

    :param model: The TrainedModel object to be scored
    :param classes: A set of possible labels for (all) the data
    :param testing_input: testing data inputs to the model (after any preprocessing)
    :param testing_output: true labels of the testing data
    :param npermute: number of times to permute each column
    :param subsamples: number of elements to sample (with replacement) per bootstrap. If between 0 and 1, this is
     treated as a fraction of the number of total number of events (e.g. 0.5 means half the number of events). If not
     specified, subsampling will not be used and the entire data will be used (without replacement)
    :param score_fn: a function of the form (original_predictions, new_predictions, truths) -> float. Defaults to
    an accuracy scorer.
    :param zscores: a boolean for whether to return the zscores of the variables with regards to each other. If true,
    will return zscores for each variable based on the mean of the mean for each variable score. Defaults to false.
    :param njobs: number of process (threads) to run this process in. If not specified, defaults to the number of cpus
    minus one.
    :param share_vars: Set to true to try to share memory between threads. Defaults to true. Will override to false on
    windows.
    :return: If zscores is False:
        An array of pairs of (possibly vector) scores, i.e. [(mean_score, std_score) for var in variables]
             If zscores is True:
        An array of individual scores, corresponding to the zscore of the 'mean score' for each variable.
        In this case, the 'mean score' is the average of the (possibly vector) of means returned for each variable."""

    if sys.platform == "win32":
        print("OS is windows. Using memory greedy version")
        share_vars = False

    if not share_vars:
        return compute_variable_importances_windows(model, classes, testing_input, testing_output,
                                                    npermute, subsamples, score_fn)

    # If subsamples is less than 1, treat it as a fraction of the total number of events
    if subsamples is not None and (0 < subsamples <= 1):
        subsamples = int(len(testing_output) * subsamples)
    # Check if the score_fn is None
    if score_fn is None:
        score_fn = accuracy_scorer

    # Predict on the original inputs for a baseline
    original_predictions = model.predict(testing_input)

    # ---- Prepackage the shared vars to the shared_variables_for_concurrency (shared_vars) module ----
    global inputs_share
    global outputs_share
    global original_predictions_share
    # Since the arrays are very large, we put them into shared memory
    # The multi-dimensional array is tricky
    # First, we flatten the array, then put it in memory, then grab it back out using numpy and then reshape it
    inputs_flattened = np.asarray(testing_input).flatten()  # flatten the inputs array
    inputs_shape = np.asarray(testing_input).shape
    inputs_share_base = RawArray(c_double, inputs_flattened)  # array of doubles (not write safe)
    inputs_share_np = np.frombuffer(inputs_share_base)  # shares memory with above, but is numpy array
    inputs_share = inputs_share_np.reshape(inputs_shape)  # reshape back to the way we expect
    # The single-dimensional arrays are simpler
    if isinstance(testing_output[0], str):
        # The outputs are strings
        outputs_share = RawArray(c_char_p, testing_output)  # array of strings (not write safe)
        original_predictions_share = RawArray(c_char_p, original_predictions)  # same as outputs
    else:
        # Assume the outputs are doubles
        outputs_share = RawArray(c_double, testing_output)  # array of doubles (not write safe)
        original_predictions_share = RawArray(c_double, original_predictions)  # same as outputs
    # ---- the three *_share arrays are now located in shared memory ----

    # ---- Construct the pool and parallelize! ----
    if njobs is None:
        njobs = cpu_count() - 1  # Minus 1 so that we don't completely overload everything
    pool = Pool(processes=njobs)

    # Compute the score for each column in parallel
    scores = pool.map(compute_score_for_column_unpack_share, [(i, model, classes, npermute, subsamples, score_fn)
                                                              for i in range(len(testing_input[0]))])

    # Clean up
    pool.close()
    pool.join()

    if zscores:
        # grab only the means (the average will convert the vector scores to scalar scores)
        means = [np.average(mean) for (mean, std) in scores]
        # Compute the z-scores for the variables (considered against each other)
        mean = np.average(means)
        std = np.std(means)
        # z-score(x) = (x - mean) / std
        return - (mean - means) / std
    else:
        # Otherwise just return the mean and std scores for each variable
        return scores


def compute_variable_importances_windows(model, classes, testing_input, testing_output, npermute, subsamples=None,
                                         score_fn=None, zscores=False, njobs=None):
    """Determines the variable importances for an estimator for a particular testing set

    :param model: The TrainedModel object to be scored
    :param classes: A set of possible labels for (all) the data
    :param testing_input: testing data inputs to the model (after any preprocessing)
    :param testing_output: true labels of the testing data
    :param npermute: number of bootstrapping runs (number of times to permute each column)
    :param subsamples: number of elements to sample (with replacement) per bootstrap. If between 0 and 1, this is
     treated as a fraction of the number of total number of events (e.g. 0.5 means half the number of events). If not
     specified, subsampling will not be used and the entire data will be used (without replacement)
    :param score_fn: a function of the form (original_predictions, new_predictions, truths) -> float. Defaults to
     an accuracy scorer.
    :param zscores: a boolean for whether to return the zscores of the variables with regards to each other. If true,
    will return zscores for each variable based on the mean of the mean for each variable score. Defaults to false.
    :param njobs: number of process (threads) to run this process in. If not specified, defaults to the number of cpus
    minus one.
    :return: If zscores is False:
        An array of pairs of (possibly vector) scores, i.e. [(mean_score, std_score) for var in variables]
             If zscores is True:
        An array of individual scores, corresponding to the zscore of the 'mean score' for each variable.
        In this case, the 'mean score' is the average of the (possibly vector) of means returned for each variable."""

    # If subsamples is less than 1, treat it as a fraction of the total number of events
    if subsamples is not None and (0 < subsamples <= 1):
        subsamples = int(len(testing_output) * subsamples)
    # Check if the score_fn is None
    if score_fn is None:
        score_fn = accuracy_scorer

    # Predict on the original inputs for a baseline
    original_predictions = model.predict(testing_input)

    # ---- Construct the pool and parallelize! ----
    if njobs is None:
        njobs = cpu_count() - 1  # Minus 1 so that we don't completely overload everything
    pool = Pool(processes=njobs)

    # Compute the score for each column in parallel
    scores = pool.map(compute_score_for_column_unpack,
                      [(i, model, classes, npermute, subsamples, testing_input, testing_output,
                        original_predictions, score_fn) for i in range(len(testing_input[0]))])

    # Clean up
    pool.close()
    pool.join()

    if zscores:
        # grab only the means (the average will convert the vector scores to scalar scores)
        means = [np.average(mean) for (mean, std) in scores]
        # Compute the z-scores for the variables (considered against each other)
        mean = np.average(means)
        std = np.std(means)
        # z-score(x) = (x - mean) / std
        return - (mean - means) / std
    else:
        # Otherwise just return the mean and std scores for each variable
        return scores


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
            matrix[i, j] = [p == c1 and t == c2 for p, t in zip(predictions, truths)].count(True)
    return matrix


def accuracy_scorer(original_predictions, new_predictions, truths, classes):
    """Scores the model based on the accuracy of the old and new predictions

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: NOT USED
    :returns: The accuracy of the original_predictions - the accuracy of the new_predictions
    """
    original_accuracy = float([original_predictions[idx] == truths[idx] for idx in range(len(truths))].count(True)) / \
        float(len(truths))

    new_accuracy = float([new_predictions[idx] == truths[idx] for idx in range(len(truths))].count(True)) / \
        float(len(truths))

    return original_accuracy - new_accuracy


def bias_scorer(original_predictions, new_predictions, truths, classes):
    """Scores the change in biases for each possible class of the data, returning a vector

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a vector (numpy array) of the change in biases for each class
    """
    print "We got here"
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)

    biases = np.zeros((len(classes), ), dtype=np.float32)
    for i, c in enumerate(classes):
        old_hits = old_cm[i, i]
        new_hits = new_cm[i, i]
        # Total number of predictions of storms of this type
        old_prediction_positive = np.sum(old_cm[i, :])
        new_prediction_positive = np.sum(new_cm[i, :])
        old_false_alarms = old_prediction_positive - old_hits
        new_false_alarms = new_prediction_positive - new_hits
        # Total number of storms of this type
        old_condition_positive = np.sum(old_cm[:, i])
        new_condition_positive = np.sum(new_cm[:, i])
        old_misses = old_condition_positive - old_hits
        new_misses = new_condition_positive - new_hits

        old_bias = old_hits + old_false_alarms / old_hits + old_misses
        new_bias = new_hits + new_false_alarms / new_hits + new_misses
        biases[i] = old_bias - new_bias
    return biases


def hitrate_scorer(original_predictions, new_predictions, truths, classes):
    """Scores the change in hitrates for each possible class of the data, returning a vector

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a vector (numpy array) of the change in hitrates for each class
        """
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)

    hitrates = np.zeros((len(classes), ), dtype=np.float32)
    for i, c in enumerate(classes):
        old_hits = old_cm[i, i]
        new_hits = new_cm[i, i]
        # Total number of storms of this type
        old_condition_positive = np.sum(old_cm[:, i])
        new_condition_positive = np.sum(new_cm[:, i])

        # Ensure that 0/0 -> hitrate of 0
        old_hr = 0 if (old_condition_positive == 0) else old_hits / old_condition_positive
        new_hr = 0 if (new_condition_positive == 0) else new_hits / new_condition_positive

        hitrates[i] = old_hr - new_hr
    return hitrates

# UNTESTED!!!
def false_alarm_ratio_scorer(original_predictions, new_predictions, truths, classes):
    """Scores the change in false alarm ratio for each possible class of the data, returning a vector

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a vector (numpy array) of the change in false alarm ratio for each class
    """
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)

    fars = np.zeros((len(classes),), dtype=np.float32)
    for i, c in enumerate(classes):
        old_hits = old_cm[i, i]
        new_hits = new_cm[i, i]
        # Total number of predictions of this type
        old_prediction_positive = np.sum(old_cm[i, :])
        new_prediction_positive = np.sum(new_cm[i, :])

        # Ensure that 0/0 -> false alarm ratio of 1
        old_far = 0 if (old_prediction_positive == 0) else (old_prediction_positive - old_hits) / old_prediction_positive
        new_far = 0 if (new_prediction_positive == 0) else (new_prediction_positive - new_hits) / new_prediction_positive

        fars[i] = old_far - new_far
    return fars

# UNTESTED!!!
def probability_of_false_detection_scorer(original_predictions, new_predictions, truths, classes):
    """Scores the change in probability of false detection for each possible class of the data, returning a vector

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a vector (numpy array) of the change in probability of false detection for each class
    """
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)

    pofds = np.zeros((len(classes),), dtype=np.float32)
    for i, c in enumerate(classes):
        old_hits = old_cm[i, i]
        new_hits = new_cm[i, i]
        # Total number of storms
        total_storm_count = len(truths)
        old_condition_positive = np.sum(old_cm[:, i])
        old_prediction_positive = np.sum(old_cm[i, :])
        old_correct_negatives = total_storm_count - old_prediction_positive - old_condition_positive + old_hits
        new_condition_positive = np.sum(new_cm[:, i])
        new_prediction_positive = np.sum(new_cm[i, :])
        new_correct_negatives = total_storm_count - new_prediction_positive - new_condition_positive + new_hits

        old_false_alarms = old_prediction_positive - old_hits
        new_false_alarms = new_prediction_positive - new_hits

        old_pofd = old_false_alarms / (old_false_alarms + old_correct_negatives)
        new_pofd = new_false_alarms / (new_false_alarms + new_correct_negatives)
        pofds[i] = old_pofd - new_pofd
    return pofds

# UNTESTED!!!
def success_ratio_scorer(original_predictions, new_predictions, truths, classes):
    """Scores the change in success ratio for each possible class of the data, returning a vector

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a vector (numpy array) of the change in success ratio for each class
    """
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)

    srs = np.zeros((len(classes),), dtype=np.float32)
    for i, c in enumerate(classes):
        old_hits = old_cm[i, i]
        new_hits = new_cm[i, i]
        old_prediction_positive = np.sum(old_cm[i, :])
        new_prediction_positive = np.sum(new_cm[i, :])

        old_sr = old_hits / old_prediction_positive
        new_sr = new_hits / new_prediction_positive
        srs[i] = old_sr - new_sr
    return srs

# UNTESTED!!!
def threat_scorer(original_predictions, new_predictions, truths, classes):
    """Scores the change in threat score for each possible class of the data, returning a vector

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a vector (numpy array) of the change in threat score for each class
    """
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)

    tss = np.zeros((len(classes),), dtype=np.float32)
    for i, c in enumerate(classes):
        old_hits = old_cm[i, i]
        new_hits = new_cm[i, i]
        old_condition_positive = np.sum(old_cm[:, i])
        old_prediction_positive = np.sum(old_cm[i, :])
        new_condition_positive = np.sum(new_cm[:, i])
        new_prediction_positive = np.sum(new_cm[i, :])
        # Events = hits + false alarms + misses
        old_events = old_prediction_positive + old_condition_positive - old_hits
        new_events = new_prediction_positive + new_condition_positive - new_hits

        old_ts = old_hits / old_events
        new_ts = new_hits / new_events
        tss[i] = old_ts - new_ts
    return tss


# UNTESTED!!
def gerrity_skill_scorer(original_predictions, new_predictions, truths, classes):
    """Determines the change in gerrity skill score, returning a scalar

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a single value for the change in gerrity skill score
    """
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)
    # make into contingency tables
    old_table = MulticlassContingencyTable(table=old_cm, n_classes=len(classes), class_names=classes)
    new_table = MulticlassContingencyTable(table=new_cm, n_classes=len(classes), class_names=classes)

    return old_table.gerrity_skill_score() - new_table.gerrity_skill_score()


def peirce_skill_scorer(original_predictions, new_predictions, truths, classes):
    """Determines the change in peirce skill score, returning a scalar

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a single value for the change in peirce skill score
    """
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)
    # make into contingency tables
    old_table = MulticlassContingencyTable(table=old_cm, n_classes=len(classes), class_names=classes)
    new_table = MulticlassContingencyTable(table=new_cm, n_classes=len(classes), class_names=classes)

    return old_table.peirce_skill_score() - new_table.peirce_skill_score()


def heidke_skill_scorer(original_predictions, new_predictions, truths, classes):
    """Determines the change in heidke skill score, returning a scalar

    :param original_predictions: The predictions of the model on the unpermuted data
    :param new_predictions: The predictions of the model after the permutation
    :param truths: The true labels of these data
    :param classes: an ordered set for the label possibilities
    :returns: a single value for the change in heidke skill score
    """
    # get the confusion matrices
    old_cm = confusion_matrix(original_predictions, truths, classes)
    new_cm = confusion_matrix(new_predictions, truths, classes)
    # make into contingency tables
    old_table = MulticlassContingencyTable(table=old_cm, n_classes=len(classes), class_names=classes)
    new_table = MulticlassContingencyTable(table=new_cm, n_classes=len(classes), class_names=classes)

    return old_table.heidke_skill_score() - new_table.heidke_skill_score()


# def notes():
#     total_storm_count = len(truths)
#     # Total number of storms of this type
#     condition_positive = sum(cm[:, l])
#     # Total number of storms of not this type
#     condition_negative = total_storm_count - condition_positive
#     # Total number of predictions of storms of this type
#     prediction_positive = sum(cm[l, :])
#     # Total number of predictions of storms of other types
#     prediction_negative = total_storm_count - prediction_positive
#
#     # Correct predictions of this storm
#     hits = cm[l, l]
#     # Predicted this storm, but was wrong
#     false_alarms = prediction_positive - hits
#     # Predicted not this storm, but it was
#     misses = condition_positive - hits
#     # Correctly predicted not this storm
#     correct_negatives = total_storm_count - hits - false_alarms - misses