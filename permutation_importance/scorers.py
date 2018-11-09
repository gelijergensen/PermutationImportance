"""These functions can all be used as the "score_fn" parameter in the main permutation selection variable importance 
call. You can also use your own"""

import numpy as np

from .utilities import confusion_matrix, MulticlassContingencyTable


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
