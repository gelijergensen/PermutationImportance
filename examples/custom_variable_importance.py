"""An example of several different custom components that PermutationImportance
allows. Here, we are attempting to look at the predictors which are impacting
the forecasting bias of the model. To do this, we first construct a custom 
metric ``bias_score``, and also construct an optimization strategy which selects
the index of the predictor which induces the least bias in the model
``argmin_of_ratio_from_unity``. Additionally, rather than using a typical method
for evaluating this (like permutation importance), we develop our own custom
method, "zero-filled importance", which operates like permutation importance,
but rather than permuting the values of a predictor to destroy the relationship
between the predictor and the target, it simply sets all of the values of the 
predictor to 0 (which could have some interesting, undesired side-effects). This
is done by constructing a custom selection strategy 
``ZeroFilledSelectionStrategy`` and using this to build both the method-specific
(``zero_filled_importance``) and model-based 
(``sklearn_zero_filled_importance``) versions of the predictor importance 
method.

As a side note, notice below that we leverage the utilities of 
PermutationImportance.sklearn_api to help build the model-based version in a way
which also allows us to even do bootstrapping!
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

from PermutationImportance.abstract_runner import abstract_variable_importance
from PermutationImportance.metrics import _get_contingency_table
from PermutationImportance.selection_strategies import SelectionStrategy
from PermutationImportance.sklearn_api import score_trained_sklearn_model, score_trained_sklearn_model_with_probabilities
from PermutationImportance.utils import get_data_subset, make_data_from_columns


# Example of a custom metric / evaluation_fn
def bias_score(truths, predictions, classes=None):
    """Determines the Forecast Bias of a model, returning a scalar. See 
    `here <http://www.cawcr.gov.au/projects/verification/#Methods_for_dichotomous_forecasts>`_
    for more details on the bias score.

    To handle multi-class predictions, this takes the AVERAGE bias score for
    each of the classes independently

    :param truths: The true labels of these data
    :param predictions: The predictions of the model
    :param classes: an ordered set for the label possibilities. If not given,
        will be deduced from the truth values
    :returns: a single value for the gerrity score
    """
    table = _get_contingency_table(truths, predictions, classes)
    biases = np.zeros((len(table), ))
    for i in range(len(biases)):
        biases[i] = np.sum(table[i, :], dtype='float32') / \
            (np.sum(table[:, i], dtype='float32') +
             1e-16)  # epsilon for numerical stability
    return np.average(biases)


def _ratio_from_unity(score):
    """Returns the smaller of (score, 1/score). This can be thought of as a 
    score in [0, 1], where 1 is the best and 0 is the worst

    :param score: either a single value or an array of values, in which case
        the mean is taken first
    :returns: a single scalar in [0, 1], where 1 is best"""
    mean_score = np.average(score)
    if mean_score > 1:
        return 1.0 / float(mean_score)
    else:
        return float(mean_score)


# Example of a custom optimization strategy
def argmin_of_ratio_from_unity(scores):
    """Returns the argmin of each of the "ratios from unity". This has the 
    effect of returning the index of the predictor which caused the worst bias

    NOTE: This could have also been done with
    :class:`PermutationImportance.scoring_strategies.indexer_of_converter```(np.armin, _ratio_from_unity)``
    """
    return np.argmin([_ratio_from_unity(score) for score in scores])


# Example of a custom selection strategy
class ZeroFilledSelectionStrategy(SelectionStrategy):
    """"Zero-Filled Importance" is a made-up predictor importance method which 
    tests all predictors which are not yet considered importance by setting all 
    of the values of that column to be zero. This destroys the information 
    present in the column much in the same way as Permutation Importance, but 
    may have weird side-effects because zero is not necessarily a neutral value 
    (e.g. Temperature in kelvins). The shape of the training data will remain
    constant, but many columns may contain only 0's."""

    name = "Zero Filled Importance"

    def __init__(self, training_data, scoring_data, num_vars, important_vars):
        """Initializes the object by storing the data and keeping track of other
        important information

        :param training_data: (training_inputs, training_outputs)
        :param scoring_data: (scoring_inputs, scoring_outputs)
        :param num_vars: integer for the total number of variables
        :param important_vars: a list of the indices of variables which are 
            already considered important
        """
        super(ZeroFilledSelectionStrategy, self).__init__(
            training_data, scoring_data, num_vars, important_vars)
        # Also initialize the zero data
        scoring_inputs, __ = self.scoring_data
        self.zero_scoring_inputs = np.zeros(scoring_inputs.shape)

    def generate_datasets(self, important_variables):
        """Check each of the non-important variables. Dataset has columns which
        are important shuffled. Notice that although we could modify the 
        training data as well, we are going to assume that this behaves like
        Permutation Importance, in which case the training data will always be
        empty

        :returns: (training_data, scoring_data)
        """
        scoring_inputs, scoring_outputs = self.scoring_data
        complete_scoring_inputs = make_data_from_columns(
            [get_data_subset(self.zero_scoring_inputs if i in important_variables else scoring_inputs, None, [i]) for i in range(self.num_vars)])

        return self.training_data, (complete_scoring_inputs, scoring_outputs)


# Example of the Method-Specific custom predictor importance
def zero_filled_importance(scoring_data, scoring_fn, scoring_strategy, variable_names=None, nimportant_vars=None, njobs=1):
    """Performs "zero-filled importance" over data given a particular
    set of functions for scoring and determining optimal variables

    :param scoring_data: a 2-tuple ``(inputs, outputs)`` for scoring in the
        ``scoring_fn``
    :param scoring_fn: a function to be used for scoring. Should be of the form
        ``(training_data, scoring_data) -> some_value``
    :param scoring_strategy: a function to be used for determining optimal
        variables. Should be of the form ``([some_value]) -> index``
    :param variable_names: an optional list for variable names. If not given,
        will use names of columns of data (if pandas dataframe) or column
        indices
    :param nimportant_vars: number of variables to compute multipass importance
        for. Defaults to all variables
    :param njobs: an integer for the number of threads to use. If negative, will
        use ``num_cpus + njobs``. Defaults to 1
    :returns: :class:`PermutationImportance.result.ImportanceResult` object 
        which contains the results for each run
    """
    # We don't need the training data, so pass empty arrays to the abstract runner
    return abstract_variable_importance((np.array([]), np.array([])), scoring_data, scoring_fn, scoring_strategy, ZeroFilledSelectionStrategy, variable_names=variable_names, nimportant_vars=nimportant_vars, njobs=njobs)


# Example of a Model-Based custom predictor importance
def sklearn_zero_filled_importance(model, scoring_data, evaluation_fn, scoring_strategy, variable_names=None, nimportant_vars=None, njobs=1, nbootstrap=1, subsample=1, **kwargs):
    """Performs "zero-filled importance" for a particular model, 
    ``scoring_data``, ``evaluation_fn``, and strategy for determining optimal 
    variables

    :param model: a trained sklearn model
    :param scoring_data: a 2-tuple ``(inputs, outputs)`` for scoring in the
        ``scoring_fn``
    :param evaluation_fn: a function which takes the deterministic or 
        probabilistic model predictions and scores them against the true 
        values. Must be of the form ``(truths, predictions) -> some_value``
        Probably one of the metrics in 
        :mod:`PermutationImportance.metrics` or 
        `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
    :param scoring_strategy: a function to be used for determining optimal
        variables. Should be of the form ``([some_value]) -> index``
    :param variable_names: an optional list for variable names. If not given,
        will use names of columns of data (if pandas dataframe) or column
        indices
    :param nimportant_vars: number of variables to compute multipass importance
        for. Defaults to all variables
    :param njobs: an integer for the number of threads to use. If negative, will
        use ``num_cpus + njobs``. Defaults to 1
    :param nbootstrap: number of times to perform scoring on each variable.
        Results over different bootstrap iterations are averaged. Defaults to 1
    :param subsample: number of elements to sample (with replacement) per
        bootstrap round. If between 0 and 1, treated as a fraction of the number
        of total number of events (e.g. 0.5 means half the number of events).
        If not specified, subsampling will not be used and the entire data will
        be used (without replacement)
    :param kwargs: all other kwargs will be passed on to the ``evaluation_fn``
    :returns: :class:`PermutationImportance.result.ImportanceResult` object 
        which contains the results for each run
    """
    # Check if the data is probabilistic
    if len(scoring_data[1].shape) > 1 and scoring_data[1].shape[1] > 1:
        # Take advantage of the tools in PermutationImportance.sklearn_api to
        # build a probabilistic scoring function from the evaluation function
        scoring_fn = score_trained_sklearn_model_with_probabilities(
            model, evaluation_fn, nbootstrap=nbootstrap, subsample=subsample, **kwargs)
    else:
        # Take advantage of the tools in PermutationImportance.sklearn_api to
        # build a deterministic scoring function from the evaluation function
        scoring_fn = score_trained_sklearn_model(
            model, evaluation_fn, nbootstrap=nbootstrap, subsample=subsample, **kwargs)
    return zero_filled_importance(scoring_data, scoring_fn, scoring_strategy, variable_names=variable_names, nimportant_vars=nimportant_vars, njobs=njobs)


# ----------------- Example Usage of Custom Method -----------------------------
"""
Here, our goal is to try and determine which predictors most drastically impact
the bias of a model. Notice that the "_ratio_from_unity" above basically acts
as a way to convert the bias to a more traditional score. Here, we also use a 
custom predictor importance method, "Zero-Filled Importance"
"""

# Separate out the last 20% for scoring data
breast_cancer = load_breast_cancer(return_X_y=False)
inputs = breast_cancer.get('data')
outputs = breast_cancer.get('target')
predictor_names = breast_cancer.get('feature_names')
training_inputs = inputs[:int(0.8 * len(inputs))]
training_outputs = outputs[:int(0.8 * len(outputs))]
scoring_inputs = inputs[int(0.8 * len(inputs)):]
scoring_outputs = outputs[int(0.8 * len(outputs)):]

# Train a quick forest on the data
model = RandomForestClassifier(n_estimators=100, max_depth=4)
model.fit(training_inputs, training_outputs)

# Package the data into the right shape
scoring_data = (scoring_inputs, scoring_outputs)

# Use the sklearn_zero_filled_importance to compute importances
result = sklearn_zero_filled_importance(
    model, scoring_data, bias_score, argmin_of_ratio_from_unity,
    variable_names=predictor_names,
    # Notice that we can use bootstrapping here thanks to the
    # PermutationImportance.sklearn_api tools for constructing a score function
    nbootstrap=5, subsample=1,  # nboostrap=1000 would be better
    nimportant_vars=None)  # perform for all predictors

# Get the Breiman-like singlepass results
print("Singlepass")
singlepass = result.retrieve_singlepass()
for predictor in singlepass.keys():
    rank, score = singlepass[predictor]
    print("Predictor: %s, Rank: %i, Score: %r" % (predictor, rank, score))
# Get the Lakshmanan-like multipass results
print("Multipass. This should have exactly 8 items")
multipass = result.retrieve_multipass()
for predictor in multipass.keys():
    rank, score = multipass[predictor]
    print("Predictor: %s, Rank: %i, Score: %r" % (predictor, rank, score))
# Iterate over the (context, result) pairs
for i, (cntxt, res) in enumerate(result):
    print("Context %i: %r" % (i, cntxt))
    print("Result %i: %r" % (i, res))
# ------------------------------------------------------------------------------
