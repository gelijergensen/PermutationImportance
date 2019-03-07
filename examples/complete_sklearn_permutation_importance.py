from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from PermutationImportance import sklearn_permutation_importance
from PermutationImportance.metrics import peirce_skill_score

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

# ----------- Version to use when only wanting singlepass results --------------
# Use the sklearn_permutation_importance to compute importances
result = sklearn_permutation_importance(
    # argmin_of_mean handles bootstrapped metrics
    model, scoring_data, peirce_skill_score, 'argmin_of_mean',
    variable_names=predictor_names,
    # sample (with replacement) 1*(number of samples) 5 times to compute metric distribution
    # nbootstrap should typically be 1000, but this is kept small here for printing purposes
    nbootstrap=5, subsample=1,
    # only perform for the very top predictor (effectively means only compute singlepass results)
    nimportant_vars=1)

# Get the Breiman-like singlepass results
print("Singlepass")
singlepass = result.retrieve_singlepass()
for predictor in singlepass.keys():
    rank, score = singlepass[predictor]
    print("Predictor: %s, Rank: %i, Score: %r" % (predictor, rank, score))
# Get the Lakshmanan-like multipass results
print("Multipass. This should only have 1 item and be not very useful")
multipass = result.retrieve_multipass()
for predictor in multipass.keys():
    rank, score = multipass[predictor]
    print("Predictor: %s, Rank: %i, Score: %r" % (predictor, rank, score))
# Iterate over the (context, result) pairs
for i, (cntxt, res) in enumerate(result):
    print("Context %i: %r" % (i, cntxt))
    print("Result %i: %r" % (i, res))
# ------------------------------------------------------------------------------

# ----------- Version to use when wanting multipass results --------------------
# Use the sklearn_permutation_importance to compute importances
result = sklearn_permutation_importance(
    # argmin_of_mean handles bootstrapped metrics
    model, scoring_data, peirce_skill_score, 'argmin_of_mean',
    variable_names=predictor_names,
    # sample (with replacement) 1*(number of samples) 5 times to compute metric distribution
    # nbootstrap should typically be 1000, but this is kept small here for printing purposes
    nbootstrap=5, subsample=1,
    nimportant_vars=8)  # only perform for the top 8 predictors

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
