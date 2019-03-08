from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from PermutationImportance import permutation_importance

# Separate out the last 20% for scoring data
iris = load_iris(return_X_y=False)
inputs = iris.get('data')
outputs = iris.get('target')
predictor_names = iris.get('feature_names')
training_inputs = inputs[:int(0.8 * len(inputs))]
training_outputs = outputs[:int(0.8 * len(outputs))]
scoring_inputs = inputs[int(0.8 * len(inputs)):]
scoring_outputs = outputs[int(0.8 * len(outputs)):]

# Some model we are interested in
model = MLPClassifier(solver='lbfgs')
model.fit(training_inputs, training_outputs)


def score_model(training_data, scoring_data):
    """Custom function to use for scoring. Notice that we are using a global
    model here, rather than just reassemble the model each time

    :param training_data: should be ignored for permutation importance
    :param scoring_data: (scoring_inputs, scoring_outputs)
    """
    scoring_ins, scoring_outs = scoring_data
    return accuracy_score(scoring_outs, model.predict(scoring_ins))


# Package the data into the right shape
scoring_data = (scoring_inputs, scoring_outputs)

# Use the permutation_importance to compute importances
result = permutation_importance(
    scoring_data, score_model, 'min', variable_names=predictor_names)

# Get the Breiman-like singlepass results
print("Singlepass")
singlepass = result.retrieve_singlepass()
for predictor in singlepass.keys():
    rank, score = singlepass[predictor]
    print("Predictor: %s, Rank: %i, Score: %f" % (predictor, rank, score))
# Get the Lakshmanan-like multipass results
print("Multipass")
multipass = result.retrieve_multipass()
for predictor in multipass.keys():
    rank, score = multipass[predictor]
    print("Predictor: %s, Rank: %i, Score: %f" % (predictor, rank, score))
# Iterate over the (context, result) pairs
for i, (cntxt, res) in enumerate(result):
    print("Context %i: %r" % (i, cntxt))
    print("Result %i: %r" % (i, res))
