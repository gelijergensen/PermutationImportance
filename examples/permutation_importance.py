from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from PermutationImportance import sklearn_permutation_importance

# Separate out the last 20% for scoring data
iris = load_iris(return_X_y=False)
inputs = iris.get('data')
outputs = iris.get('target')
predictor_names = iris.get('feature_names')
training_inputs = inputs[:int(0.8 * len(inputs))]
training_outputs = outputs[:int(0.8 * len(outputs))]
scoring_inputs = inputs[int(0.8 * len(inputs)):]
scoring_outputs = outputs[int(0.8 * len(outputs)):]

# Train a quick neural net on the data
model = MLPClassifier(solver='lbfgs')
model.fit(training_inputs, training_outputs)

# Package the data into the right shape
scoring_data = (scoring_inputs, scoring_outputs)

# Use the sklearn_permutation_importance to compute importances
result = sklearn_permutation_importance(
    model, scoring_data, accuracy_score, 'min', variable_names=predictor_names)

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
for cntxt, res in result:
    print("Context: %r" % cntxt)
    print("Result: %r" % res)
