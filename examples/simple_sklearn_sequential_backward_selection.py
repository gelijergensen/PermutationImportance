from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from PermutationImportance import sklearn_sequential_backward_selection

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
training_data = (training_inputs, training_outputs)
scoring_data = (scoring_inputs, scoring_outputs)

# Use the sklearn_sequential_backward_selection to compute importances
result = sklearn_sequential_backward_selection(
    model,  training_data, scoring_data, accuracy_score, 'max',
    variable_names=predictor_names)

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
