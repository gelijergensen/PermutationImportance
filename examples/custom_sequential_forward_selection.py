from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from PermutationImportance import sequential_forward_selection

# Separate out the last 20% for scoring data
iris = load_iris(return_X_y=False)
inputs = iris.get('data')
outputs = iris.get('target')
predictor_names = iris.get('feature_names')
training_inputs = inputs[:int(0.8 * len(inputs))]
training_outputs = outputs[:int(0.8 * len(outputs))]
scoring_inputs = inputs[int(0.8 * len(inputs)):]
scoring_outputs = outputs[int(0.8 * len(outputs)):]


def score_model(training_data, scoring_data):
    """Custom function to use for scoring. Notice that because this is 
    sequential selection, we need to retrain the model each time

    :param training_data: (training_inputs, training_outputs)
    :param scoring_data: (scoring_inputs, scoring_outputs)
    """
    training_ins, training_outs = training_data
    scoring_ins, scoring_outs = scoring_data

    # Some model we are interested in
    model = MLPClassifier(solver='lbfgs')
    model.fit(training_ins, training_outs)

    return accuracy_score(scoring_outs, model.predict(scoring_ins))


# Package the data into the right shape
training_data = (training_inputs, training_outputs)
scoring_data = (scoring_inputs, scoring_outputs)

# Use the sequential_forward_selection to compute importances
result = sequential_forward_selection(
    training_data, scoring_data, score_model, 'max',
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
