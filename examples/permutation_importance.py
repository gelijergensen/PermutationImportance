from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from PermutationImportance import sklearn_permutation_importance
from sklearn.metrics import accuracy_score

# Separate out the last 20% for scoring data
iris = load_iris(return_X_y=False)
inputs = iris.get('data')
outputs = iris.get('target')
predictor_names = iris.get('feature_names')
training_inputs = inputs[:int(0.8 * len(inputs))]
training_outputs = outputs[:int(0.8 * len(outputs))]
scoring_inputs = inputs[int(0.8 * len(inputs)):]
scoring_outputs = outputs[int(0.8 * len(outputs)):]

# Train a quick random forest model on the data
forest = RandomForestClassifier()
forest.fit(training_inputs, training_outputs)

# Use the sklearn_permutation_importance to compute importances
result = sklearn_permutation_importance(
    forest, (scoring_inputs, scoring_outputs), accuracy_score, 'argmin', variable_names=predictor_names)
print(result)
