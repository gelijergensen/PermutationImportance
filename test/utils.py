"""These are just a handful of functions which are useful only for helping run
certain tests

Note: These tests aren't automatically run by pytest, but can be manually run
by calling pytest on this file"""

import numpy as np


def make_test_data():
    """This is a useful tool to help with making a dataset where the relative
    ranks of the variable importances is known"""
    class_0 = np.random.uniform(size=(250, 3)) * \
        np.array([4, 2, 1]) + np.array([-4, 9, 1])
    class_1 = np.random.uniform(size=(250, 3)) * \
        np.array([4, 2, 1]) + np.array([-5, 7, -1])
    training_inputs = np.concatenate((class_0[:200], class_1[:200]), axis=0)
    scoring_inputs = np.concatenate((class_0[200:], class_1[200:]), axis=0)
    training_outputs = np.array([(0 if i < 200 else 1) for i in range(400)])
    scoring_outputs = np.array([(0 if i < 50 else 1) for i in range(100)])
    indices = np.random.permutation(400)
    training_inputs = training_inputs[indices]
    training_outputs = training_outputs[indices]
    indices = np.random.permutation(100)
    scoring_inputs = scoring_inputs[indices]
    scoring_outputs = scoring_outputs[indices]

    return (training_inputs, training_outputs), (scoring_inputs, scoring_outputs)


def test_make_test_data():
    (training_inputs, training_outputs), (scoring_inputs,
                                          scoring_outputs) = make_test_data()

    assert len(training_inputs) == len(training_outputs)
    assert len(training_inputs) == 400
    assert len(scoring_inputs) == len(scoring_outputs)
    assert len(scoring_inputs) == 100
    assert training_inputs.shape[1] == scoring_inputs.shape[1]
    assert len(np.unique(scoring_outputs)) == len(np.unique(training_outputs))
    assert len(np.unique(scoring_outputs)) == 2


def make_proba_test_data():
    """This is a useful tool to help with making a dataset where the relative
    ranks of the variable importances is known"""
    class_0 = np.random.uniform(size=(250, 3)) * \
        np.array([4, 2, 1]) + np.array([-4, 9, 1])
    class_1 = np.random.uniform(size=(250, 3)) * \
        np.array([4, 2, 1]) + np.array([-5, 7, -1])
    training_inputs = np.concatenate((class_0[:200], class_1[:200]), axis=0)
    scoring_inputs = np.concatenate((class_0[200:], class_1[200:]), axis=0)
    training_outputs = np.array([(0 if i < 200 else 1) for i in range(400)])
    scoring_outputs = np.array([(0 if i < 50 else 1) for i in range(100)])
    indices = np.random.permutation(400)
    training_inputs = training_inputs[indices]
    training_outputs = training_outputs[indices]
    indices = np.random.permutation(100)
    scoring_inputs = scoring_inputs[indices]
    scoring_outputs = np.stack(
        (scoring_outputs[indices], 1 - scoring_outputs[indices]), axis=-1)

    return (training_inputs, training_outputs), (scoring_inputs, scoring_outputs)


def test_make_proba_test_data():
    (training_inputs, training_outputs), (scoring_inputs,
                                          scoring_outputs) = make_proba_test_data()

    assert len(training_inputs) == len(training_outputs)
    assert len(training_inputs) == 400
    assert len(scoring_inputs) == len(scoring_outputs)
    assert len(scoring_inputs) == 100
    assert training_inputs.shape[1] == scoring_inputs.shape[1]
    assert len(np.unique(scoring_outputs)) == len(np.unique(training_outputs))
    assert len(np.unique(scoring_outputs)) == 2
    assert scoring_outputs.shape[1] == 2
