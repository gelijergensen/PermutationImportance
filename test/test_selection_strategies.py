
import numpy as np
import pytest

from PermutationImportance.selection_strategies import SequentialForwardSelectionStrategy, SequentialBackwardSelectionStrategy, SelectionStrategy, PermutationImportanceSelectionStrategy


def test_selection_strategy():
    x = np.array([])
    strategy = SelectionStrategy((x, x), (x, x), 1, [])

    assert getattr(strategy, "name") == "Abstract Selection Strategy"

    with pytest.raises(NotImplementedError):
        next(iter(strategy))


def test_sfs_strategy():

    training_data = (np.random.rand(5, 3), np.random.rand(5, ))
    scoring_data = (np.random.rand(5, 3), np.random.rand(5, ))

    num_vars = training_data[0].shape[1]
    important_vars = [1]

    strategy = SequentialForwardSelectionStrategy(
        training_data, scoring_data, num_vars, important_vars)

    assert getattr(strategy, "name") == "Sequential Forward Selection"

    expected = [(0, (training_data[0][:, [1, 0]], training_data[1]), (scoring_data[0][:, [1, 0]], scoring_data[1])),
                (2, (training_data[0][:, [1, 2]], training_data[1]), (scoring_data[0][:, [1, 2]], scoring_data[1]))]
    for (exp_var, exp_train_data, exp_score_data), (res_var, res_train_data, res_score_data) in zip(expected, strategy):
        assert exp_var == res_var
        assert (exp_train_data[0] == res_train_data[0]).all()
        assert (exp_train_data[1] == res_train_data[1]).all()
        assert (exp_score_data[0] == res_score_data[0]).all()
        assert (exp_score_data[1] == res_score_data[1]).all()


def test_sbs_strategy():

    training_data = (np.random.rand(5, 3), np.random.rand(5, ))
    scoring_data = (np.random.rand(5, 3), np.random.rand(5, ))

    num_vars = training_data[0].shape[1]
    important_vars = [1]

    strategy = SequentialBackwardSelectionStrategy(
        training_data, scoring_data, num_vars, important_vars)

    assert getattr(strategy, "name") == "Sequential Backward Selection"

    expected = [(0, (training_data[0][:, [2]], training_data[1]), (scoring_data[0][:, [2]], scoring_data[1])),
                (2, (training_data[0][:, [0]], training_data[1]), (scoring_data[0][:, [0]], scoring_data[1]))]
    for (exp_var, exp_train_data, exp_score_data), (res_var, res_train_data, res_score_data) in zip(expected, strategy):
        assert exp_var == res_var
        assert (exp_train_data[0] == res_train_data[0]).all()
        assert (exp_train_data[1] == res_train_data[1]).all()
        assert (exp_score_data[0] == res_score_data[0]).all()
        assert (exp_score_data[1] == res_score_data[1]).all()


def test_permutation_strategy():

    training_data = (np.random.rand(5, 3), np.random.rand(5, ))
    scoring_data = (np.random.rand(5, 3), np.random.rand(5, ))

    num_vars = training_data[0].shape[1]
    important_vars = [1]

    strategy = PermutationImportanceSelectionStrategy(
        training_data, scoring_data, num_vars, important_vars)

    assert getattr(strategy, "name") == "Permutation Importance"

    shuffled_scoring_inputs = strategy.shuffled_scoring_inputs

    for i in range(shuffled_scoring_inputs.shape[1]):
        assert (np.unique(scoring_data[0][:, i]) == np.unique(
            shuffled_scoring_inputs[:, i])).all()

    expected = [(0, training_data, (np.column_stack([shuffled_scoring_inputs[:, 0], shuffled_scoring_inputs[:, 1], scoring_data[0][:, 2]]), scoring_data[1])),
                (2, training_data, (np.column_stack([scoring_data[0][:, 0], shuffled_scoring_inputs[:, 1], shuffled_scoring_inputs[:, 2]]), scoring_data[1]))]

    for (exp_var, exp_train_data, exp_score_data), (res_var, res_train_data, res_score_data) in zip(expected, strategy):
        assert exp_var == res_var
        assert (exp_train_data[0] == res_train_data[0]).all()
        assert (exp_train_data[1] == res_train_data[1]).all()
        assert (exp_score_data[0] == res_score_data[0]).all()
        assert (exp_score_data[1] == res_score_data[1]).all()
