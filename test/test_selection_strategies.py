
import numpy as np
import pytest

from src.selection_strategies import SequentialForwardSelectionStrategy, SequentialBackwardSelectionStrategy, SelectionStrategy


def test_selection_strategy():

    strategy = SelectionStrategy(None, None, None, None, None, None)

    assert getattr(strategy, "name") == "Abstract Selection Strategy"

    with pytest.raises(NotImplementedError):
        iter(strategy)


def test_sfs_strategy():

    training_data = (np.random.rand(5, 3), np.random.rand(5, ))
    scoring_data = (np.random.rand(5, 3), np.random.rand(5, ))

    subsample = 5
    bootstrap_iter = 0
    num_vars = training_data[0].shape[1]
    important_vars = [1]

    strategy = SequentialForwardSelectionStrategy(
        training_data, scoring_data, num_vars, important_vars, bootstrap_iter, subsample)

    assert getattr(strategy, "name") == "Sequential Forward Selection"

    expected = [(0, (training_data[0][:, [1, 0]], training_data[1]), (scoring_data[0][:, [1, 0]], scoring_data[1])),
                (2, (training_data[0][:, [1, 2]], training_data[1]), (scoring_data[0][:, [1, 2]], scoring_data[1]))]
    for (exp_var, exp_train_data, exp_score_data), (res_var, res_train_data, res_score_data) in zip(expected, strategy):
        assert exp_var == res_var
        assert (exp_train_data[0] == res_train_data[0]).all()
        assert (exp_train_data[1] == res_train_data[1]).all()
        assert (exp_score_data[0] == res_score_data[0]).all()
        assert (exp_score_data[1] == res_score_data[1]).all()

    subsample = 3
    strategy = SequentialForwardSelectionStrategy(
        training_data, scoring_data, num_vars, important_vars, bootstrap_iter, subsample)

    for (res_var, res_train_data, res_score_data) in strategy:
        assert len(res_train_data[0]) == subsample
        assert len(res_train_data[1]) == subsample
        assert len(res_score_data[0]) == len(scoring_data[0])
        assert len(res_score_data[1]) == len(scoring_data[1])


def test_sbs_strategy():

    training_data = (np.random.rand(5, 3), np.random.rand(5, ))
    scoring_data = (np.random.rand(5, 3), np.random.rand(5, ))

    subsample = 5
    bootstrap_iter = 0
    num_vars = training_data[0].shape[1]
    important_vars = [1]

    strategy = SequentialBackwardSelectionStrategy(
        training_data, scoring_data, num_vars, important_vars, bootstrap_iter, subsample)

    assert getattr(strategy, "name") == "Sequential Backward Selection"

    expected = [(0, (training_data[0][:, [2]], training_data[1]), (scoring_data[0][:, [2]], scoring_data[1])),
                (2, (training_data[0][:, [0]], training_data[1]), (scoring_data[0][:, [0]], scoring_data[1]))]
    for (exp_var, exp_train_data, exp_score_data), (res_var, res_train_data, res_score_data) in zip(expected, strategy):
        assert exp_var == res_var
        assert (exp_train_data[0] == res_train_data[0]).all()
        assert (exp_train_data[1] == res_train_data[1]).all()
        assert (exp_score_data[0] == res_score_data[0]).all()
        assert (exp_score_data[1] == res_score_data[1]).all()

    subsample = 3
    strategy = SequentialForwardSelectionStrategy(
        training_data, scoring_data, num_vars, important_vars, bootstrap_iter, subsample)

    for (res_var, res_train_data, res_score_data) in strategy:
        assert len(res_train_data[0]) == subsample
        assert len(res_train_data[1]) == subsample
        assert len(res_score_data[0]) == len(scoring_data[0])
        assert len(res_score_data[1]) == len(scoring_data[1])
