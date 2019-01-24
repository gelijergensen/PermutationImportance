

from src.selection_strategies import sfs_strategy


def test_sfs_strategy():
    assert [(1, [0, 2, 1]), (3, [0, 2, 3])] == sfs_strategy(
        4, [0, 2], None, None)
