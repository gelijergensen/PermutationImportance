

from src.selection_strategies import sfs_strategy


def test_sfs_strategy():
    assert [0, 2, 4, 9] == sfs_strategy(10, [0, 2, 4], 9)
