import numpy as np

from edgar.data.neural import trials


def test_shuffle_returns_empty_list_when_no_arrays_are_given():
    assert trials.shuffle() == []


def test_shuffle_applies_the_same_permutation_to_every_array(monkeypatch):
    monkeypatch.setattr(trials.np.random, "permutation", lambda n: np.array([2, 0, 1]))
    first = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    second = np.array([10.0, 20.0, 30.0])

    shuffled_first, shuffled_second = trials.shuffle(first, second)

    assert np.array_equal(shuffled_first, [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
    assert np.array_equal(shuffled_second, [30.0, 10.0, 20.0])


def test_shuffle_respects_the_requested_axis(monkeypatch):
    monkeypatch.setattr(trials.np.random, "permutation", lambda n: np.array([2, 0, 1]))
    first = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    second = np.array([10.0, 20.0, 30.0])

    shuffled_first, shuffled_second = trials.shuffle(first, second, axis=0)

    assert np.array_equal(shuffled_first, [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(shuffled_second, [30.0, 10.0, 20.0])
