import numpy as np

from edgar.data.neural import normalization


def test_by_vector_norm_normalizes_each_column_by_default():
    response = np.array([[3.0, 0.0], [4.0, 0.0], [0.0, 2.0]])

    normalized = normalization.by_vector_norm(response)

    assert np.allclose(normalized, [[0.6, 0.0], [0.8, 0.0], [0.0, 1.0]])


def test_by_vector_norm_can_normalize_along_a_non_default_axis():
    response = np.array([[3.0, 4.0], [0.0, 2.0]])

    normalized = normalization.by_vector_norm(response, axis=1)

    assert np.allclose(normalized, [[0.6, 0.8], [0.0, 1.0]])


def test_by_peak_uses_a_safe_value_for_zero_peaks():
    response = np.array([[2.0, 4.0], [0.0, 0.0]])
    peaks = np.array([2.0, 0.0])

    normalized = normalization.by_peak(response, peaks)

    assert np.allclose(normalized, [[1.0, 2.0], [0.0, 0.0]])
    assert np.all(np.isfinite(normalized))
