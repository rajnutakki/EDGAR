import numpy as np
import pytest

from edgar.data.neural import signal


def test_extract_stimulus_related_returns_data_without_preprocessing():
    response = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data = {"sresp": response.tolist()}

    extracted = signal.extract_stimulus_related(data, n_pcs=0)

    assert isinstance(extracted, np.ndarray)
    assert np.array_equal(extracted, response)


def test_extract_stimulus_related_removes_spontaneous_mean_per_neuron():
    data = {
        "sresp": np.array([[10.0, 12.0, 14.0], [20.0, 22.0, 24.0]]),
        "mean_spont": np.array([10.0, 20.0]),
    }

    extracted = signal.extract_stimulus_related(data, n_pcs=0, spont_mean_removal=True)

    assert np.allclose(extracted, [[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]])


def test_extract_stimulus_related_projects_out_requested_spontaneous_pcs():
    data = {
        "sresp": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "u_spont": np.eye(2),
    }

    one_pc = signal.extract_stimulus_related(data, n_pcs=1)
    both_pcs = signal.extract_stimulus_related(data, n_pcs=2)

    assert np.allclose(one_pc, [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
    assert np.allclose(both_pcs, np.zeros((2, 3)))


def test_extract_stimulus_related_z_scores_each_neuron():
    data = {"sresp": np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])}

    extracted = signal.extract_stimulus_related(data, n_pcs=0, z_score=True)

    assert np.allclose(extracted.mean(axis=1), 0.0)
    assert np.allclose(extracted.std(axis=1), 1.0)


def test_unbiased_fraction_requires_minimum_number_of_repeats():
    response = np.ones((1, 2, 3))

    with pytest.raises(ValueError, match="at least 2 repeats"):
        signal._unbiased_fraction(response, min_repeats=2)


def test_unbiased_fraction_is_one_for_noiseless_repeats():
    means = np.array([0.0, 10.0, 20.0])
    response = np.tile(means, (2, 1, 1))

    fraction, intermediates = signal._unbiased_fraction(response)

    assert np.allclose(fraction, [1.0])
    assert np.allclose(intermediates["S2"], [200.0 / 3.0])
    assert np.allclose(intermediates["V2"], [0.0])
    assert np.allclose(intermediates["mu_angles"], [means])
    assert np.allclose(intermediates["var_angles"], np.zeros((1, 3)))


def test_unbiased_fraction_is_reduced_by_noise_with_zero_mean_per_angle():
    means = np.array([0.0, 1.0, 2.0])
    clean = np.tile(means, (3, 1, 1))
    noise = np.array(
        [
            [[-0.1, 0.0, 0.1]],
            [[0.0, 0.0, 0.0]],
            [[0.1, 0.0, -0.1]],
        ]
    )

    noiseless_fraction, _ = signal._unbiased_fraction(clean)
    noisy_fraction, intermediates = signal._unbiased_fraction(clean + noise)

    assert np.allclose(noiseless_fraction, [1.0])
    assert 0.0 < noisy_fraction[0] < 1.0
    assert np.allclose(intermediates["mu_angles"], [means])
    assert np.all(intermediates["V2"] > 0)


def test_binned_mean_averages_values_within_grid_centered_bins():
    x = np.array([0.0, 0.4, 0.6, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    binned = signal.binned_mean(x, y, np.array([0.0, 1.0]))

    assert np.allclose(binned, [1.5, 3.5])


def test_binned_mean_returns_bin_indices_when_requested():
    x = np.array([0.0, 0.4, 0.6, 1.0])
    y = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])

    binned, indices = signal.binned_mean(
        x, y, np.array([0.0, 1.0]), return_indices=True
    )

    assert np.array_equal(indices, [0, 0, 1, 1])
    assert np.allclose(binned, [[1.5, 3.5], [15.0, 35.0]])


def test_binned_mean_handles_empty_and_single_point_grids():
    x = np.array([0.0, 0.4, 0.6])
    y = np.array([1.0, 2.0, 5.0])

    empty = signal.binned_mean(x, y, np.array([]))
    single = signal.binned_mean(x, y, np.array([0.5]))
    single_with_indices, indices = signal.binned_mean(
        x, y, np.array([0.5]), return_indices=True
    )

    assert empty.size == 0
    assert np.allclose(single, [8.0 / 3.0])
    assert np.allclose(single_with_indices, [8.0 / 3.0])
    assert np.array_equal(indices, np.zeros_like(x, dtype=int))


def test_binned_mean_clips_out_of_range_values_into_edge_bins():
    x = np.array([-10.0, 0.5, 10.0])
    y = np.array([1.0, 2.0, 3.0])

    binned, indices = signal.binned_mean(
        x, y, np.array([0.0, 1.0]), return_indices=True
    )

    # Values below/above the outer edges are clipped, while the right edge of
    # the first bin (0.5) belongs to the second bin.
    assert np.array_equal(indices, [0, 1, 1])
    assert np.allclose(binned, [1.0, 2.5])
