import numpy as np

from edgar.data.neural import filtering, signal


def test_activity_is_fraction_of_positive_responses_per_cell():
    response = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, -1.0, 0.0, 1.0],
        ]
    )

    # Negative and zero responses are both inactive.
    assert np.allclose(filtering.activity(response), [0.5, 0.0, 0.5])


def test_vector_concentration_is_one_for_perfect_tuning_and_zero_for_uniform():
    angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    response = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # responds only at 0
            [0.0, 1.0, 0.0, 0.0],  # responds only at pi/2 (2*theta = pi)
            [1.0, 1.0, 1.0, 1.0],  # uniform response cancels in the sum
            [1.0, 0.0, 1.0, 0.0],  # opposite phases line up
        ]
    )

    concentration = filtering.vector_concentration(response, angles)

    assert np.allclose(concentration, [1.0, 1.0, 0.0, 1.0], atol=1e-12)


def test_signal_fraction_wraps_unbiased_fraction():
    rng = np.random.default_rng(0)
    response = rng.random((3, 4, 5))

    fraction = filtering.signal_fraction(response)
    expected, _ = signal._unbiased_fraction(response)

    assert fraction.shape == (4,)
    assert np.allclose(fraction, expected)
    assert np.all((fraction >= 0) & (fraction <= 1))


def test_apply_filters_along_the_requested_cell_axis_with_kwargs():
    response = np.arange(12.0).reshape(3, 4)

    def column_mean(response, scale):
        return response.mean(axis=1) * scale

    filtered = filtering.apply(
        response, column_mean, threshold=2.0, cell_axis=1, scale=1.0
    )

    # Column means are [1.5, 5.5, 9.5]; only the last two exceed the threshold.
    assert np.array_equal(filtered, response[:, [1, 2]])


def test_by_activity_keeps_only_cells_above_threshold():
    response = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    filtered = filtering.by_activity(response, threshold=0.75)

    assert np.array_equal(filtered, response[[2]])


def test_by_vector_concentration_keeps_only_directionally_tuned_cells():
    angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    response = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    filtered = filtering.by_vector_concentration(response, angles=angles, threshold=0.5)

    assert np.array_equal(filtered, response[[0]])


def test_by_signal_fraction_filters_cells_stored_on_the_second_axis():
    clean = np.tile(np.array([[[0.0, 1.0, 2.0]]]), (3, 1, 1))
    noise = np.array(
        [
            [[-0.1, 0.0, 0.1]],
            [[0.0, 0.0, 0.0]],
            [[0.1, 0.0, -0.1]],
        ]
    )
    response = np.concatenate([clean, clean + noise], axis=1)

    fractions, _ = signal._unbiased_fraction(response)
    assert fractions[0] == 1.0
    assert fractions[1] < 0.999

    filtered = filtering.by_signal_fraction(response, threshold=0.999)

    assert np.array_equal(filtered, response[:, [0], :])
