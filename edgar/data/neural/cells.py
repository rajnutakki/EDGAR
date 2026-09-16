import numpy as np


def preferred_orientation(responses, angles):
    """Computes the preferred orientation of each cell based on its responses.

    Args:
        responses: A NumPy array of shape (n_trials, n_cells).
        angles: A NumPy array of shape (n_trials,) containing the angles
            (in radians).

    Returns:
        A NumPy array of shape (n_cells,) containing the preferred orientation
        (in radians) for each cell.
    """
    # Compute the vector sum of responses weighted by their corresponding angles
    vector_sum = np.sum(responses * np.exp(1j * angles)[:, np.newaxis], axis=0)
    # Compute the preferred orientation as the angle of the vector sum
    preferred_orientations = np.angle(vector_sum) % (2 * np.pi)

    return preferred_orientations
