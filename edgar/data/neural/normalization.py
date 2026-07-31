import numpy as np


def by_vector_norm(response: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalizes neural responses to have a unit L2 norm along the specified axis.

    This function divides each vector (along the specified `axis`) in the `response`
    array by its L2 norm, effectively scaling it to have a magnitude of 1.

    Args:
        response: The input neural response data, typically a multi-dimensional array.
        axis: The axis along which to compute the L2 norm and normalize.
            Defaults to 0.

    Returns:
        A new `np.ndarray` with neural responses normalized to a unit L2 norm
        along the specified axis.
    """
    return response / np.linalg.norm(response, axis=axis, keepdims=True)


def by_peak(response: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalizes neural responses to have a maximum value of 1 along the specified axis.
    """
    peaks = np.max(response, axis=axis, keepdims=True)
    return response / peaks
