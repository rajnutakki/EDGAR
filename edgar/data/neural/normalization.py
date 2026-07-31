import numpy as np


def by_vector_norm(response: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalizes neural responses to have a unit L2 norm along the specified axis.
    """
    return response / np.linalg.norm(response, axis=axis, keepdims=True)


def by_peak(response: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalizes neural responses to have a maximum value of 1 along the specified axis.
    """
    peaks = np.max(response, axis=axis, keepdims=True)
    return response / peaks
