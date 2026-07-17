import numpy as np


def parameter_estimator(data):
    """
    Estimates initial parameters for the affine model with signal binned by angle.

    Args:
        data (dict): Dictionary containing:
            - 'response': Normalized neural responses, shape (n_trials, n_cells)
            - 'stimulus': Stimulus orientations in radians, shape (n_trials,) or (1, n_trials).

    Returns:
        dict: Estimated parameters dictionary matching model1.py expected inputs:
            - "multiplicative_gain" (n_trials,)
            - "additive_offset" (n_trials,)
            - "coupling_factor" (n_cells,)
    """
    y = np.asarray(data["response"])  # shape (n_trials, n_cells)
    r_c = data["signal"]  # shape (n_trials, n_cells)

    # Best least-squares estimate of a_t h_c is rank-1 SVD of y_{t,c} - \bar{r}_c.
    # We impute NaNs with 0.0 to prevent standard SVD from propagating NaNs.
    residual_1 = y - r_c  # shape (n_trials, n_cells)
    residual_1_imputed = np.nan_to_num(residual_1, nan=0.0)

    U, S, Vt = np.linalg.svd(residual_1_imputed, full_matrices=False)
    # Use convention of splitting singular value evenly between left and right singular vectors
    a_t = U[:, 0] * np.sqrt(S[0])  # shape (n_trials,)
    h_c = Vt[0, :] * np.sqrt(S[0])  # shape (n_cells,)

    # Now fit g_t with rank-1 SVD.
    # Again, we impute NaNs with 0.0 before performing the SVD.
    residual_2 = y - np.outer(a_t, h_c)  # shape (n_trials, n_cells)
    residual_2_imputed = np.nan_to_num(residual_2, nan=0.0)

    U, S, Vt = np.linalg.svd(residual_2_imputed, full_matrices=False)
    g_t = U[:, 0]
    v_c = Vt[0, :]

    # To figure out how to parcel out singular value, minimize distance between v_c and r_c
    b = np.dot(r_c, v_c) / (np.linalg.norm(v_c) ** 2 + 1e-8)
    g_t = S[0] * g_t / b  # scale g_t by singular value and b to make v_c close to r_c
    g_t = np.abs(g_t)
    g_t = np.clip(g_t, 0.01, None)

    return {
        "multiplicative_gain": g_t.astype(float),
        "additive_offset": a_t.astype(float),
        "coupling_factor": h_c.astype(float),
    }
