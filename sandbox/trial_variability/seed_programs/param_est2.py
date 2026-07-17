import numpy as np


def parameter_estimator(data):
    """
    Estimates initial parameters for the population skewed asymmetric double gaussian with affine shared variability.
    First estimates gain, additive offset and coupling via SVD, and then estimates tuning curve parameters from the resulting residuals.
    """
    y = np.asarray(data["response"])  # shape (n_trials, n_cells)

    theta = np.asarray(data["stimulus"])  # shape (n_trials,)

    n_trials, n_cells = y.shape

    # First approximate r_{t,c} as the mean response across trials for each cell.
    # We use np.nanmean to safely ignore the corner-masked NaNs.
    r_c = np.nanmean(y, axis=0)  # shape (n_cells,)
    r_c = np.nan_to_num(r_c, nan=0.0)

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

    # Finally estimate parameters for the tuning curve from residual
    residual = (y - np.outer(a_t, h_c)) / g_t[
        :, np.newaxis
    ]  # shape (n_trials, n_cells)

    # Use the residual as the tuning response input (handling NaNs as done below)
    r = residual

    # We will compute binned response per cell using 16 bins
    num_bins = 16
    bins = np.linspace(0, 2 * np.pi, num_bins + 1)
    bin_c = (bins[:-1] + bins[1:]) / 2

    binned_r_list = []
    for i in range(num_bins):
        mask = (theta >= bins[i]) & (theta < bins[i + 1])
        if np.any(mask):
            bin_mean = np.nanmean(r[mask], axis=0)  # shape (n_cells,)
        else:
            bin_mean = np.zeros(n_cells)
        binned_r_list.append(bin_mean)

    binned_r = np.stack(binned_r_list, axis=0)  # shape (num_bins, n_cells)

    # Calculate baseline for each cell as the minimum binned response
    baseline = np.nanmin(binned_r, axis=0)
    baseline = np.nan_to_num(baseline, nan=0.0)

    r_above_baseline = binned_r - baseline  # (num_bins, n_cells)

    # Find max response and preferred orientation per cell, handling possible all-NaN cases
    theta_pref = np.zeros(n_cells)
    amp1 = np.zeros(n_cells)
    for c in range(n_cells):
        col = r_above_baseline[:, c]
        if np.all(np.isnan(col)):
            theta_pref[c] = np.pi
            amp1[c] = 0.0
        else:
            max_idx = np.nanargmax(col)
            theta_pref[c] = bin_c[max_idx]
            amp1[c] = col[max_idx]

    amp1 = np.maximum(0.0, amp1)

    # Second peak at opposite orientation
    theta_pref_2 = (theta_pref + np.pi) % (2 * np.pi)
    amp2 = np.zeros(n_cells)
    for c in range(n_cells):
        col = r_above_baseline[:, c]
        if not np.all(np.isnan(col)):
            closest_idx_2 = np.argmin(np.abs(bin_c - theta_pref_2[c]))
            amp2[c] = col[closest_idx_2]

    amp2 = np.maximum(0.0, amp2)

    # Default widths and exponents
    default_w = np.pi / 6
    default_e = 2.0

    # If a cell has extremely low amplitude, give a safe non-zero fallback
    mean_r = np.nanmean(r, axis=0)
    mean_r = np.nan_to_num(mean_r, nan=1.0)
    for c in range(n_cells):
        if amp1[c] < 1e-6:
            amp1[c] = max(0.0, mean_r[c] - baseline[c])
            theta_pref[c] = np.pi

    return {
        "multiplicative_gain": g_t.astype(float),
        "additive_offset": a_t.astype(float),
        "coupling_factor": h_c.astype(float),
        "theta_pref": theta_pref.astype(float),
        "baseline": baseline.astype(float),
        "amplitude_1": amp1.astype(float),
        "amplitude_2": amp2.astype(float),
        "tuning_width_1_left": np.full(n_cells, default_w, dtype=float),
        "tuning_width_1_right": np.full(n_cells, default_w, dtype=float),
        "tuning_width_2_left": np.full(n_cells, default_w, dtype=float),
        "tuning_width_2_right": np.full(n_cells, default_w, dtype=float),
        "peak_exponent_1": np.full(n_cells, default_e, dtype=float),
        "peak_exponent_2": np.full(n_cells, default_e, dtype=float),
        "angle_offset_2": np.zeros(n_cells, dtype=float),
    }
