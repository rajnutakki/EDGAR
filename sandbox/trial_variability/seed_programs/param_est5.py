import numpy as np


def parameter_estimator(data):
    """
    Estimates initial parameters for the generalized gaussian.
    """
    y = np.asarray(data["response"])  # shape (n_trials, n_cells)

    theta = np.asarray(data["stimulus"])  # shape (n_trials,)

    n_trials, n_cells = y.shape
    r = y  # shape (n_trials, n_cells)
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
