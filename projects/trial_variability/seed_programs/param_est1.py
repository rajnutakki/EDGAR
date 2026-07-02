import numpy as np

def parameter_estimator(data):
    """
    Estimates initial parameters for the population Double Gaussian Model with affine shared variability.
    Takes into account NaNs in data by zeroing out those parts of the residuals.

    Args:
        data (dict): Dictionary containing:
            - 'response': Normalized neural responses, shape (n_trials, n_cells) or (1, n_trials, n_cells).
            - 'stimulus': Stimulus orientations in radians, shape (n_trials,) or (1, n_trials).

    Returns:
        dict: Estimated parameters dictionary matching model1.py expected inputs:
            - "multiplicative_gain" (n_trials,)
            - "additive_offset" (n_trials,)
            - "coupling_factor" (n_cells,)
            - "theta_pref" (n_cells,)
            - "baseline" (n_cells,)
            - "amplitude_1" (n_cells,)
            - "amplitude_2" (n_cells,)
            - "tuning_width" (n_cells,)
    """
    y = np.asarray(data["response"]) #shape (n_trials, n_cells)
        
    theta = np.asarray(data["stimulus"]) #shape (n_trials,)

    # First approximate r_{t,c} as the mean response across trials for each cell.
    # We use np.nanmean to safely ignore the corner-masked NaNs.
    r_c = np.nanmean(y, axis=0) # shape (n_cells,)
    r_c = np.nan_to_num(r_c, nan=0.0)
    
    # Best least-squares estimate of a_t h_c is rank-1 SVD of y_{t,c} - \bar{r}_c.
    # We impute NaNs with 0.0 to prevent standard SVD from propagating NaNs.
    residual_1 = y - r_c # shape (n_trials, n_cells)
    residual_1_imputed = np.nan_to_num(residual_1, nan=0.0)
    
    U, S, Vt = np.linalg.svd(residual_1_imputed, full_matrices=False)
    # Use convention of splitting singular value evenly between left and right singular vectors
    a_t = U[:, 0] * np.sqrt(S[0]) # shape (n_trials,) 
    h_c = Vt[0, :] * np.sqrt(S[0]) # shape (n_cells,)

    # Now fit g_t with rank-1 SVD.
    # Again, we impute NaNs with 0.0 before performing the SVD.
    residual_2 = y - np.outer(a_t, h_c) # shape (n_trials, n_cells)
    residual_2_imputed = np.nan_to_num(residual_2, nan=0.0)
    
    U, S, Vt = np.linalg.svd(residual_2_imputed, full_matrices=False)
    g_t = U[:, 0] 
    v_c = Vt[0, :]
    
    # To figure out how to parcel out singular value, minimize distance between v_c and r_c
    b = np.dot(r_c, v_c) / (np.linalg.norm(v_c)**2 + 1e-8)
    g_t = S[0] * g_t / b # scale g_t by singular value and b to make v_c close to r_c
    g_t = np.abs(g_t)
    g_t = np.clip(g_t, 0.01, None)

    # Finally estimate parameters for the tuning curve from residual
    residual = (y - np.outer(a_t, h_c)) / g_t[:, np.newaxis] # shape (n_trials, n_cells)
    
    # NaN-aware vectorized double-peaked Gaussian tuning curve parameter estimation
    n_bins = 50
    bin_idx = ((theta * n_bins) / (2 * np.pi)).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    n_cells = y.shape[1]
    sums = np.zeros((n_bins, n_cells))
    counts = np.zeros((n_bins, n_cells))
    
    valid_mask = ~np.isnan(residual)
    np.add.at(sums, bin_idx, np.nan_to_num(residual, nan=0.0))
    np.add.at(counts, bin_idx, valid_mask.astype(float))

    sig = 2
    x = np.arange(-int(3 * sig), int(3 * sig) + 1)
    k = np.exp(-0.5 * (x / sig) ** 2)
    k = k / np.sum(k)
    pad = len(k) // 2
    
    sums_padded = np.pad(sums, ((pad, pad), (0, 0)), mode='wrap')
    counts_padded = np.pad(counts, ((pad, pad), (0, 0)), mode='wrap')
    
    num_conv = np.zeros((n_bins, n_cells))
    den_conv = np.zeros((n_bins, n_cells))
    for i, val in enumerate(k):
        num_conv += sums_padded[i : i + n_bins, :] * val
        den_conv += counts_padded[i : i + n_bins, :] * val
        
    tuning_curve = num_conv / (den_conv + 1e-8)

    pref_idx = np.argmax(tuning_curve, axis=0)
    theta_pref = pref_idx * (2 * np.pi / n_bins)
    baseline = np.min(tuning_curve, axis=0)
    amplitude_1 = np.max(tuning_curve, axis=0) - baseline
    
    row_indices = (pref_idx + n_bins // 2) % n_bins
    col_indices = np.arange(n_cells)
    amplitude_2 = tuning_curve[row_indices, col_indices] - baseline
    
    half_max = baseline + amplitude_1 / 2.0
    indices = (np.arange(-5, 6)[:, np.newaxis] + pref_idx) % n_bins
    
    tuning_curve_subset = tuning_curve[indices, col_indices]
    above_half_max = tuning_curve_subset >= half_max
    
    full_width_half_max = 2 * np.pi * np.sum(above_half_max, axis=0) / n_bins
    tuning_width = full_width_half_max / (2.0 * np.sqrt(2 * np.log(2)))
    
    return {
        "multiplicative_gain": g_t.astype(float),
        "additive_offset": a_t.astype(float),
        "coupling_factor": h_c.astype(float),
        "theta_pref": theta_pref.astype(float),
        "baseline": baseline.astype(float),
        "amplitude_1": amplitude_1.astype(float),
        "amplitude_2": amplitude_2.astype(float),
        "tuning_width": tuning_width.astype(float),
    }
