#Tuning curve taken from /home/rajah/projects/orientation_tuning/run_output/05-22/10-05-13
import numpy as np

def model(data, params):
    """
    Affine shared variability model, consisting of gain modulation + additive offset,
    using a parameterized Double Split Generalized-Gaussian orientation tuning curve per cell.

    Equation : f(t,c) = multiplicative_gain(t) * r(theta(t), c) + additive_offset(t) * coupling_factor(c)
    where r(theta(t), c) is the Double Split Generalized-Gaussian orientation tuning curve for cell c.

    data['stimulus'] = theta  # stimulus angle (radians), shape (n_trials,)
    data['response'] = response # shape (n_trials, n_cells)

    params:
        multiplicative_gain: shape (n_trials,)
        additive_offset: shape (n_trials,)
        coupling_factor: shape (n_cells,)
        theta_pref: Preferred direction for the primary peak, shape (n_cells,)
        baseline: Baseline firing rate, shape (n_cells,)
        amplitude_1: Amplitude of the first peak, shape (n_cells,)
        amplitude_2: Amplitude of the second peak, shape (n_cells,)
        tuning_width_1_left: Width of the left side of the first peak, shape (n_cells,)
        tuning_width_1_right: Width of the right side of the first peak, shape (n_cells,)
        tuning_width_2_left: Width of the left side of the second peak, shape (n_cells,)
        tuning_width_2_right: Width of the right side of the second peak, shape (n_cells,)
        peak_exponent_1: Exponent for the generalized Gaussian shape of the first peak, shape (n_cells,)
        peak_exponent_2: Exponent for the generalized Gaussian shape of the second peak, shape (n_cells,)

    Returns:
        np.ndarray: Predicted response, shape (n_trials, n_cells).
    """
    theta = data['stimulus']
    gain = params["multiplicative_gain"]
    offset = params["additive_offset"]
    coupling = params["coupling_factor"]

    # Clip parameters to biologically plausible ranges
    theta_pref = np.clip(params["theta_pref"], 0, 2 * np.pi)
    baseline = np.clip(params["baseline"], 0, None)
    amplitude_1 = np.clip(params["amplitude_1"], 0, None)
    amplitude_2 = np.clip(params["amplitude_2"], 0, None)
    tuning_width_1_left = np.clip(params["tuning_width_1_left"], 0.001, None)
    tuning_width_1_right = np.clip(params["tuning_width_1_right"], 0.001, None)
    tuning_width_2_left = np.clip(params["tuning_width_2_left"], 0.001, None)
    tuning_width_2_right = np.clip(params["tuning_width_2_right"], 0.001, None)
    peak_exponent_1 = np.clip(params["peak_exponent_1"], 0.5, 10.0)
    peak_exponent_2 = np.clip(params["peak_exponent_2"], 0.5, 10.0)

    # Signed circular distance
    def signed_circ_dist(angle1, angle2):
        return np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))

    # Calculate for the first peak
    # Broadcast theta (n_trials, 1) against theta_pref (n_cells,)
    dist_1_signed = signed_circ_dist(theta[:, np.newaxis], theta_pref)
    abs_dist_1 = np.abs(dist_1_signed)
    effective_width_1 = np.where(dist_1_signed < 0, tuning_width_1_left, tuning_width_1_right)
    peak_1 = amplitude_1 * np.exp(-0.5 * (abs_dist_1 / effective_width_1) ** peak_exponent_1)

    # Calculate for the second peak (opposite orientation)
    theta_pref_2 = (theta_pref + np.pi) % (2 * np.pi)
    dist_2_signed = signed_circ_dist(theta[:, np.newaxis], theta_pref_2)
    abs_dist_2 = np.abs(dist_2_signed)
    effective_width_2 = np.where(dist_2_signed < 0, tuning_width_2_left, tuning_width_2_right)
    peak_2 = amplitude_2 * np.exp(-0.5 * (abs_dist_2 / effective_width_2) ** peak_exponent_2)

    tuning_curve = baseline + peak_1 + peak_2

    return gain[:, np.newaxis] * tuning_curve + np.outer(offset, coupling)

#Data is shaped (n_trials, n_cells)
model.DEFAULT_PARAMS = lambda data: {
    "multiplicative_gain": np.ones(data['response'].shape[-2]),
    "additive_offset": np.zeros(data['response'].shape[-2]),
    "coupling_factor": np.ones(data['response'].shape[-1]),
    "theta_pref": np.zeros(data['response'].shape[-1]),
    "baseline": np.zeros(data['response'].shape[-1]),
    "amplitude_1": np.ones(data['response'].shape[-1]),
    "amplitude_2": np.zeros(data['response'].shape[-1]),
    "tuning_width_1_left": np.full(data['response'].shape[-1], np.pi / 6),
    "tuning_width_1_right": np.full(data['response'].shape[-1], np.pi / 6),
    "tuning_width_2_left": np.full(data['response'].shape[-1], np.pi / 6),
    "tuning_width_2_right": np.full(data['response'].shape[-1], np.pi / 6),
    "peak_exponent_1": np.full(data['response'].shape[-1], 2.0),
    "peak_exponent_2": np.full(data['response'].shape[-1], 2.0),
}

    