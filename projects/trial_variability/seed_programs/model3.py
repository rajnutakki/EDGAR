import numpy as np

def model(data, params):
    """
    Model with a trial-varying non-linear exponent instead of multiplicative gain.
    Double peaked Gaussian tuning curve per cell raised to a trial-by-trial exponent.

    Equation : f(t,c) = r(theta(t), c) ^ trial_exponent(t) + additive_offset(t) * coupling_factor(c) 
    where r(theta(t), c) is the double-peaked gaussian tuning curve.

    data['stimulus'] = theta  # stimulus angle (radians), shape (n_trials,)

    params:
        trial_exponent: shape (n_trials,)
        additive_offset: shape (n_trials,)
        coupling_factor: shape (n_cells,)
        theta_pref: Preferred direction for the primary peak, shape (n_cells,)
        baseline: Baseline firing rate, shape (n_cells,)
        amplitude_1: Amplitude of the first peak, shape (n_cells,)
        amplitude_2: Amplitude of the second peak, shape (n_cells,)
        tuning_width: Width of both peaks, shape (n_cells,)

    Returns:
        np.ndarray: Predicted response, shape (n_trials, n_cells).
    """
    theta = data['stimulus']
    exponent = params["trial_exponent"]
    offset = params["additive_offset"]
    coupling = params["coupling_factor"]

    # Clip parameters to biologically plausible/numerically stable ranges
    offset = np.clip(offset, 0, None)
    coupling = np.clip(coupling, 0, None)
    theta_pref = np.clip(params["theta_pref"], 0, 2 * np.pi)
    baseline = np.clip(params["baseline"], 0, None)
    amplitude_1 = np.clip(params["amplitude_1"], 0, None)
    amplitude_2 = np.clip(params["amplitude_2"], 0, None)
    tuning_width = np.clip(params["tuning_width"], 0.01, None)
    
    # Clip exponent to avoid negative or extremely large values
    trial_exponent = np.clip(exponent, 0.05, 5.0)

    # circular distance
    def circ_dist(angle1, angle2):
        return np.abs(np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2)))

    dist_1 = circ_dist(theta[:, np.newaxis], theta_pref)
    dist_2 = circ_dist(theta[:, np.newaxis], (theta_pref + np.pi) % (2 * np.pi))
    peak1 = amplitude_1 * np.exp(-0.5 * (dist_1 / tuning_width) ** 2)
    peak2 = amplitude_2 * np.exp(-0.5 * (dist_2 / tuning_width) ** 2)
    tuning_curve = baseline + peak1 + peak2

    # Protect against 0^trial_exponent gradient singularities
    tuning_curve = np.clip(tuning_curve, 1e-6, None)

    return tuning_curve ** trial_exponent[:, np.newaxis] + np.outer(offset, coupling)

# Each sample of data is shaped (n_trials, n_cells)
model.DEFAULT_PARAMS = lambda data: {
    "trial_exponent": np.ones(data['response'].shape[-2]),
    "additive_offset": np.zeros(data['response'].shape[-2]),
    "coupling_factor": -0.03*np.ones(data['response'].shape[-1]),
    "theta_pref": 3*np.ones(data['response'].shape[-1]),
    "baseline": 0.0007*np.ones(data['response'].shape[-1]),
    "amplitude_1": 0.02*np.ones(data['response'].shape[-1]),
    "amplitude_2": 0.01*np.ones(data['response'].shape[-1]),
    "tuning_width": np.full(data['response'].shape[-1], 0.4),
}
