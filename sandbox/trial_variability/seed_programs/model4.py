import numpy as np


def model(data, params):
    """
    Affine model for trial-to-trial variability with signal binned by angle

    Equation : f(t,c) = multiplicative_gain(t) * s_c(\theta_t) + additive_offset(t) * coupling_factor(c) where s_c(\theta(t)) is the binned signal.

    data['signal'] = binned mean signal, shape (n_trials, n_cells)

    params:
        multiplicative_gain: shape (n_trials,)
        additive_offset: shape (n_trials,)
        coupling_factor: shape (n_cells,)

    Returns:
        np.ndarray: Predicted response, shape (n_trials, n_cells).
    """
    signal = data["signal"]
    gain = params["multiplicative_gain"]
    offset = params["additive_offset"]
    coupling = params["coupling_factor"]

    # Clip parameters to biologically plausible ranges
    gain = np.clip(gain, 0, None)
    offset = np.clip(offset, 0, None)
    coupling = np.clip(coupling, 0, None)

    return gain[:, np.newaxis] * signal + np.outer(offset, coupling)


# Each sample of data is shaped (n_trials, n_cells), taken values based on mean of values given by parameter estimator.
model.DEFAULT_PARAMS = lambda data: {
    "multiplicative_gain": 0.8 * np.ones(data["response"].shape[-2]),
    "additive_offset": np.zeros(data["response"].shape[-2]),
    "coupling_factor": -0.03 * np.ones(data["response"].shape[-1]),
}
