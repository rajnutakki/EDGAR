import numpy as np
import jax

def model(data: dict, params: dict) -> np.ndarray:
    """Predicts target cell activity from source cells using Principal Component Regression, X_target = X_source @ V @ W + b,
    with a fixed rank of 50. 

    The matrices V and W are computed using PCR in the parameter estimator.
    The reduced rank of the matrices V and W ensures the model generalizes to held-out data.
    We fix the projection matrix V to the PCR solution, so that optimization does not move us away from this stable, generalizable solution.
    """
    V = params["V"]  # Fixed projection matrix, shape: (n_source, 50)
    W = params["W"]  # Target weights, shape: (50, n_target)
    b = params["b"]      # Bias, shape: (n_target,)

    V = jax.lax.stop_gradient(V)

    responses = data["response"]  # (n_trials, n_cells)
    n_trials, n_cells = responses.shape
    source = responses[:, :n_cells//2]  # (n_trials, n_source)

    # Project raw source cells into the stable 50-PC space
    source_proj = source @ V  # (n_trials, 50)
    predicted_target = source_proj @ W + b  # (n_trials, n_target)

    return np.concatenate([np.zeros((n_trials, n_cells//2)), predicted_target], axis=1) # (n_trials, n_cells), concatenate with zeros as loss only evaluated on target cells

# Default parameter initializers
model.DEFAULT_PARAMS = lambda data: {
    "V": np.ones((data['response'].shape[-1]//2, 50)),
    "W": np.ones((50, data['response'].shape[-1] - data['response'].shape[-1]//2)),
    "b": np.zeros(data['response'].shape[-1] - data['response'].shape[-1]//2),
}
