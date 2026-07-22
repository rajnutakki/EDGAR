import numpy as np

def model(data, params):
    """
    Peer prediction with rank 30 matrix factorization W = W_A @ W_B.
    """
    W_A = params["W_A"] #(n_source, 30)
    W_B = params["W_B"] #(30, n_target)
    b = params["b"]
    responses = data["response"]  # (n_trials, n_cells)
    n_trials, n_cells = responses.shape
    source = responses[:, :n_cells//2]  # (n_trials, n_source)
    predicted_target = source @ W_A @ W_B + b  # (n_trials, n_target)
    return np.concatenate([np.zeros((n_trials, n_cells//2)), predicted_target], axis=1)  # (n_trials, n_cells)

# Each sample of data is shaped (n_trials, n_cells)
model.DEFAULT_PARAMS = lambda data: {
    "W_A": np.ones((data['response'].shape[-1]//2, 30)),
    "W_B": np.ones((30, data['response'].shape[-1] - data['response'].shape[-1]//2)),
    "b": np.zeros(data['response'].shape[-1] - data['response'].shape[-1]//2),
}
