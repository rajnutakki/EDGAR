import numpy as np

def model(data, params):
    """
        Peer prediction, Y_{target} = Y_{source}@W + b, where W is a (n_source, n_target) weight matrix and b is a (n_target,) bias vector. 
    """
    W = params["W"]
    b = params["b"]
    responses = data["response"] #(n_trials, n_cells)
    n_trials, n_cells = responses.shape
    source = responses[:,:n_cells//2] # (n_trials, n_source)
    predicted_target = source @ W + b # (n_trials, n_target)
    return np.concatenate([np.zeros((n_trials, n_cells//2)), predicted_target], axis=1) # (n_trials, n_cells) 

#Each sample of data is shaped (n_trials, n_cells)
model.DEFAULT_PARAMS = lambda data: {
    "W": np.ones((data['response'].shape[-1]//2, data['response'].shape[-1] - data['response'].shape[-1]//2)), # (n_source, n_target)
    "b": np.zeros(data['response'].shape[-1] - data['response'].shape[-1]//2), # (n_target,)
}