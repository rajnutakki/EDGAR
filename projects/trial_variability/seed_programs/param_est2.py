import numpy as np

def parameter_estimator(data):
    """
    Compute the weight matrix, W_A and W_B, and bias vector, b, for peer prediction model, using reduced rank regression.
    """
    responses = data["response"] #(n_trials, n_cells)
    n_trials, n_cells = responses.shape

    # Fit only on the first half of trials
    trial_mid = n_trials // 2
    X = responses[:trial_mid, :n_cells//2] #(n_trials//2, n_source)
    Y = responses[:trial_mid, n_cells//2:] #(n_trials//2, n_target)

    # Compute the least squares solution for W using the normal equations
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_c = X - X_mean
    Y_c = Y - Y_mean
    A = X_c.T @ X_c
    B = X_c.T @ Y_c
    W_ols = np.linalg.solve(A, B)

    # Perform SVD to reduce rank of W_ols, will generalize better to held-out data.
    Y_hat = X_c @ W_ols
    U, S, Vh = np.linalg.svd(Y_hat, full_matrices=False)
    V = Vh.T
    fixed_rank = 30
    r = min(fixed_rank, S.shape[0])

    n_source = X.shape[1]
    n_target = Y.shape[1]

    # Initialize low-rank matrices W_A and W_B with zeros
    W_A = np.zeros((n_source, fixed_rank))
    W_B = np.zeros((fixed_rank, n_target))

    # Populate top r components
    W_A[:, :r] = W_ols @ V[:, :r]
    W_B[:r, :] = Vh[:r, :]

    # Compute the intercept using the low-rank reconstructed weight matrix
    W_lowrank = W_A @ W_B
    b = Y_mean - X_mean @ W_lowrank #compute the intercept

    return {
        "W_A": W_A.astype(float),
        "W_B": W_B.astype(float),
        "b": b.astype(float)
    }
