import numpy as np

def parameter_estimator(data: dict) -> dict:
    """Computes the optimal PCR projection and weight parameters using least squares.
    """
    fixed_rank = 50
    responses = data["response"]  # (n_trials, n_cells)
    n_trials, n_cells = responses.shape

    # Fit only on the first half of trials
    trial_mid = n_trials // 2
    X = responses[:trial_mid, :n_cells//2]  # (n_trials//2, n_source)
    Y = responses[:trial_mid, n_cells//2:]  # (n_trials//2, n_target)

    # Center training data
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    # Perform SVD/PCA on centered source cells to find the top 50 principal components
    U, S, Vh = np.linalg.svd(X_c, full_matrices=False)
    V = Vh[:fixed_rank, :].T  # (n_source, 50)

    # Project the centered source cells into the 50-PC space
    X_proj = X_c @ V  # (n_trials//2, 50)

    # Solve the OLS problem in the PC space (stable)
    A = X_proj.T @ X_proj
    B = X_proj.T @ Y_c
    W = np.linalg.solve(A, B)  # (50, n_target)

    # Compute the intercept using the projected weights
    b = Y_mean - X_mean @ V @ W

    return {
        "V": V.astype(float),
        "W": W.astype(float),
        "b": b.astype(float)
    }
