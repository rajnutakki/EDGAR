# ruff: noqa: E402

import sys
import os
import pathlib

# Configure JAX memory allocation flags to avoid CUDA OOM errors during scoring
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

repo_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(repo_root / "sandbox" / "trial_variability"))
sys.path.append(str(repo_root / "projects" / "trial_variability" / "data_loader"))
from inspect_data import load_and_score_seeds

import jax
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load and score seed programs
    print("\n--- Loading and scoring seed programs ---")
    population, X_disc, X_val = load_and_score_seeds()
    model_predictions = [
        jax.vmap(p.compile_model(), in_axes=(0, 0))(X_val[1], p.params)[0]
        for p in population
    ]  # each element (n_trials, n_cells

    mid_trials, mid_cells = (
        X_val[1]["response"].shape[1] // 2,
        X_val[1]["response"].shape[2] // 2,
    )
    Y_test = X_val[1]["response"][0, mid_trials:, mid_cells:]
    mp1 = model_predictions[0][mid_trials:, mid_cells:]
    mp2 = model_predictions[1][mid_trials:, mid_cells:]
    Y_centered = Y_test - np.mean(Y_test, axis=0)  # average over trials
    U, S, Vt = np.linalg.svd(Y_centered)

    fig, ax = plt.subplots()
    plot_root = str(repo_root / "sandbox" / "trial_variability")
    ax.plot(S)
    ax.set_yscale("log")
    plt.savefig(plot_root + "/response_evalues.png")
