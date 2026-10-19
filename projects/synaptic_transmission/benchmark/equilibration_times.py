"""Plot unstimulated equilibration rollouts for both seed models."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SEED_ROOT = PROJECT_ROOT / "seed_programs"
DATA_ROOT = PROJECT_ROOT / "data_loader"
DATA_PATH = Path("/home/rajah/projects/synaptic_transmission/dataset/data.npz")
OUTPUT_PATH = HERE / "equilibration_rollout.png"

TIME_STEP_MS = 0.1
N_STEPS = 300_000

sys.path.insert(0, str(SEED_ROOT))
sys.path.insert(0, str(DATA_ROOT))

from load_data import load_data
from model1 import model as model1
from model2 import model as model2
from param_est1 import parameter_estimator as param_est1
from param_est2 import parameter_estimator as param_est2


def _rollout_model(model, params, n_steps):
    """Return release and state trajectories during an unstimulated rollout."""
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    initial_state = model.INITIAL_STATE(params)
    state_history = {
        name: np.empty(n_steps + 1, dtype=np.float64) for name in initial_state
    }
    for name, value in initial_state.items():
        state_history[name][0] = value

    release_history = np.empty(n_steps + 1, dtype=np.float64)
    release_history[0] = 0.0
    state = initial_state
    release = 0.0

    for step_index in range(1, n_steps + 1):
        state, release = model(
            state,
            {"stim_prev": 0.0, "release_prev": release},
            params,
        )
        release_history[step_index] = release
        for name, value in state.items():
            state_history[name][step_index] = value

    return {"release": release_history, **state_history}


def _plot_rollouts(rollouts):
    time_seconds = np.arange(N_STEPS + 1) * TIME_STEP_MS / 1_000.0
    quantity_names = list(
        dict.fromkeys(name for _, _, trajectory in rollouts for name in trajectory)
    )
    colours = {
        name: plt.colormaps["tab10"](index)
        for index, name in enumerate(quantity_names)
    }

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, (model_name, parameter_name, trajectory) in zip(
        axes.flat, rollouts, strict=True
    ):
        for quantity_name, values in trajectory.items():
            axis.plot(
                time_seconds,
                values,
                color=colours[quantity_name],
                label=quantity_name,
            )
        axis.set_title(f"{model_name} — {parameter_name} parameters")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Value")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Unstimulated model equilibration")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    (discovery_data, _), _, _ = load_data(DATA_PATH)

    rollouts = []
    for model_name, model, parameter_estimator in (
        ("Model 1", model1, param_est1),
        ("Model 2", model2, param_est2),
    ):
        parameter_sets = (
            ("estimated", parameter_estimator(discovery_data)),
            ("default", model.DEFAULT_PARAMS),
        )
        for parameter_name, params in parameter_sets:
            trajectory = _rollout_model(model, params, N_STEPS)
            rollouts.append((model_name, parameter_name, trajectory))

    _plot_rollouts(rollouts)
