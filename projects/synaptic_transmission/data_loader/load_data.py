"""Data loading and rollout hooks for synaptic-transmission discovery.

One EDGAR sample is one experiment.  Its observations are the protocols run on
that experiment, so one parameter set is shared across every protocol in a
sample.  Protocols have different recording times and are padded with NaNs;
``stim`` is the dense, 0.1 ms stimulus grid used by the rollout.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


DT = 0.1
DEFAULT_DATA_PATH = "/home/rajah/projects/synaptic_transmission/dataset/data.npz"
DEFAULT_DISCOVERY_EQUILIBRATION_MS = 30_000.0
DEFAULT_VALIDATION_EQUILIBRATION_MS = 24_000.0


def _data_file(data_path: str) -> Path:
    path = Path(data_path or DEFAULT_DATA_PATH)
    if path.is_dir():
        path /= "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Synaptic-transmission dataset not found: {path}")
    return path


def _load_protocol(npz, dataset_name: str):
    time = np.asarray(npz[f"{dataset_name}__time_ms"], dtype=np.float64)
    release = np.asarray(npz[f"{dataset_name}__release_sv"], dtype=np.float64)
    if time.ndim != 1 or release.ndim != 1 or time.shape != release.shape:
        raise ValueError(f"{dataset_name}: time and release must be matching 1-D arrays")
    if time.size == 0 or not np.isclose(time[0], 0.0):
        raise ValueError(f"{dataset_name}: observations must start at time zero")
    if np.any(np.diff(time) <= 0.0) or not np.all(np.isfinite(time)):
        raise ValueError(f"{dataset_name}: times must be finite and strictly increasing")
    grid = time / DT
    if not np.allclose(grid, np.rint(grid), rtol=0.0, atol=1e-6):
        raise ValueError(f"{dataset_name}: times must lie on the {DT} ms grid")
    return time, release, np.asarray(np.rint(grid), dtype=np.int32)


def _equilibration_stim(equilibration_ms: float) -> np.ndarray:
    if not np.isfinite(equilibration_ms) or equilibration_ms < 0.0:
        raise ValueError("equilibration_ms must be finite and non-negative")
    steps = equilibration_ms / DT
    if not np.isclose(steps, np.rint(steps), rtol=0.0, atol=1e-6):
        raise ValueError(f"equilibration_ms must lie on the {DT} ms grid")
    return np.zeros((1, int(np.rint(steps))), dtype=np.float64)


def _make_split(npz, names: tuple[str, ...], equilibration_ms: float):
    protocols = [_load_protocol(npz, name) for name in names]
    n_protocols = len(protocols)
    n_observations = max(item[0].size for item in protocols)
    max_step = max(int(item[2][-1]) for item in protocols)

    time = np.full((1, n_protocols, n_observations), np.nan, dtype=np.float64)
    release = np.full_like(time, np.nan)
    event_index = np.full((1, n_protocols, n_observations), -1, dtype=np.int32)
    stim = np.zeros((1, n_protocols, max_step + 1), dtype=np.float64)

    for protocol_index, (protocol_time, protocol_release, indices) in enumerate(protocols):
        n = protocol_time.size
        time[0, protocol_index, :n] = protocol_time
        release[0, protocol_index, :n] = protocol_release
        event_index[0, protocol_index, :n] = indices
        stim[0, protocol_index, indices] = 1.0

    return {
        "release": release,
        "time": time,
        "stim": stim,
        "_event_index": event_index,
        "_equilibration_stim": _equilibration_stim(equilibration_ms),
    }


def load_data(
    data_path: str = "",
    discovery_equilibration_ms: float = DEFAULT_DISCOVERY_EQUILIBRATION_MS,
    validation_equilibration_ms: float = DEFAULT_VALIDATION_EQUILIBRATION_MS,
    **kwargs,
):
    """Load the two experiments and their protocol train/test splits.

    Discover contains datasets 01--04 and validation contains datasets 05--09.
    The train/test split is across protocols, while each split remains one EDGAR
    sample so its fitted parameters are shared across all protocols in that split.
    """
    del kwargs
    with np.load(_data_file(data_path), allow_pickle=False) as npz:
        disc_train = _make_split(
            npz,
            ("dataset_01", "dataset_03"),
            discovery_equilibration_ms,
        )
        disc_test = _make_split(
            npz,
            ("dataset_02", "dataset_04"),
            discovery_equilibration_ms,
        )
        val_train = _make_split(
            npz,
            ("dataset_05", "dataset_07", "dataset_09"),
            validation_equilibration_ms,
        )
        val_test = _make_split(
            npz,
            ("dataset_06", "dataset_08"),
            validation_equilibration_ms,
        )

    X_eval = {key: value.copy() for key, value in disc_train.items()}
    X_eval["_sample_indices"] = np.array([0], dtype=np.int32)
    return ((disc_train, disc_test), (val_train, val_test), X_eval)


def _initial_state(model_fn, params: dict) -> dict:
    """Build a model's fixed or structurally derived initial state."""
    initial_state_fn = getattr(model_fn, "INITIAL_STATE", None)
    if not callable(initial_state_fn):
        raise ValueError("synaptic models must define callable model.INITIAL_STATE")
    return initial_state_fn(params)


def _apply_legacy_model(model_fn, data, params):
    """Run the original one-dimensional dense-stimulus interface."""
    stim = data["stim"]
    release = data["release"]

    def per_sample(stim_s, release_s, params_s):
        init_hidden_state = _initial_state(model_fn, params_s)

        def step(carry, stim_prev):
            hidden_state, release_prev = carry
            new_hidden_state, release_next = model_fn(
                hidden_state,
                {"stim_prev": stim_prev, "release_prev": release_prev},
                params_s,
            )
            return (new_hidden_state, release_next), release_next

        initial_release = jnp.zeros_like(release_s[0])
        _, predictions = jax.lax.scan(
            step, (init_hidden_state, initial_release), stim_s
        )
        return predictions

    return jax.vmap(per_sample, in_axes=(0, 0, 0))(stim, release, params)


def _apply_protocol_model(model_fn, data, params):
    """Roll out each protocol on the dense grid and gather event-time outputs."""
    stim = data["stim"]
    event_index = data["_event_index"]
    equilibration_stim = data["_equilibration_stim"]

    def per_sample(stim_s, event_index_s, equilibration_stim_s, params_s):
        init_hidden_state = _initial_state(model_fn, params_s)

        def step(carry, stim_value):
            hidden_state, release_prev = carry
            new_hidden_state, release_next = model_fn(
                hidden_state,
                {"stim_prev": stim_value, "release_prev": release_prev},
                params_s,
            )
            return (new_hidden_state, release_next), release_next

        initial_release = jnp.zeros((), dtype=stim_s.dtype)
        (equilibrated_state, equilibrated_release), _ = jax.lax.scan(
            step,
            (init_hidden_state, initial_release),
            equilibration_stim_s,
        )

        def per_protocol(stim_p, event_index_p):
            _, dense_prediction = jax.lax.scan(
                step,
                (equilibrated_state, equilibrated_release),
                stim_p,
            )
            safe_index = jnp.maximum(event_index_p, 0)
            gathered = dense_prediction[safe_index]
            valid = event_index_p >= 0
            return jnp.where(valid, gathered, 0.0)

        return jax.vmap(per_protocol)(stim_s, event_index_s)

    return jax.vmap(per_sample, in_axes=(0, 0, 0, 0))(
        stim, event_index, equilibration_stim, params
    )


def apply_model(model_fn, data, params):
    """Apply a seed model to either legacy trajectories or protocol data."""
    if "_event_index" in data:
        return _apply_protocol_model(model_fn, data, params)
    return _apply_legacy_model(model_fn, data, params)


def loss_fn(model_output, data):
    """Sum protocol RMSEs, returning one loss for each EDGAR sample.

    NaN-padded observations are excluded. Protocols are summed after their
    individual RMSEs are calculated, so a long protocol does not receive more
    weight than a short protocol.
    """
    target = data["release"] #(n_samples, n_protcols, n_obs)
    valid = jnp.isfinite(target) & jnp.isfinite(data["time"])
    valid = valid & (data["_event_index"] >= 0)
    safe_target = jnp.where(valid, target, 0.0)
    residual = jnp.where(valid, model_output - safe_target, 0.0)
    squared_error = residual**2
    counts = jnp.sum(valid, axis=-1) #(n_samples, n_protocols)
    protocol_rmse = jnp.sqrt(
        jnp.sum(squared_error, axis=-1) / jnp.maximum(counts, 1)
    ) #(n_samples, n_protocols)
    return jnp.sum(protocol_rmse, axis=-1)
