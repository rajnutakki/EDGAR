"""Data loading and rollout hooks for synaptic-transmission discovery.

One EDGAR sample is one experiment.  Its observations are the protocols run on
that experiment, so one parameter set is shared across every protocol in a
sample.  Protocols have different recording times and are padded with NaNs;
``stim`` is the dense, 0.1 ms stimulus grid used by the rollout.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import csv

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


def _load_protocol(npz, dataset_name: str, equilibration_ms: float):
    """
        Returns: time (ms) - time in ms from start of the protocol, (n_observations,)
                 release (SV) - release observations at each time point (n_observations,)
                 grid - time indices on the DT grid, including equilibration time (n_observations,)
    """
    time = np.asarray(npz[f"{dataset_name}__time_ms"], dtype=np.float64)
    release = np.asarray(npz[f"{dataset_name}__release_sv"], dtype=np.float64)
    if time.ndim != 1 or release.ndim != 1 or time.shape != release.shape:
        raise ValueError(f"{dataset_name}: time and release must be matching 1-D arrays")
    if time.size == 0 or not np.isclose(time[0], 0.0):
        raise ValueError(f"{dataset_name}: observations must start at time zero")
    if np.any(np.diff(time) <= 0.0) or not np.all(np.isfinite(time)):
        raise ValueError(f"{dataset_name}: times must be finite and strictly increasing")
    equilibration_steps = equilibration_ms / DT
    grid = time / DT + equilibration_steps
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


def _make_split(npz, folds: tuple[tuple[str, ...], ...], equilibration_ms: float):
    """Build a batched split, with one leave-one-out fold per sample."""
    protocols_by_fold = [
        [_load_protocol(npz, name, equilibration_ms) for name in names]
        for names in folds
    ]
    n_samples = len(protocols_by_fold)
    n_protocols = max(len(protocols) for protocols in protocols_by_fold)
    n_observations = max(
        time.size
        for protocols in protocols_by_fold
        for (time, _, _) in protocols
    )
    max_observation_index = max(
        int(obs_indices[-1])
        for protocols in protocols_by_fold
        for (_, _, obs_indices) in protocols
    )

    time = np.full((n_samples, n_protocols, n_observations), np.nan, dtype=np.float64)
    release = np.full_like(time, np.nan)
    observation_index = np.full(
        (n_samples, n_protocols, n_observations), -1, dtype=np.int32
    )
    stim = np.zeros(
        (n_samples, n_protocols, max_observation_index + 1), dtype=np.float64
    )

    for sample_index, protocols in enumerate(protocols_by_fold):
        for protocol_index, (protocol_time, protocol_release, observation_indices) in enumerate(protocols):
            n_obs = protocol_time.size
            time[sample_index, protocol_index, :n_obs] = protocol_time
            release[sample_index, protocol_index, :n_obs] = protocol_release
            observation_index[sample_index, protocol_index, :n_obs] = observation_indices
            stim[sample_index, protocol_index, observation_indices] = 1.0

    return {
        "release": release,
        "time": time,
        "stim": stim,
        "_observation_index": observation_index,
        "_equilibration_time": np.full(
            (n_samples, n_protocols), equilibration_ms, dtype=np.float64
        ),
    }


def load_data(
    data_path: str = "",
    discovery_equilibration_ms: float = DEFAULT_DISCOVERY_EQUILIBRATION_MS,
    validation_equilibration_ms: float = DEFAULT_VALIDATION_EQUILIBRATION_MS,
    **kwargs,
):
    """Load leave-one-dataset-out protocol splits for discovery and validation.

    Each fold is an EDGAR sample with its own fitted parameters.  The test set
    holds out one dataset, and the scorer averages the fold losses.
    """
    discovery_datasets = ("dataset_01", "dataset_02", "dataset_03", "dataset_04")
    validation_datasets = (
        "dataset_05", "dataset_06", "dataset_07", "dataset_08", "dataset_09"
    )

    def leave_one_out_folds(datasets):
        return tuple(
            tuple(name for name in datasets if name != held_out)
            for held_out in datasets
        )

    with np.load(_data_file(data_path), allow_pickle=False) as npz:
        disc_train = _make_split(
            npz,
            leave_one_out_folds(discovery_datasets),
            discovery_equilibration_ms,
        )
        disc_test = _make_split(
            npz,
            tuple((name,) for name in discovery_datasets),
            discovery_equilibration_ms,
        )
        val_train = _make_split(
            npz,
            leave_one_out_folds(validation_datasets),
            validation_equilibration_ms,
        )
        val_test = _make_split(
            npz,
            tuple((name,) for name in validation_datasets),
            validation_equilibration_ms,
        )

    X_eval = {key: value[:1].copy() for key, value in disc_train.items()}
    X_eval["_sample_indices"] = np.array([0], dtype=np.int32)
    return ((disc_train, disc_test), (val_train, val_test), X_eval)


def _initial_state(model_fn, params: dict) -> dict:
    """Build a model's fixed or structurally derived initial state."""
    initial_state_fn = getattr(model_fn, "INITIAL_STATE", None)
    if not callable(initial_state_fn):
        raise ValueError("synaptic models must define callable model.INITIAL_STATE")
    return initial_state_fn(params)

@partial(jax.jit, static_argnames=("model_fn",))
def apply_model(model_fn, data, params):
    stim = data["stim"]

    def per_sample(stim_s, params_s):

        init_hidden_state = _initial_state(model_fn, params_s)

        def step(carry, stim):
            hidden_state, release_prev = carry
            new_hidden_state, release_next = model_fn(
                hidden_state,
                {"stim_prev": stim, "release_prev": release_prev},
                params_s,
            )
            output = {**new_hidden_state, "release": release_next}
            return (new_hidden_state, release_next), output

        def per_protocol(stim_p):
            initial_release = jnp.zeros((), dtype=stim_p.dtype)
            _, dense_output = jax.lax.scan(
                step,
                (init_hidden_state, initial_release),
                stim_p,
            )
            dense_output["time"] = jnp.arange(stim_p.shape[0]) * DT
            return dense_output

        return jax.vmap(per_protocol)(stim_s)

    return jax.vmap(per_sample, in_axes=(0, 0))(stim, params)

def loss_fn(model_output, data):
    """Calculate per-sample RMSE over all non-padded observations."""
    target = data["release"]  # (n_samples, n_protocols, n_observations)
    observation_index = data["_observation_index"]
    valid = (observation_index >= 0) & jnp.isfinite(target)

    # Preserve the sample and protocol axes while selecting from dense rollout time.
    safe_index = jnp.maximum(observation_index, 0)
    predicted_release = jnp.take_along_axis(
        model_output["release"], safe_index, axis=-1
    )
    residual = jnp.where(valid, predicted_release - target, 0.0)
    squared_error = residual**2
    observation_axes = tuple(range(1, squared_error.ndim))
    counts = jnp.sum(valid, axis=observation_axes)
    mean_squared_error = (
        jnp.sum(squared_error, axis=observation_axes) / jnp.maximum(counts, 1)
    )
    return jnp.sqrt(mean_squared_error)

def _read_bm_parameters(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["name", "value"]:
            raise ValueError(
                f"Expected columns 'name,value' in {path}, got {reader.fieldnames}"
            )
        for row in reader:
            name = row["name"].strip()
            if name in values:
                raise ValueError(f"Duplicate parameter {name!r} in {path}")
            value = float(row["value"])
            if not np.isfinite(value):
                raise ValueError(f"Parameter {name!r} is not finite in {path}")
            values[name] = value
    return values
