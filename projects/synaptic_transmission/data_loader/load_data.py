"""EDGAR data and rollout hooks for synaptic-transmission discovery.

The intended data layout is two experiments, each containing two protocols:

    experiment 0, protocol 0 -> discover train
    experiment 0, protocol 1 -> discover test
    experiment 1, protocol 0 -> validate train
    experiment 1, protocol 1 -> validate test

Each trajectory may have a different duration.  Consequently, each returned
split contains one EDGAR sample and one one-dimensional trajectory rather than
one padded array containing all four trajectories.  The eventual data-loading
implementation will populate ``stim`` and ``release`` with arrays of shape
``(1, T_split)``.

The loader itself is intentionally left as a placeholder until the on-disk
format is settled.  The rollout and loss functions below are independent of
that format.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def load_data(data_path: str = "", **kwargs):
    """Placeholder for the synaptic-transmission data loader.

    Expected eventual return value::

        ((X_disc_train, X_disc_test),
         (X_val_train, X_val_test),
         X_eval)

    Every split dictionary should contain ``stim`` and ``release`` arrays of
    shape ``(1, T_split)``.  The four trajectories may have different lengths.
    ``X_eval`` must additionally contain ``_sample_indices``.
    """
    raise NotImplementedError(
        "Synaptic-transmission data loading is intentionally deferred. "
        "Implement the project-specific file format here."
    )


def _split_params_s0(params: dict) -> tuple[dict, dict]:
    """Split learnable initial-state parameters from dynamics parameters."""
    init_hidden_state = {}
    dyn_params = {}
    for key, value in params.items():
        if key.startswith("s0_") and len(key) > 3:
            init_hidden_state[key[3:]] = value
        else:
            dyn_params[key] = value
    return init_hidden_state, dyn_params


def apply_model(model_fn, data, params):
    """Run a full free rollout for every EDGAR sample.

    ``data`` contains ``stim`` and ``release`` with shape ``(n_samples, T)``.
    The previous observed release is provided in ``y_prev`` for interface
    compatibility, although the current seed models use the stimulus and their
    carried reservoir state to determine release.

    The observable is seeded with the first recorded release for interface
    compatibility.  Thereafter the model's own prediction is carried forward.
    The returned prediction at index ``t`` uses ``stim[t]`` and the hidden
    state at the same time, and therefore corresponds to ``release[t]``.
    """
    stim = data["stim"]
    release = data["release"]

    def per_sample(stim_s, release_s, params_s):
        init_hidden_state, dyn_params = _split_params_s0(params_s)

        def step(carry, stim_prev):
            hidden_state, release_prev = carry
            y_prev = {
                "stim_prev": stim_prev,
                "release_prev": release_prev,
            }
            new_hidden_state, release_next = model_fn(
                hidden_state, y_prev, dyn_params
            )
            return (new_hidden_state, release_next), release_next

        xs = stim_s
        init_carry = (init_hidden_state, release_s[0])
        _, predictions = jax.lax.scan(step, init_carry, xs)
        return predictions

    return jax.vmap(per_sample, in_axes=(0, 0, 0))(stim, release, params)


def loss_fn(model_output, data):
    """Return full-trajectory RMSE for each EDGAR sample.

    ``model_output[:, t]`` predicts ``data["release"][:, t]``.  The reduction
    retains only the leading sample axis, as required by EDGAR.
    """
    target = data["release"]
    return jnp.sqrt(jnp.mean((model_output - target) ** 2, axis=-1))
