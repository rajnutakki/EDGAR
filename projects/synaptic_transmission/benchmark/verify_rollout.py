"""Verify EDGAR seed-model rollouts against the reference implementation.

The reference project owns the parameter-set parsing, BM-compatible time grid,
and stimulus protocol generation. This benchmark feeds those exact stimulus
values into the EDGAR ``apply_model`` implementation and compares the complete
release trajectory, including the value at STOPTIME.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jax

# Match the precision of the reference NumPy implementation over long runs.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
REPO_ROOT = PROJECT_DIR.parent.parent
REFERENCE_ROOT = Path(
    os.environ.get(
        "SYNAPTIC_TRANSMISSION_REFERENCE",
        "/home/rajah/projects/synaptic_transmission",
    )
).resolve()

sys.path.insert(0, str(PROJECT_DIR / "seed_programs"))
sys.path.insert(0, str(PROJECT_DIR / "data_loader"))
sys.path.insert(0, str(REFERENCE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from edgar.llm.utils import translate_to_jax
import run_models
from load_data import apply_model


SEED_MODEL_PATHS = {
    "RP": PROJECT_DIR / "seed_programs" / "model2.py",
    "Neher": PROJECT_DIR / "seed_programs" / "model1.py",
}

# The seed programs integrate with their own fixed DT. Keep this explicit so a
# parameter set cannot silently be verified with a different timestep.
ROLLOUT_ATOL = 1e-12
ROLLOUT_RTOL = 1e-10


def _load_translated_model(model_name):
    """Translate a NumPy seed source in memory and load its model function."""
    source_path = SEED_MODEL_PATHS[model_name]
    source = translate_to_jax(source_path.read_text(encoding="ascii"))
    namespace = {"__name__": f"translated_{model_name.lower()}_seed"}
    exec(compile(source, str(source_path), "exec"), namespace)
    return SimpleNamespace(model=namespace["model"], DT=namespace["DT"])


def _seed_params(reference_model, seed_model, params):
    """Remove reference INIT values handled by the seed's initialization hook."""
    seed_params = {
        key: value for key, value in params.items() if not key.startswith("INIT_")
    }

    if not np.isclose(reference_model.DT, seed_model.DT, rtol=0.0, atol=0.0):
        raise ValueError(
            f"reference DT {reference_model.DT!r} differs from seed DT "
            f"{seed_model.DT!r}"
        )
    return seed_params


def _rollout(seed_model, stimulus, params):
    """Run one full seed-model rollout with one sample in the batch."""
    data = {
        "stim": jnp.asarray(stimulus[None, :]),
        # The current seed models do not use release_prev, but apply_model's
        # interface requires an observed first value to seed that carry.
        "release": jnp.zeros((1, stimulus.size), dtype=jnp.float64),
    }
    batched_params = {key: jnp.asarray([value]) for key, value in params.items()}
    return np.asarray(apply_model(seed_model.model, data, batched_params)[0])


def verify_case(model_name, parameter_path):
    """Verify one reference parameter set and return its error summary."""
    reference_model = run_models.MODELS[model_name]
    seed_model = _load_translated_model(model_name)
    params, protocol, sim = run_models.read_parameter_set(parameter_path, reference_model)

    if not np.isclose(sim["DT"], seed_model.DT, rtol=0.0, atol=0.0):
        raise ValueError(
            f"{parameter_path}: parameter-set DT {sim['DT']!r} differs from "
            f"seed DT {seed_model.DT!r}"
        )

    reference = run_models.simulate(reference_model, params, protocol, sim)
    predicted = _rollout(
        seed_model,
        reference["P"],
        _seed_params(reference_model, seed_model, params),
    )
    expected = reference[reference_model.OUTPUT_NAME]

    if predicted.shape != expected.shape:
        raise AssertionError(
            f"{parameter_path}: seed rollout shape {predicted.shape} differs "
            f"from reference shape {expected.shape}"
        )
    if not np.all(np.isfinite(predicted)):
        raise AssertionError(f"{parameter_path}: seed release contains non-finite values")

    difference = np.abs(predicted - expected)
    np.testing.assert_allclose(
        predicted,
        expected,
        rtol=ROLLOUT_RTOL,
        atol=ROLLOUT_ATOL,
        err_msg=f"{model_name} seed rollout differs from {parameter_path}",
    )
    return expected.size, float(difference.max())


def main():
    results = []
    for model_name, parameter_file, _ in run_models.RUNS:
        parameter_path = REFERENCE_ROOT / parameter_file
        n_values, max_error = verify_case(model_name, str(parameter_path))
        results.append((model_name, parameter_file, n_values, max_error))
        print(
            f"PASS  {model_name} {parameter_file}  "
            f"{n_values} releases; max |seed - reference| {max_error:.3e}"
        )

    print(f"RESULT: PASS ({len(results)} of {len(run_models.RUNS)} runs match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
