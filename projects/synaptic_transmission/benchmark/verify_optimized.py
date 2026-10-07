"""Compare optimized synaptic seed models and reference RP parameter sets.

Edit ``RP_PARAMETER_SET_FILES`` to select the external RP fits to display. The
script performs each model's full dense rollout, including its unstimulated
equilibration period, and samples predictions only at experimental event times.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
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

DATA_PATH = REFERENCE_ROOT / "dataset" / "data.npz"
SCORED_SEEDS_PATH = PROJECT_DIR / "scored_seeds.jsonl"
RP_PARAMETER_SET_FILES = (
    Path("parameter_sets/RP/MZM_FSIN_4Prot_v11b.csv"),
    Path("parameter_sets/RP/MZM_FSIN_HighPv_v12c.csv"),
)
DATASET_NAMES = tuple(f"dataset_{index:02d}" for index in range(1, 10))
OUTPUT_PATH = HERE / "verify_optimized.png"

sys.path.insert(0, str(PROJECT_DIR / "data_loader"))
sys.path.insert(0, str(REFERENCE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from edgar.evolution.population import Population
from edgar.llm.utils import translate_to_jax
import run_models
from load_data import (
    DEFAULT_DISCOVERY_EQUILIBRATION_MS,
    DEFAULT_VALIDATION_EQUILIBRATION_MS,
    _make_split,
    apply_model,
)


def _load_rp_model():
    """Load the verified RP seed implementation as a JAX model."""
    source_path = PROJECT_DIR / "seed_programs" / "model2.py"
    source = translate_to_jax(source_path.read_text(encoding="ascii"))
    namespace = {"__name__": "translated_reference_rp"}
    exec(compile(source, str(source_path), "exec"), namespace)
    return SimpleNamespace(model=namespace["model"], DT=namespace["DT"])


def _batched_params(params: dict) -> dict:
    batched = {}
    for key, value in params.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = array[None]
        if array.shape != (1,):
            raise ValueError(f"parameter {key!r} has shape {array.shape}, expected (1,)")
        batched[key] = jnp.asarray(array)
    return batched


def _predict(model_fn, npz, names, equilibration_ms, params):
    data = _make_split(npz, tuple(names), equilibration_ms)
    prediction = np.asarray(
        apply_model(model_fn, data, _batched_params(params))[0],
        dtype=np.float64,
    )
    return {
        name: prediction[index, : npz[f"{name}__time_ms"].size]
        for index, name in enumerate(names)
    }


def _reference_models(npz):
    rp_seed = _load_rp_model()
    models = []
    for relative_path in RP_PARAMETER_SET_FILES:
        path = REFERENCE_ROOT / relative_path
        params, protocol, sim = run_models.read_parameter_set(
            str(path), run_models.MODELS["RP"]
        )
        if not np.isclose(sim["DT"], rp_seed.DT, rtol=0.0, atol=0.0):
            raise ValueError(f"{path}: DT {sim['DT']} differs from model DT {rp_seed.DT}")
        delays = [
            protocol[f"Pt{index}_delay1"]
            for index in range(1, 6)
            if protocol[f"Pt{index}_StimNum1"] > 0.0
        ]
        if not delays or not np.allclose(delays, delays[0]):
            raise ValueError(f"{path}: expected one common first-train delay")
        dynamics_params = {
            key: value for key, value in params.items() if not key.startswith("INIT_")
        }
        predictions = _predict(
            rp_seed.model,
            npz,
            DATASET_NAMES,
            float(delays[0]),
            dynamics_params,
        )
        models.append((f"RP {path.stem}", predictions))
    return models


def _scored_seed_models(npz):
    if not SCORED_SEEDS_PATH.exists():
        raise FileNotFoundError(
            f"{SCORED_SEEDS_PATH} does not exist; generate it with "
            "`uv run python scripts/score_seeds.py synaptic_transmission 1`"
        )

    models = []
    for program in Population.load(str(SCORED_SEEDS_PATH)):
        if program.params is None:
            raise ValueError(f"{program.name} has no optimized parameters")
        if any(key.startswith("s0_") for key in program.params):
            raise ValueError(
                f"{program.name} uses obsolete s0_* parameters; regenerate "
                f"{SCORED_SEEDS_PATH.name} with the fixed-state seed models"
            )
        model_fn = program.compile_model()
        if not callable(getattr(model_fn, "INITIAL_STATE", None)):
            raise ValueError(f"{program.name} does not define model.INITIAL_STATE")

        predictions = {}
        predictions.update(
            _predict(
                model_fn,
                npz,
                DATASET_NAMES[:4],
                DEFAULT_DISCOVERY_EQUILIBRATION_MS,
                program.params,
            )
        )
        predictions.update(
            _predict(
                model_fn,
                npz,
                DATASET_NAMES[4:],
                DEFAULT_VALIDATION_EQUILIBRATION_MS,
                program.params,
            )
        )
        models.append((program.name, predictions))
    return models


def _rmse(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} differs from target {target.shape}"
        )
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def main():
    with np.load(DATA_PATH, allow_pickle=False) as npz:
        models = _reference_models(npz) + _scored_seed_models(npz)
        fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)

        print("dataset\tmodel\trmse")
        for axis, dataset_name in zip(axes.flat, DATASET_NAMES):
            time = np.asarray(npz[f"{dataset_name}__time_ms"], dtype=np.float64)
            release = np.asarray(
                npz[f"{dataset_name}__release_sv"], dtype=np.float64
            )
            condition = str(npz[f"{dataset_name}__synapse_condition"].item())
            protocol = str(npz[f"{dataset_name}__protocol"].item())

            axis.plot(time, release, "ko-", linewidth=1.8, markersize=4, label="data")
            for label, predictions in models:
                prediction = predictions[dataset_name]
                error = _rmse(prediction, release)
                print(f"{dataset_name}\t{label}\t{error:.8g}")
                axis.plot(time, prediction, "o-", markersize=3, label=f"{label} ({error:.3g})")

            axis.set_title(f"{dataset_name}: {condition}\n{protocol}", fontsize=10)
            axis.set_xlabel("Time (ms)")
            axis.set_ylabel("Release (SV)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Experimental release and equilibrated model rollouts", fontsize=14)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
