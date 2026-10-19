"""Score one synaptic-transmission seed from Berkeley Madonna parameters.

Only one seed and parameter set is scored per invocation because these
optimizations are expensive. Each initial parameter is independently multiplied
by a random factor drawn from ``Normal(mean=1, std=strength)``.

Examples::

    uv run python projects/synaptic_transmission/benchmark/score_seeds_BM_params.py \
        --seed 1 --parameter-set NM_FSIN_4Pr_v3 --strength 0.1 --random-seed 42

    uv run python projects/synaptic_transmission/benchmark/score_seeds_BM_params.py \
        --seed 2 --parameter-set MZM_FSIN_4Prot_v11b --output rp_scored.jsonl
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        _xla_flags + " --xla_gpu_enable_command_buffer="
    ).strip()

if not hasattr(sys.modules["__main__"], "__spec__"):
    sys.modules["__main__"].__spec__ = None

from edgar.evolution.population import Population
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.utils import translate_to_jax
from edgar.scoring.scoring import score


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
DEFAULT_REFERENCE_ROOT = Path(
    os.environ.get(
        "SYNAPTIC_TRANSMISSION_REFERENCE",
        "/home/rajah/projects/synaptic_transmission",
    )
)

SEED_MODELS = {
    1: "Neher",
    2: "RP",
}
SAVE_TRAJECTORIES = True
RECIPROCAL_BM_PARAMETERS = ("b3", "cadecay")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one synaptic-transmission seed using one Berkeley Madonna "
            "parameter set as the optimizer's initial parameters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=sorted(SEED_MODELS),
        required=True,
        help="Seed model number (1 = Neher, 2 = replacement-pool model).",
    )
    parser.add_argument(
        "--parameter-set",
        required=True,
        help=(
            "CSV path, file name, or stem under parameter_sets/Neher or "
            "parameter_sets/RP. HighPv sets are intentionally unsupported."
        ),
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.0,
        help=(
            "Standard deviation of the independent Normal(mean=1, std=strength) "
            "multipliers applied to every parameter. Use 0 for no perturbation."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducible perturbations; defaults to config.run.random_seed.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Root of the external synaptic_transmission reference project.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path at which to save the one-program scored population.",
    )
    return parser.parse_args()


def _resolve_parameter_set(
    selection: str, seed_number: int, reference_root: Path
) -> Path:
    """Resolve a CSV selection and enforce its compatibility with the seed."""
    if "highpv" in selection.lower():
        raise ValueError("HighPv parameter sets are not applicable to this benchmark")

    model_name = SEED_MODELS[seed_number]
    parameter_dir = (reference_root / "parameter_sets" / model_name).resolve()
    if not parameter_dir.is_dir():
        raise FileNotFoundError(f"Parameter-set directory does not exist: {parameter_dir}")

    supplied = Path(selection).expanduser()
    if supplied.is_absolute() or supplied.parent != Path("."):
        candidate = supplied.resolve()
        try:
            candidate.relative_to(parameter_dir)
        except ValueError as exc:
            raise ValueError(
                f"Seed {seed_number} requires a parameter set from {parameter_dir}"
            ) from exc
        candidates = [candidate]
    else:
        candidates = [
            path
            for path in parameter_dir.glob("*.csv")
            if path.name == selection or path.stem == selection
        ]

    candidates = [path for path in candidates if "highpv" not in path.name.lower()]
    if not candidates:
        available = ", ".join(
            sorted(
                path.stem
                for path in parameter_dir.glob("*.csv")
                if "highpv" not in path.name.lower()
            )
        )
        raise FileNotFoundError(
            f"Could not resolve parameter set {selection!r}. Available: {available}"
        )
    if len(candidates) > 1:
        raise ValueError(f"Parameter-set selection {selection!r} is ambiguous")

    path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"Parameter-set file does not exist: {path}")
    return path


def _read_bm_parameters(path: Path) -> dict[str, float]:
    """Read a two-column Berkeley Madonna parameter CSV."""
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
            if not math.isfinite(value):
                raise ValueError(f"Parameter {name!r} is not finite in {path}")
            values[name] = value
    return values


def _model_parameters(
    bm_values: dict[str, float], expected_names: tuple[str, ...], path: Path
) -> dict[str, float]:
    """Convert Berkeley Madonna parameters to the EDGAR seed convention."""
    missing = set(expected_names) - bm_values.keys()
    if missing:
        raise ValueError(f"Missing model parameters in {path}: {sorted(missing)}")

    if "DT" not in bm_values or bm_values["DT"] != 0.1:
        raise ValueError(f"{path} must define DT=0.1 to match the EDGAR seeds")

    # STARTTIME, STOPTIME, INIT_*, Crest, and Pt* protocol settings belong to
    # the BM simulation rather than the EDGAR model parameter dictionary.
    params = {name: bm_values[name] for name in expected_names}
    for name in RECIPROCAL_BM_PARAMETERS:
        if name not in params:
            continue
        if params[name] <= 0.0:
            raise ValueError(f"Parameter {name!r} must be positive in {path}")
        params[name] = 1.0 / params[name]
    return params


def _perturb_parameters(
    params: dict[str, float], strength: float, rng: np.random.Generator
) -> tuple[dict[str, float], dict[str, float]]:
    """Multiply every parameter by an independent N(1, strength) draw."""
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("Perturbation strength must be finite and non-negative")

    names = tuple(params)
    factors_array = rng.normal(loc=1.0, scale=strength, size=len(names))
    factors = {
        name: float(factor) for name, factor in zip(names, factors_array, strict=True)
    }
    perturbed = {name: params[name] * factors[name] for name in names}
    return perturbed, factors


def _parameter_estimator_source(params: dict[str, float]) -> str:
    """Create source for EDGAR's subprocess-compiled parameter estimator."""
    entries = ",\n".join(
        f"        {name!r}: {value!r}" for name, value in params.items()
    )
    return (
        "def parameter_estimator(data):\n"
        "    # Fixed BM initialization; data is intentionally unused.\n"
        "    return {\n"
        f"{entries}\n"
        "    }\n"
    )


def score_seed(
    seed_number: int,
    parameter_set: str,
    strength: float = 0.0,
    random_seed: int | None = None,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    output: Path | None = None,
) -> None:
    """Score one BM-initialized seed and optionally save its population."""
    parameter_path = _resolve_parameter_set(
        parameter_set, seed_number, reference_root.resolve()
    )

    config_path = PROJECT_DIR / "config.yaml"
    print(f"Loading config from: {config_path}")
    config = Config.from_yaml(config_path)
    spec = TaskSpec.from_config(config)

    program = copy.deepcopy(spec.seed_programs[seed_number - 1])
    expected_names = tuple(program.default_params)
    bm_values = _read_bm_parameters(parameter_path)
    initial_params = _model_parameters(bm_values, expected_names, parameter_path)
    if random_seed is None:
        random_seed = spec.run["random_seed"]
    initial_params, perturbation_factors = _perturb_parameters(
        initial_params, strength, np.random.default_rng(random_seed)
    )

    print(f"Seed: {seed_number} ({SEED_MODELS[seed_number]})")
    print(f"Parameter set: {parameter_path}")
    print(f"Perturbation: Normal(mean=1, std={strength:g}), random seed={random_seed}")
    for name in expected_names:
        print(
            f"  {name}: factor={perturbation_factors[name]:.8g}, "
            f"value={initial_params[name]:.12g}"
        )

    program.name = (
        f"Seed Model {seed_number} | {parameter_path.stem} | "
        f"normal std={strength:g} rng={random_seed}"
    )
    program.code.param_est = _parameter_estimator_source(initial_params)
    print("Translating model to JAX...")
    program.code.model_jax = translate_to_jax(program.code.model)

    print("Loading data...")
    X_discover, _X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )

    population = Population()
    population.add(program)
    print("Scoring discovery split...")
    score(
        population,
        X_discover,
        X_eval,
        spec.scoring,
        spec.loss_fn,
        split="discover",
        apply_model_fn=spec.apply_model_fn,
    )

    loss = program.program_losses.discover
    print("Scoring complete")
    print(f"Initial discovery score: {loss.init}")
    print(f"Final discovery score:   {loss.final}")

    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving scored population to: {output}")
        population.save(output, save_trajectories = SAVE_TRAJECTORIES)


def main() -> None:
    args = _parse_args()
    score_seed(
        seed_number=args.seed,
        parameter_set=args.parameter_set,
        strength=args.strength,
        random_seed=args.random_seed,
        reference_root=args.reference_root,
        output=args.output,
    )


if __name__ == "__main__":
    main()
