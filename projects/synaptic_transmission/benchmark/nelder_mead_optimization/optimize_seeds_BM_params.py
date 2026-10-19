"""Optimize a synaptic-transmission seed with SciPy's Nelder-Mead method.

This is a standalone counterpart to ``optimization/score_seeds_BM_params.py``:
it reads a NumPy seed model and the experimental data directly, without using
EDGAR's population, scoring, JAX translation, or parameter optimization code.

Example::

    uv run python projects/synaptic_transmission/benchmark/nelder_mead_optimization/optimize_seeds_BM_params.py \
        --seed 1 --parameter-set NM_FSIN_4Pr_v3 --strength 0.1 \
        --random-seed 42 --output neher_nelder_mead.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
DEFAULT_REFERENCE_ROOT = Path(
    os.environ.get(
        "SYNAPTIC_TRANSMISSION_REFERENCE",
        "/home/rajah/projects/synaptic_transmission",
    )
)
DEFAULT_DATA_PATH = DEFAULT_REFERENCE_ROOT / "dataset" / "data.npz"

SEED_MODELS = {
    1: ("Neher", PROJECT_DIR / "seed_programs" / "model1.py"),
    2: ("RP", PROJECT_DIR / "seed_programs" / "model2.py"),
}
DISCOVERY_PROTOCOL_ORDER = (
    "dataset_04",
    "dataset_03",
    "dataset_02",
    "dataset_01",
)
DT = 0.1
DISCOVERY_EQUILIBRATION_MS = 30_000.0
DISCOVERY_PROTOCOL_TAIL_MS = 10.0
INVALID_LOSS = 1e100
RECIPROCAL_BM_PARAMETERS = ("b3", "cadecay")


@dataclass(frozen=True)
class Protocol:
    event_indices: np.ndarray
    release: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one synaptic-transmission seed from Berkeley Madonna "
            "parameters using scipy.optimize.minimize(method='Nelder-Mead')."
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
            "Standard deviation of independent Normal(mean=1, std=strength) "
            "multipliers applied before optimization."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for parameter perturbations.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Root of the external synaptic_transmission reference project.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Experimental data.npz path.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum number of Nelder-Mead iterations.",
    )
    parser.add_argument(
        "--maxfev",
        type=int,
        help="Optional maximum number of objective evaluations.",
    )
    parser.add_argument("--xatol", type=float, default=1e-4)
    parser.add_argument("--fatol", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to save results. Defaults to "
            "<HERE>/<model>_<parameter_set>_perturbation_<strength>.jsonl"
        ),
    )
    return parser.parse_args()


def _resolve_parameter_set(
    selection: str, seed_number: int, reference_root: Path
) -> Path:
    if "highpv" in selection.lower():
        raise ValueError("HighPv parameter sets are not applicable to this benchmark")

    model_name = SEED_MODELS[seed_number][0]
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
    if not candidates[0].is_file():
        raise FileNotFoundError(f"Parameter-set file does not exist: {candidates[0]}")
    return candidates[0]


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
            if not math.isfinite(value):
                raise ValueError(f"Parameter {name!r} is not finite in {path}")
            values[name] = value
    return values


def _load_seed(seed_number: int) -> tuple[str, ModuleType, Callable]:
    model_name, path = SEED_MODELS[seed_number]
    spec = importlib.util.spec_from_file_location(f"synaptic_seed_{seed_number}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load seed model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_fn = module.model
    if not callable(model_fn) or not callable(getattr(model_fn, "INITIAL_STATE", None)):
        raise ValueError(f"Seed model {path} does not expose the expected model interface")
    if not np.isclose(float(module.DT), DT, rtol=0.0, atol=0.0):
        raise ValueError(f"Seed model DT={module.DT} does not match benchmark DT={DT}")
    return model_name, module, model_fn


def _initial_parameters(
    module: ModuleType, bm_values: dict[str, float], parameter_path: Path
) -> dict[str, float]:
    names = tuple(module.model.DEFAULT_PARAMS)
    missing = set(names) - bm_values.keys()
    if missing:
        raise ValueError(f"Missing model parameters in {parameter_path}: {sorted(missing)}")
    if "DT" not in bm_values or not np.isclose(
        bm_values["DT"], DT, rtol=0.0, atol=0.0
    ):
        raise ValueError(f"{parameter_path} must define DT={DT}")
    params = {name: bm_values[name] for name in names}
    for name in RECIPROCAL_BM_PARAMETERS:
        if name not in params:
            continue
        if params[name] <= 0.0:
            raise ValueError(f"Parameter {name!r} must be positive in {parameter_path}")
        params[name] = 1.0 / params[name]
    nonpositive = [name for name, value in params.items() if value <= 0.0]
    if nonpositive:
        raise ValueError(
            "Nelder-Mead multiplier scaling requires positive initial values; "
            f"non-positive parameters: {nonpositive}"
        )
    return params


def _perturb_parameters(
    params: dict[str, float], strength: float, rng: np.random.Generator
) -> tuple[dict[str, float], dict[str, float]]:
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("Perturbation strength must be finite and non-negative")
    names = tuple(params)
    factors_array = rng.normal(loc=1.0, scale=strength, size=len(names))
    if np.any(factors_array <= 0.0):
        raise ValueError(
            "Perturbation produced a non-positive parameter multiplier; use a "
            "smaller --strength or a different --random-seed"
        )
    factors = {
        name: float(factor) for name, factor in zip(names, factors_array, strict=True)
    }
    return ({name: params[name] * factors[name] for name in names}, factors)


def _load_protocols(data_path: Path) -> tuple[Protocol, ...]:
    data_path = data_path.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {data_path}")

    protocols = []
    with np.load(data_path, allow_pickle=False) as npz:
        for name in DISCOVERY_PROTOCOL_ORDER:
            time = np.asarray(npz[f"{name}__time_ms"], dtype=np.float64)
            release = np.asarray(npz[f"{name}__release_sv"], dtype=np.float64)
            if time.ndim != 1 or release.shape != time.shape or time.size == 0:
                raise ValueError(f"{name}: time and release must be matching nonempty arrays")
            grid = time / DT
            if not np.all(np.isfinite(time)) or not np.allclose(
                grid, np.rint(grid), rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"{name}: times must be finite and lie on the {DT} ms grid")
            if not np.isclose(time[0], 0.0) or np.any(np.diff(time) <= 0.0):
                raise ValueError(f"{name}: times must start at zero and strictly increase")
            if not np.all(np.isfinite(release)):
                raise ValueError(f"{name}: release observations must be finite")
            protocols.append(
                Protocol(np.asarray(np.rint(grid), dtype=np.int64), release)
            )
    return tuple(protocols)


def _rollout(
    model_fn: Callable, params: dict[str, float], protocols: tuple[Protocol, ...]
) -> np.ndarray:
    """Reproduce the discovery split's continuous dense NumPy rollout."""
    state = model_fn.INITIAL_STATE(params)
    release_previous = 0.0
    predictions = []
    equilibration_steps = round(DISCOVERY_EQUILIBRATION_MS / DT)
    tail_steps = round(DISCOVERY_PROTOCOL_TAIL_MS / DT)

    # The data loader places the first pulse one grid row after equilibration.
    zero_steps = equilibration_steps + 1
    for protocol in protocols:
        previous_event_index = 0
        for event_number, event_index in enumerate(protocol.event_indices):
            if event_number == 0:
                steps_before_event = zero_steps
            else:
                steps_before_event = int(event_index) - previous_event_index - 1
            for _ in range(steps_before_event):
                state, release_previous = model_fn(
                    state,
                    {"stim_prev": 0.0, "release_prev": release_previous},
                    params,
                )
            state, release_previous = model_fn(
                state,
                {"stim_prev": 1.0, "release_prev": release_previous},
                params,
            )
            predictions.append(float(release_previous))
            previous_event_index = int(event_index)

        # Between protocols the loader adds the tail and a new equilibration.
        zero_steps = tail_steps + equilibration_steps - 1

    return np.asarray(predictions, dtype=np.float64)


def _optimize(
    model_fn: Callable,
    initial_params: dict[str, float],
    protocols: tuple[Protocol, ...],
    maxiter: int,
    maxfev: int | None,
    xatol: float,
    fatol: float,
) -> tuple[OptimizeResult, float, np.ndarray, int, list[dict[str, float]], float]:
    if maxiter <= 0 or (maxfev is not None and maxfev <= 0):
        raise ValueError("--maxiter and --maxfev must be positive")
    if xatol < 0.0 or fatol < 0.0:
        raise ValueError("--xatol and --fatol must be non-negative")

    names = tuple(initial_params)
    scales = np.asarray([initial_params[name] for name in names], dtype=np.float64)
    targets = np.concatenate([protocol.release for protocol in protocols])
    evaluations = 0
    best_loss = math.inf

    def objective(multipliers: np.ndarray) -> float:
        nonlocal evaluations, best_loss
        evaluations += 1
        params = {
            name: float(value)
            for name, value in zip(names, scales * multipliers, strict=True)
        }
        try:
            with np.errstate(all="ignore"):
                prediction = _rollout(model_fn, params, protocols)
                loss = float(np.mean((prediction - targets) ** 2))
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError):
            loss = INVALID_LOSS
        if not math.isfinite(loss):
            loss = INVALID_LOSS
        if loss < best_loss:
            best_loss = loss
            print(f"Evaluation {evaluations}: best MSE={loss:.12g}", flush=True)
        return loss

    start_time = time.perf_counter()
    initial_loss = objective(np.ones(len(names), dtype=np.float64))
    initial_rmse = (
        math.sqrt(initial_loss)
        if math.isfinite(initial_loss) and initial_loss >= 0.0
        else INVALID_LOSS
    )
    trajectory: list[dict[str, float]] = [
        {
            "iteration": 0,
            "rmse": initial_rmse,
            "time_seconds": time.perf_counter() - start_time,
        }
    ]

    iteration = 0

    def callback(intermediate_result: OptimizeResult) -> None:
        nonlocal iteration
        iteration += 1
        elapsed = time.perf_counter() - start_time
        fun_val = float(intermediate_result.fun)
        rmse_val = (
            math.sqrt(fun_val)
            if math.isfinite(fun_val) and fun_val >= 0.0
            else INVALID_LOSS
        )
        trajectory.append({
            "iteration": iteration,
            "rmse": rmse_val,
            "time_seconds": elapsed,
        })

    options: dict[str, float | int] = {
        "maxiter": maxiter,
        "xatol": xatol,
        "fatol": fatol,
        "disp": True,
    }
    if maxfev is not None:
        options["maxfev"] = maxfev
    result = minimize(
        objective,
        np.ones(len(names), dtype=np.float64),
        method="Nelder-Mead",
        bounds=[(np.finfo(np.float64).eps, None)] * len(names),
        callback=callback,
        options=options,
    )
    total_time = time.perf_counter() - start_time
    if result.nit > iteration:
        fun_val = float(result.fun)
        rmse_val = (
            math.sqrt(fun_val)
            if math.isfinite(fun_val) and fun_val >= 0.0
            else INVALID_LOSS
        )
        trajectory.append({
            "iteration": int(result.nit),
            "rmse": rmse_val,
            "time_seconds": total_time,
        })

    return result, initial_loss, scales, evaluations, trajectory, total_time


def optimize_seed(
    seed_number: int,
    parameter_set: str,
    strength: float = 0.0,
    random_seed: int = 42,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    data_path: Path = DEFAULT_DATA_PATH,
    maxiter: int = 200,
    maxfev: int | None = None,
    xatol: float = 1e-4,
    fatol: float = 1e-6,
    output: Path | None = None,
) -> dict:
    reference_root = reference_root.expanduser().resolve()
    parameter_path = _resolve_parameter_set(parameter_set, seed_number, reference_root)
    model_name, module, model_fn = _load_seed(seed_number)
    bm_values = _read_bm_parameters(parameter_path)
    bm_params = _initial_parameters(module, bm_values, parameter_path)
    initial_params, perturbation_factors = _perturb_parameters(
        bm_params, strength, np.random.default_rng(random_seed)
    )
    protocols = _load_protocols(data_path)

    print(f"Seed: {seed_number} ({model_name})")
    print(f"Parameter set: {parameter_path}")
    print(f"Data: {data_path.expanduser().resolve()}")
    print(f"Perturbation: Normal(mean=1, std={strength:g}), random seed={random_seed}")
    for name in initial_params:
        print(
            f"  {name}: factor={perturbation_factors[name]:.8g}, "
            f"value={initial_params[name]:.12g}"
        )

    result, initial_loss, scales, evaluations, trajectory, total_time = _optimize(
        model_fn,
        initial_params,
        protocols,
        maxiter,
        maxfev,
        xatol,
        fatol,
    )
    names = tuple(initial_params)
    optimized_params = {
        name: float(value)
        for name, value in zip(names, scales * result.x, strict=True)
    }
    initial_rmse = (
        math.sqrt(initial_loss)
        if math.isfinite(initial_loss) and initial_loss >= 0.0
        else INVALID_LOSS
    )
    final_rmse = (
        math.sqrt(float(result.fun))
        if math.isfinite(result.fun) and result.fun >= 0.0
        else INVALID_LOSS
    )
    payload = {
        "seed": seed_number,
        "model": model_name,
        "parameter_set": str(parameter_path),
        "data_path": str(data_path.expanduser().resolve()),
        "method": "Nelder-Mead",
        "strength": strength,
        "random_seed": random_seed,
        "perturbation_factors": perturbation_factors,
        "initial_parameters": initial_params,
        "optimized_parameters": optimized_params,
        "initial_mse": initial_loss,
        "final_mse": float(result.fun),
        "initial_rmse": initial_rmse,
        "final_rmse": final_rmse,
        "time_taken_seconds": total_time,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective_calls": evaluations,
        "trajectory": trajectory,
        "iterations_trajectory": [step["iteration"] for step in trajectory],
        "rmse_trajectory": [step["rmse"] for step in trajectory],
        "time_trajectory": [step["time_seconds"] for step in trajectory],
    }

    print("Optimization complete")
    print(f"Initial discovery MSE:  {initial_loss:.12g} (RMSE: {initial_rmse:.12g})")
    print(f"Final discovery MSE:    {result.fun:.12g} (RMSE: {final_rmse:.12g})")
    print(f"Time taken:             {total_time:.2f} s")
    print(f"Status: {result.message} (iterations={result.nit}, evaluations={result.nfev})")
    print("Optimized parameters:")
    for name, value in optimized_params.items():
        print(f"  {name}: {value:.12g}")

    if output is None:
        output = HERE / f"{model_name}_{parameter_path.stem}_perturbation_{strength}.jsonl"

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".jsonl":
        output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return payload


def main() -> None:
    args = _parse_args()
    optimize_seed(
        seed_number=args.seed,
        parameter_set=args.parameter_set,
        strength=args.strength,
        random_seed=args.random_seed,
        reference_root=args.reference_root,
        data_path=args.data_path,
        maxiter=args.maxiter,
        maxfev=args.maxfev,
        xatol=args.xatol,
        fatol=args.fatol,
        output=args.output,
    )


if __name__ == "__main__":
    main()
