"""Reproduce SciPy scoring timings on the synaptic-transmission discovery split.

Run from the EDGAR checkout with
``JAX_PLATFORMS=cpu uv run python benchmarks/scipy_synaptic.py``.
Use the same script from an unmodified checkout for the before measurement.
The external dataset and parameter CSV used by the original comparison are required.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from edgar.evolution.population import Population
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.utils import translate_to_jax
from edgar.scoring.scoring import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--equilibration-ms", type=float, default=30_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1] / "projects/synaptic_transmission"
    spec = TaskSpec.from_config(Config.from_yaml(project / "config.yaml"))
    discover, _, _ = spec.load_data_fn(
        data_path=spec.io["data_path"],
        discovery_equilibration_ms=args.equilibration_ms,
        validation_equilibration_ms=args.equilibration_ms,
    )
    # Same model, CSV and deterministic 30% perturbation as the original benchmark.
    sys.path.insert(0, str(project))
    from data_loader.load_data import _read_bm_parameters

    param_path = Path(
        "/home/rajah/projects/synaptic_transmission/parameter_sets/Neher/NM_FSIN_4Pr_v3.csv"
    )
    program = copy.deepcopy(spec.seed_programs[0])
    program.code.model_jax = translate_to_jax(program.code.model)
    csv_params = _read_bm_parameters(param_path)
    reciprocal = {"b3", "cadecay"}
    names = tuple(program.default_params)
    rng = np.random.default_rng(11)
    factors = rng.normal(1.0, 0.3, len(names))
    if np.any(factors <= 0):
        raise ValueError("Non-positive perturbation")
    starts = {
        name: float(
            (1 / csv_params[name] if name in reciprocal else csv_params[name]) * factor
        )
        for name, factor in zip(names, factors, strict=True)
    }
    program.default_params = starts
    program.code.param_est = f"def parameter_estimator(data):\n    return {starts!r}\n"
    config = {
        "optimizer": "scipy",
        "timeout_s": 600.0,
        "jax_backend": "cpu",
        "param_penalty_weight": 0.0,
        "optimizer_config": {
            "method": "Nelder-Mead",
            "options": {"maxiter": args.maxiter},
        },
    }
    population = Population()
    population.add(program)
    t0 = time.perf_counter()
    score(
        population,
        discover,
        None,
        config,
        spec.loss_fn,
        split="discover",
        apply_model_fn=spec.apply_model_fn,
    )
    score_s = time.perf_counter() - t0
    score_loss = program.program_losses.discover.trajectories[0][-1]

    model = program.compile_model()
    initial = np.asarray(list(starts.values()))
    data = discover[0]

    def objective(values):
        params = {
            name: np.asarray([value]) for name, value in zip(names, values, strict=True)
        }
        prediction = spec.apply_model_fn(model, data, params)
        return float(np.asarray(spec.loss_fn(prediction, data)).item())

    t0 = time.perf_counter()
    manual = minimize(
        objective, initial, method="Nelder-Mead", options={"maxiter": args.maxiter}
    )
    manual_s = time.perf_counter() - t0
    result = {
        "score_s": score_s,
        "manual_s": manual_s,
        "score_final_loss": float(program.program_losses.discover.final),
        "score_last_trace_loss": float(score_loss),
        "manual_final_loss": float(manual.fun),
        "manual_nfev": manual.nfev,
        "maxiter": args.maxiter,
        "equilibration_ms": args.equilibration_ms,
    }
    if not np.isclose(result["score_last_trace_loss"], manual.fun, rtol=1e-4):
        raise AssertionError("EDGAR and manual SciPy optimizations diverged")
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
