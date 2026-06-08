"""Orchestrates the entire EDGAR evolutionary experiment.

This module serves as the main entry point for running an EDGAR experiment. It initializes the environment,
sets up logging and status tracking, loads data, and then executes the core evolutionary loop.
The loop involves spawning new programs using LLMs, scoring them, and applying evolutionary
operations like deduplication, pruning, and migration across islands. It ensures robust
persistence of experiment state for live monitoring and post-hoc analysis.

JAX/XLA Runtime Guards:
Before any JAX-related imports, this module sets critical environment variables for JAX/XLA.
These guards are crucial for GPU memory management, specifically to mitigate out-of-memory (OOM)
errors during the subprocess-based scoring sweeps. They ensure that JAX does not preallocate
all GPU memory and uses the platform allocator, enhancing stability when multiple JAX processes
are spawned.
"""

# ruff: noqa: E402
from __future__ import annotations

# JAX/XLA runtime guards — must be set before any import that loads JAX.
# Reduces GPU OOM during the spawn-subprocess scoring sweeps.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

import asyncio
import argparse
import time
import traceback
import sys
from pathlib import Path

from .io.task_spec import TaskSpec
from .io.logging import open_log, log_generation, close_log, print_and_log
from .io.status import write_status
from .evolution.population import Population
from .evolution.island import (
    seed,
    spawn,
    deduplicate,
    prune,
    migrate,
    save_island_census,
)
from .llm.generate import (
    generate_models,
    generate_param_ests,
    translate_programs,
)
from .io.config import RetryConfig
from .scoring.scoring import rank, score
from .io.plotting import generate_program_fits


async def run(spec: TaskSpec, log_level: str = "compact") -> None:
    """Orchestrates and executes the entire EDGAR evolutionary experiment.

    This asynchronous function manages the full lifecycle of an EDGAR run, from initialization
    and data loading to the generational evolutionary loop, LLM interactions, scoring, and final
    validation. It ensures robust logging, status tracking, and persistence of results for
    real-time dashboard monitoring and post-hoc analysis.

    The core algorithm follows these steps:
    1.  **Initialization**: Sets up the run environment, creates the output directory, and
        saves the `TaskSpec` for reproducibility. Initializes structured logging and real-time
        status tracking.
    2.  **Data Loading**: Loads the scientific problem data into `X_discover`, `X_validate`,
        and `X_eval` splits using the `spec.load_data_fn`.
    3.  **Seed Phase**:
        *   Initializes the `Population` with user-provided seed programs.
        *   Distributes these programs across `n_islands` (defined in `spec.evolution`).
        *   Translates initial NumPy models to JAX-compatible versions using an LLM.
        *   Scores all seed programs on the `discover` data split, performing parameter
            estimation and optimization.
        *   Generates initial visualization plots (`generate_program_fits`) for these programs.
        *   Persists the initial state of the `Population` and `Island` census, and updates
            the run status to 'running'.
    4.  **Generational Loop**: Iterates for `spec.evolution["n_generations"]` generations.
        In each generation `gen`:
        *   The `mode` (explore/exploit), `temperature`, and specific LLM models are
            determined by `spec.schedule(gen)`. The temperature generally decays to
            bias towards exploitation over time.
        *   **Spawn**: New programs are created as empty shells on each island. Parents are
            selected from the current island population using either uniform or Boltzmann
            sampling, where Boltzmann sampling biases selection towards better-performing
            programs based on their relative, standard-normalized losses and a `temperature`
            parameter.
        *   **LLM Generation**: Asynchronously calls Large Language Models to:
            *   Generate new NumPy `model` code and descriptive names (`generate_models`).
            *   Generate `parameter_estimator` code (`generate_param_ests`).
            *   Translate NumPy `model` code into JAX-compatible `model_jax` code (`translate_programs`).
            *   Image-based feedback is integrated here, providing visual context to the LLMs.
        *   **Scoring**: Newly generated programs are scored on the `discover` data split.
            This involves dynamic loading of generated code, parameter estimation, gradient
            descent optimization (using JAX and `optax.adam`), and calculation of a scalar loss
            (including a complexity penalty) and per-sample losses. A low-dimensional
            `eval_fingerprint` is also generated for deduplication.
        *   **Plotting**: Program fit visualizations are generated.
        *   **Evolutionary Operations**:
            *   `deduplicate`: Identifies and removes functionally identical programs within
                and across islands based on parameter count, loss similarity, and cosine
                similarity of `eval_fingerprint`.
            *   `prune`: Reduces the size of each island to a fixed number of best programs,
                maintaining computational efficiency and diversity.
            *   `migrate`: Exchanges programs between islands based on a defined `topology`,
                using Boltzmann sampling with a warped temperature to enhance selection pressure
                in later generations.
        *   **Persistence**: The current generation's `Population` and `Island` census
            are saved using atomic writes to ensure data integrity, and the run status is updated
            for live dashboard monitoring.
    5.  **Final Validation**: After the generational loop completes:
        *   Programs are prepared for final scoring on the `validate` data split.
        *   Programs are scored (`score`) on the `validate` split, but without `eval_fingerprint`
            calculation.
        *   Programs are `rank`ed based on their final validation losses.
    6.  **Error Handling**: A `finally` block ensures that the `Population`, `census`, and
        `status.json` are always saved, even if an exception occurs during the run. Any exceptions
        are captured, logged with their traceback, and the run status is updated to 'failed'.

    Args:
        spec: A `TaskSpec` object containing all configuration, data, and callable functions
            required for the experiment.
        log_level: The verbosity level for logging messages to the console and `run.log`.
            Can be 'compact', 'code', or 'prompts'.

    Returns:
        None. The function's primary effects are side effects: creating output files, updating
        status, and logging.
    """
    os.makedirs(spec.output_dir, exist_ok=True)
    spec.save(spec.output_dir)
    log = open_log(spec.output_dir, log_level)

    n_gens = spec.evolution["n_generations"]
    started_at = time.time()
    write_status(
        spec.output_dir, state="starting", n_gens=n_gens, started_at=started_at
    )
    pop_path = os.path.join(spec.output_dir, "population.jsonl")
    census_path = os.path.join(spec.output_dir, "island_census.jsonl")

    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    retry_config = RetryConfig(**spec.llms.get("retry", {}))
    config = {**spec.flat_config, "retry_config": retry_config}

    population = Population()
    census = []

    try:
        islands = seed(population, spec.seed_programs, spec.evolution["n_islands"])
        await translate_programs(
            population,
            spec.prompt_schemas.jax_model,
            spec.llms["jax_model_translator_llm"],
            retry_config=retry_config,
            max_tokens=config.get("max_tokens"),
        )
        score(
            population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover"
        )
        generate_program_fits(spec, X_discover[1], population)
        population.save(
            pop_path
        )  # snapshot of seed phase so the dashboard has data before gen 0 finishes
        save_island_census(census, census_path)
        write_status(
            spec.output_dir,
            state="running",
            n_gens=n_gens,
            current_gen=-1,
            started_at=started_at,
        )

        for gen in range(spec.evolution["n_generations"]):
            print_and_log(log, f"Generation {gen} / {spec.evolution['n_generations']}")
            mode, temperature, llms = spec.schedule(gen)
            spawn(
                population,
                islands,
                gen,
                mode,
                temperature,
                batch_size=spec.evolution["batch_size"],
                num_parents=spec.llms["num_parents"],
                rng=spec.rng,
            )

            await generate_models(
                population,
                spec.prompt_schemas.model,
                llms.model[gen % len(llms.model)]
                if isinstance(llms.model, list)
                else llms.model,
                mode,
                temperature,
                config=config,
                spec=spec,
                data=X_discover[1],
            )  # use test data of X_discover for plotting
            await generate_param_ests(
                population,
                spec.prompt_schemas.param_est,
                llms.param_est,
                config,
            )
            await translate_programs(
                population,
                spec.prompt_schemas.jax_model,
                llms.model_jax,
                retry_config=retry_config,
            )

            score(
                population,
                X_discover,
                X_eval,
                spec.scoring,
                spec.loss_fn,
                split="discover",
            )
            generate_program_fits(spec, X_discover[1], population)

            deduplicate(islands, population, spec.evolution)
            prune(islands, population, spec.evolution)
            migrate(islands, population, spec.evolution, temperature, rng=spec.rng)
            census.append([set(island) for island in islands])
            log_generation(log, gen, population, islands, spec)

            # Per-generation persistence for the live dashboard. Atomic writes
            # protect a polling reader from observing torn files.
            population.save(pop_path)
            save_island_census(census, census_path)
            write_status(
                spec.output_dir,
                state="running",
                n_gens=n_gens,
                current_gen=gen,
                started_at=started_at,
            )

        population.prepare_validation_scoring(islands)
        score(
            population, X_validate, None, spec.scoring, spec.loss_fn, split="validate"
        )
        rank(population)

        print_and_log(
            log, f"***** Run complete. Output directory: {spec.output_dir} *****"
        )

    finally:  # runs whether or not an exception is raised, ensuring that results are saved
        exc_info = sys.exc_info()
        failed = exc_info[0] is not None
        if failed:
            print_and_log(
                log,
                f"***** Run failed with exception:\n{''.join(traceback.format_exception(*exc_info))}***** Output directory: {spec.output_dir} *****",
            )
        population.save(pop_path)
        save_island_census(census, census_path)
        write_status(
            spec.output_dir,
            state="failed" if failed else "complete",
            n_gens=n_gens,
            current_gen=(len(census) - 1) if census else None,
            started_at=started_at,
            error=(f"{exc_info[0].__name__}: {exc_info[1]}" if failed else None),
        )
        close_log(log)

    return


if __name__ == "__main__":
    """Main execution block for running EDGAR from the command line.

    This block parses command-line arguments to obtain the configuration file path.
    It then loads the `Config` (either from a `config.yaml` or a previously saved
    `task_spec.yaml`), initializes a `TaskSpec` object from this configuration,
    and finally runs the asynchronous EDGAR experiment using `asyncio.run()`.
    """
    parser = argparse.ArgumentParser(description="Run EDGAR")
    parser.add_argument(
        "config", type=str, help="Path to task config.yaml or task_spec.yaml"
    )
    args = parser.parse_args()

    from .io.config import Config

    path = Path(args.config)
    if path.name == "task_spec.yaml":
        config = Config.from_taskspec(path)
    else:
        config = Config.from_yaml(path)
    spec = TaskSpec.from_config(config)
    asyncio.run(run(spec))
