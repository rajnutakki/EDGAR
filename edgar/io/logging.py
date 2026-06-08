"""
src/io/logging.py

Human-readable run logging for EDGAR experiments.

Creates a single run.log file and appends one summary block per generation.
Verbosity is controlled by the level argument:

  compact  — one summary block per generation (success rates, island state, global best)
  code     — compact + generated code for each program born this generation
  prompts  — code + reconstructed LLM prompts and image paths

Prompts are reconstructed post-hoc from program birth metadata and the spec's
prompt schemas.

Warnings emitted via warnings.warn() during a generation are buffered and
appended to the end of that generation's block in the log.

Example usage:
    log = open_log(spec.output_dir, level="compact")

    for gen in range(spec.evolution["n_generations"]):
        mode, temperature, llms = spec.schedule(i)
        # ... run generation ...
        log_generation(log, gen, population, islands, spec)

    close_log(log)
"""

from __future__ import annotations

import os
import time
import warnings
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO, TYPE_CHECKING
from ..evolution.program import NotValidated

if TYPE_CHECKING:
    from ..evolution.population import Population
    from ..io.task_spec import TaskSpec


LEVELS = ("compact", "code", "prompts")


def print_and_log(log: RunLog, message: str) -> None:
    """Prints a message to the console and appends it to the run log file.

    Ensures that important messages are visible in real-time and
    persisted in the run's log file for later review.

    Args:
        log: The `RunLog` object managing the log file.
        message: The string message to print and log.
    """
    print(message)
    log.file.write(message + "\n")
    log.file.flush()


@dataclass
class RunLog:
    """A dataclass to hold the state and file handle for the EDGAR run log.

    Attributes:
        file: A file-like object (TextIO) opened for writing the log.
        level: The verbosity level of the log ("compact", "code", or "prompts").
        start_time: The monotonic time when the log was opened, used for
            calculating total elapsed time.
        previous_gen_time: The monotonic time at the end of the previous
            generation, used to calculate generation-specific elapsed time.
        warnings_buffer: A list of buffered warning messages to be flushed
            at the end of each generation.
        prev_showwarning: Stores the original `warnings.showwarning` hook
            to restore it when the log is closed.
    """

    file: TextIO
    level: str
    start_time: float
    previous_gen_time: float = 0.0
    warnings_buffer: list[str] = field(default_factory=list)
    prev_showwarning: Any = None


def open_log(output_dir: str, level: str = "compact") -> RunLog:
    """Creates `run.log` in the specified output directory and returns a RunLog handle.

    This function initializes the logging system for an EDGAR run. It also
    installs a custom `warnings.showwarning` hook that buffers any warnings
    emitted during a generation. These buffered warnings are then appended
    to the end of the current generation's log block, providing contextualized
    warning messages. The original warning hook is stored so it can be
    restored by `close_log()`.

    Args:
        output_dir: The run output directory (e.g., `spec.output_dir`) where `run.log` will be created.
        level: The verbosity level for the log, must be one of "compact", "code", or "prompts".

    Returns:
        A `RunLog` object containing the log file handle and state.

    Raises:
        ValueError: If `level` is not one of the allowed verbosity levels.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    os.makedirs(output_dir, exist_ok=True)
    f = open(os.path.join(output_dir, "run.log"), "w")
    f.write(f"EDGAR run log  |  level={level}\n{'=' * 60}\n\n")
    f.flush()
    log = RunLog(file=f, level=level, start_time=time.monotonic())

    original: Callable = warnings.showwarning

    def _hook(message, category, filename, lineno, file=None, line=None):
        original(message, category, filename, lineno, file, line)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        log.warnings_buffer.append(f"  [{ts}] {category.__name__}: {message}\n")

    log.prev_showwarning = original
    warnings.showwarning = _hook
    return log


def close_log(log: RunLog) -> None:
    """Flushes any remaining buffered warnings, closes the log file, and restores the original warnings hook.

    This function should be called at the end of an EDGAR run to ensure
    all log messages and warnings are persisted and system state is cleaned up.

    Args:
        log: The `RunLog` object to close.
    """
    _flush_warnings(log)
    log.file.close()
    if log.prev_showwarning is not None:
        warnings.showwarning = log.prev_showwarning


def _flush_warnings(log: RunLog) -> None:
    """Writes all buffered warning messages to the log file and clears the buffer.

    Args:
        log: The `RunLog` object containing the warnings buffer.
    """
    if log.warnings_buffer:
        log.file.write("  --- Warnings ---\n")
        log.file.writelines(log.warnings_buffer)
        log.warnings_buffer.clear()
        log.file.flush()


def log_generation(
    log: RunLog,
    gen: int,
    population: Population,
    islands: list[set[int]],
    spec: TaskSpec,
) -> None:
    """Appends a summary block for one generation to the run log file.

    This function compiles and writes a comprehensive summary of the current
    evolutionary generation to the `run.log` file. All statistics and details
    are dynamically derived from the `population`, `islands`, and `spec`
    objects, eliminating the need for intermediate state capture.

    The verbosity of the logged information depends on `log.level`:
    *   **"compact"**: Logs generation index, mode, temperature, elapsed times,
        LLM names, program spawning success rates, and the global best discover
        loss, along with the best program on each island.
    *   **"code"**: Includes all "compact" information, plus the generated
        `model`, `parameter_estimator`, and JAX `model_jax` code for all
        programs born in this generation.
    *   **"prompts"**: Includes all "code" information, plus the reconstructed
        LLM prompts (for model, parameter estimator, and JAX translation)
        used to generate each new program, along with paths to any associated
        feedback images.

    Args:
        log: The `RunLog` handle obtained from `open_log()`.
        gen: The current generation index (0-based).
        population: The current `Population` object containing all evolved programs.
        islands: A list of sets, where each set contains the indices of programs
            currently residing on a specific island (after pruning and deduplication).
        spec: The `TaskSpec` object containing global configuration and callables for the run.
    """
    f = log.file
    mode, temperature, llms = spec.schedule(gen)
    llm = (
        llms.model[gen % len(llms.model)]
        if isinstance(llms.model, list)
        else llms.model
    )
    born = [
        population[i]
        for i in range(len(population))
        if population[i].birth.generation == gen
    ]
    n = len(born)

    def pct(k):
        return f"{100 * k / n:.0f}%" if n else "n/a"

    n_model = sum(1 for p in born if p.code.model is not None)
    n_param_est = sum(1 for p in born if p.code.param_est is not None)
    n_jax = sum(1 for p in born if p.code.model_jax is not None)
    n_scored = sum(
        1 for p in born if p.program_losses.discover.final not in (None, float("inf"))
    )

    all_final = [
        population[i].program_losses.discover.final for i in range(len(population))
    ]
    valid = [l for l in all_final if l is not None and not isinstance(l, NotValidated)]
    global_best = f"{min(valid):.6f}" if valid else "n/a"
    elapsed = time.monotonic() - log.start_time
    this_gen_time = elapsed - log.previous_gen_time
    log.previous_gen_time = elapsed

    f.write(f"{'=' * 60}\n")
    f.write(
        f"Gen {gen:3d}  |  {mode}  |  temp={temperature:.3f}  | gen_time={this_gen_time:.1f}s | total time elapsed={elapsed:.1f}s\n"
    )
    f.write(
        f"LLMs     model={llm}  param_est={llms.param_est}  model_jax={llms.model_jax}\n"
    )
    f.write(
        f"Spawned  {n}  |  model={pct(n_model)}  param_est={pct(n_param_est)}  jax={pct(n_jax)}  scored={pct(n_scored)}\n"
    )
    f.write(f"Global best discover loss: {global_best}\n\n")
    f.write("Best programs on each island:\n")
    for idx, island in enumerate(islands):
        progs = [population[i] for i in island]
        best = min(progs, key=lambda p: p.program_losses.discover.final or float("inf"))
        f.write(
            f"  Island {idx}  size={len(island)}  best=#{best.idx} {best.name!r}  loss={best.program_losses.discover.final:.6f}\n"
        )
    f.write("\n")

    if log.level not in ("code", "prompts"):
        _flush_warnings(log)
        f.flush()
        return

    f.write("Newly-generated programs:\n")
    for p in born:
        f.write(f"  --- Program #{p.idx} (island={p.birth.island}) ---\n")
        f.write(f"  [model]\n{p.code.model or '(none)'}\n")
        f.write(f"  [param_est]\n{p.code.param_est or '(none)'}\n")
        f.write(f"  [model_jax]\n{p.code.model_jax or '(none)'}\n\n")

    if log.level != "prompts":
        _flush_warnings(log)
        f.flush()
        return

    for p in born:
        parents = [population[i] for i in p.birth.parent_indices]
        mode_p = p.birth.mode or "explore"
        f.write(f"  --- Prompts for Program #{p.idx} ---\n")
        f.write(
            f"  [model prompt]\n{spec.model_prompt_schema.build_prompt(mode_p, parents, spec.flat_config)}\n\n"
        )
        f.write(
            f"  [param_est prompt]\n{spec.param_est_prompt_schema.build_prompt('explore', parents, spec.flat_config, current_program=p)}\n\n"
        )
        f.write(
            f"  [jax model prompt]\n{spec.jax_model_prompt_schema.build_prompt('explore', current_program=p, config=spec.flat_config)}\n\n"
        )
        if p.image_path:
            f.write(f"  [image] {p.image_path}\n")
        f.write("\n")

    _flush_warnings(log)
    f.flush()
