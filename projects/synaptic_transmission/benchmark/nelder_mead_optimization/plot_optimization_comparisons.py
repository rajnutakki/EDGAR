"""Plot Nelder-Mead optimization results for each Berkeley Madonna parameter set.

Each input JSONL contains one seed program initialized from a named Berkeley
Madonna (BM) parameter set at a particular perturbation strength, optimized with
SciPy's Nelder-Mead simplex algorithm. This script creates one figure per
parameter set comparing:
1. Model rollouts against experimental data and the Berkeley Madonna ground truth
   across the four discovery protocols.
2. An RMSE summary table (per-protocol and pooled overall RMSE).
3. The optimization trajectory over iterations, displaying the total optimization time.

Run from the EDGAR repository root::

    uv run python projects/synaptic_transmission/benchmark/nelder_mead_optimization/plot_optimization_comparisons.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
REPO_ROOT = PROJECT_DIR.parent.parent
DEFAULT_REFERENCE_ROOT = Path(
    os.environ.get(
        "SYNAPTIC_TRANSMISSION_REFERENCE",
        "/home/rajah/projects/synaptic_transmission",
    )
)
DEFAULT_DATA_PATH = DEFAULT_REFERENCE_ROOT / "dataset" / "data.npz"
SEED_PROGRAMS_DIR = PROJECT_DIR / "seed_programs"

SEED_MODELS = {
    1: ("Neher", SEED_PROGRAMS_DIR / "model1.py"),
    2: ("RP", SEED_PROGRAMS_DIR / "model2.py"),
}
DATASET_NAMES = tuple(f"dataset_{index:02d}" for index in range(1, 5))
DISCOVERY_PROTOCOL_ORDER = (
    "dataset_04",
    "dataset_03",
    "dataset_02",
    "dataset_01",
)
DT = 0.1
DISCOVERY_EQUILIBRATION_MS = 30_000.0
DISCOVERY_PROTOCOL_TAIL_MS = 10.0

RESULT_PATTERN = re.compile(
    r"^(?P<model>Neher|RP)_(?P<parameter_set>.+)_perturbation_"
    r"(?P<strength>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\.jsonl$"
)


@dataclass(frozen=True)
class Protocol:
    event_indices: np.ndarray
    release: np.ndarray


@dataclass(frozen=True)
class NelderMeadRun:
    """One initial and optimized Nelder-Mead run with trajectory."""

    model: str
    parameter_set: str
    strength: float
    initial: dict[str, np.ndarray]
    optimized: dict[str, np.ndarray]
    initial_rmse: float
    final_rmse: float
    time_taken_seconds: float
    iterations: int
    function_evaluations: int
    success: bool
    status: int
    message: str
    trajectory: list[dict[str, float]]
    iterations_trajectory: list[int]
    rmse_trajectory: list[float]
    time_trajectory: list[float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot protocol fits and optimization trajectories for the "
            "Nelder-Mead synaptic-transmission optimizations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=HERE,
        help="Directory containing the Nelder-Mead optimization JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
        help="Directory in which to write one PNG per parameter set.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Root of the external synaptic_transmission reference project.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Dataset NPZ; defaults to <reference-root>/dataset/data.npz.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    return parser.parse_args()


def _load_seed(
    seed_number: int, seed_programs_dir: Path = SEED_PROGRAMS_DIR
) -> tuple[str, ModuleType, Callable]:
    model_name, filename = (
        ("Neher", seed_programs_dir / "model1.py")
        if seed_number == 1
        else ("RP", seed_programs_dir / "model2.py")
    )
    spec = importlib.util.spec_from_file_location(
        f"synaptic_seed_{seed_number}", filename
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load seed model from {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_fn = module.model
    if not callable(model_fn) or not callable(getattr(model_fn, "INITIAL_STATE", None)):
        raise ValueError(
            f"Seed model {filename} does not expose the expected model interface"
        )
    return model_name, module, model_fn


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
                raise ValueError(
                    f"{name}: time and release must be matching nonempty arrays"
                )
            grid = time / DT
            if not np.all(np.isfinite(time)) or not np.allclose(
                grid, np.rint(grid), rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f"{name}: times must be finite and lie on the {DT} ms grid"
                )
            if not np.isclose(time[0], 0.0) or np.any(np.diff(time) <= 0.0):
                raise ValueError(
                    f"{name}: times must start at zero and strictly increase"
                )
            if not np.all(np.isfinite(release)):
                raise ValueError(f"{name}: release observations must be finite")
            protocols.append(
                Protocol(np.asarray(np.rint(grid), dtype=np.int64), release)
            )
    return tuple(protocols)


def _rollout(
    model_fn: Callable, params: dict[str, float], protocols: tuple[Protocol, ...]
) -> np.ndarray:
    """Reproduce the continuous dense discovery split rollout."""
    state = model_fn.INITIAL_STATE(params)
    release_previous = 0.0
    predictions = []
    equilibration_steps = round(DISCOVERY_EQUILIBRATION_MS / DT)
    tail_steps = round(DISCOVERY_PROTOCOL_TAIL_MS / DT)

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

        zero_steps = tail_steps + equilibration_steps - 1

    return np.asarray(predictions, dtype=np.float64)


def _predict(
    model_fn: Callable,
    params: dict[str, float],
    protocols: tuple[Protocol, ...],
) -> dict[str, np.ndarray]:
    """Roll out model and map predictions into per-dataset arrays."""
    dense = _rollout(model_fn, params, protocols)
    predictions = {}
    offset = 0
    for name, protocol in zip(DISCOVERY_PROTOCOL_ORDER, protocols, strict=True):
        count = protocol.release.size
        predictions[name] = dense[offset : offset + count]
        offset += count
    return predictions


def _result_metadata(path: Path) -> tuple[str, str, float] | None:
    match = RESULT_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return (
        match.group("model"),
        match.group("parameter_set"),
        float(match.group("strength")),
    )


def _bm_export_path(reference_root: Path, model: str, parameter_set: str) -> Path:
    prefix = "RP_" if model == "RP" and not parameter_set.startswith("RP_") else ""
    return reference_root / "bm_exports" / f"{prefix}{parameter_set}.txt"


def _load_bm_reference(path: Path, npz) -> dict[str, np.ndarray]:
    """Read release values at pulse times and match them to the four protocols."""
    time, release, stimulus = np.loadtxt(
        path,
        delimiter="\t",
        skiprows=1,
        usecols=(0, 1, 7),
        unpack=True,
    )
    event_indices = np.flatnonzero(stimulus > 0.5)
    boundaries = np.flatnonzero(np.diff(time[event_indices]) > 10_000.0) + 1
    index_groups = np.split(event_indices, boundaries)
    if len(index_groups) != len(DATASET_NAMES):
        raise ValueError(
            f"{path}: expected four protocol groups, found {len(index_groups)}"
        )

    unmatched = set(DATASET_NAMES)
    predictions = {}
    for indices in index_groups:
        relative_time = time[indices] - time[indices[0]]
        candidates = [
            name
            for name in unmatched
            if relative_time.shape == npz[f"{name}__time_ms"].shape
            and np.allclose(
                relative_time,
                npz[f"{name}__time_ms"],
                rtol=0.0,
                atol=1e-4,
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{path}: could not uniquely match a BM protocol; candidates={candidates}"
            )
        name = candidates[0]
        unmatched.remove(name)
        predictions[name] = release[indices]
    return predictions


def _load_runs(
    input_dir: Path, protocols: tuple[Protocol, ...]
) -> list[NelderMeadRun]:
    runs = []
    cached_seeds: dict[int, tuple[str, ModuleType, Callable]] = {}

    for path in sorted(input_dir.glob("*.jsonl")):
        metadata = _result_metadata(path)
        if metadata is None:
            continue
        model_name, parameter_set, strength = metadata

        with path.open(encoding="utf-8") as stream:
            lines = [line.strip() for line in stream if line.strip()]
        if not lines:
            continue
        # Each JSONL file contains the Nelder-Mead result record.
        payload = json.loads(lines[0])

        seed_number = int(payload.get("seed", 1 if model_name == "Neher" else 2))
        if seed_number not in cached_seeds:
            cached_seeds[seed_number] = _load_seed(seed_number)
        _, _, model_fn = cached_seeds[seed_number]

        initial_params = payload["initial_parameters"]
        optimized_params = payload["optimized_parameters"]

        initial_preds = _predict(model_fn, initial_params, protocols)
        optimized_preds = _predict(model_fn, optimized_params, protocols)

        trajectory = payload.get("trajectory", [])
        iterations_traj = payload.get(
            "iterations_trajectory", [step["iteration"] for step in trajectory]
        )
        rmse_traj = payload.get(
            "rmse_trajectory", [step["rmse"] for step in trajectory]
        )
        time_traj = payload.get(
            "time_trajectory", [step["time_seconds"] for step in trajectory]
        )

        runs.append(
            NelderMeadRun(
                model=model_name,
                parameter_set=parameter_set,
                strength=strength,
                initial=initial_preds,
                optimized=optimized_preds,
                initial_rmse=float(payload.get("initial_rmse", math.nan)),
                final_rmse=float(payload.get("final_rmse", math.nan)),
                time_taken_seconds=float(payload.get("time_taken_seconds", 0.0)),
                iterations=int(payload.get("iterations", len(trajectory) - 1)),
                function_evaluations=int(payload.get("function_evaluations", 0)),
                success=bool(payload.get("success", False)),
                status=int(payload.get("status", 0)),
                message=str(payload.get("message", "")),
                trajectory=trajectory,
                iterations_trajectory=iterations_traj,
                rmse_trajectory=rmse_traj,
                time_trajectory=time_traj,
            )
        )

    if not runs:
        raise FileNotFoundError(
            f"No files matching {RESULT_PATTERN.pattern!r} found in {input_dir}"
        )
    return runs


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _overall_rmse(
    predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> float:
    residuals = np.concatenate(
        [predictions[name] - targets[name] for name in DATASET_NAMES]
    )
    return float(np.sqrt(np.mean(residuals**2)))


def _format_time(seconds: float) -> str:
    """Format duration into a clean string representation."""
    if not math.isfinite(seconds) or seconds < 0.0:
        return "N/A"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{seconds:.1f} s ({minutes:.1f} min)"
    hours = minutes / 60.0
    return f"{minutes:.1f} min ({hours:.1f} h)"


def _plot_protocol(
    axis,
    npz,
    name: str,
    bm_prediction: np.ndarray,
    runs: list[NelderMeadRun],
    colors: dict[float, tuple[float, ...]],
) -> None:
    time = np.asarray(npz[f"{name}__time_ms"], dtype=np.float64)
    release = np.asarray(npz[f"{name}__release_sv"], dtype=np.float64)
    protocol = str(npz[f"{name}__protocol"].item())

    axis.scatter(
        time,
        release,
        color="black",
        marker="x",
        s=30,
        linewidths=1.4,
        zorder=10,
    )
    axis.plot(
        time,
        bm_prediction,
        color="#CC79A7",
        linestyle="-",
        linewidth=3,
        zorder=3,
    )
    for run in runs:
        color = colors[run.strength]
        axis.plot(
            time,
            run.initial[name],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            zorder=4,
        )
        axis.plot(
            time,
            run.optimized[name],
            color=color,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            zorder=5,
        )

    axis.set_title(f"{name.replace('dataset_', 'Dataset ')}\n{protocol}", fontsize=10)
    axis.set_xlabel("Protocol time (ms)", fontsize=9)
    axis.grid(alpha=0.2)


def _add_legends(
    axis,
    runs: list[NelderMeadRun],
    colors: dict[float, tuple[float, ...]],
) -> None:
    """Add separate legends for series style and perturbation color."""
    axis.set_axis_off()
    series_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker="x",
            linestyle="none",
            markersize=6,
            label="Data",
        ),
        Line2D(
            [],
            [],
            color="#CC79A7",
            linestyle="-",
            linewidth=3,
            label="BM optimized reference",
        ),
        Line2D(
            [],
            [],
            color="#666666",
            linestyle="--",
            linewidth=1.4,
            label="Nelder-Mead initial",
        ),
        Line2D(
            [],
            [],
            color="#666666",
            linestyle="-",
            marker="o",
            markersize=3,
            linewidth=1.8,
            label="Nelder-Mead optimized",
        ),
    ]
    series_legend = axis.legend(
        handles=series_handles,
        title="Series",
        ncol=len(series_handles),
        loc="center",
        bbox_to_anchor=(0.31, 0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        handlelength=2.6,
        columnspacing=1.5,
    )
    axis.add_artist(series_legend)

    perturbation_handles = [
        Line2D(
            [],
            [],
            color=colors[run.strength],
            linewidth=2.5,
            label=f"{run.strength:g}",
        )
        for run in runs
    ]
    axis.legend(
        handles=perturbation_handles,
        title="Perturbation strength",
        ncol=len(perturbation_handles),
        loc="center",
        bbox_to_anchor=(0.79, 0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        handlelength=2.6,
        columnspacing=1.5,
    )


def _plot_rmse_summary(
    axis,
    runs: list[NelderMeadRun],
    bm_predictions: dict[str, np.ndarray],
    npz,
) -> None:
    """Show per-protocol and pooled RMSE values in a compact table."""
    axis.set_axis_off()
    targets = {
        name: np.asarray(npz[f"{name}__release_sv"], dtype=np.float64)
        for name in DATASET_NAMES
    }
    row_labels = ["BM optimized reference"]
    values = [
        [_rmse(bm_predictions[name], targets[name]) for name in DATASET_NAMES]
        + [_overall_rmse(bm_predictions, targets)]
    ]
    for run in runs:
        strength = f"{run.strength:g}"
        row_labels.extend(
            [
                f"Nelder-Mead initial, p={strength}",
                f"Nelder-Mead optimized, p={strength}",
            ]
        )
        values.extend(
            [
                [_rmse(run.initial[name], targets[name]) for name in DATASET_NAMES]
                + [_overall_rmse(run.initial, targets)],
                [_rmse(run.optimized[name], targets[name]) for name in DATASET_NAMES]
                + [_overall_rmse(run.optimized, targets)],
            ]
        )

    table = axis.table(
        cellText=[[f"{value:.3f}" for value in row] for row in values],
        rowLabels=row_labels,
        colLabels=[
            *(name.replace("dataset_", "Dataset ") for name in DATASET_NAMES),
            "Overall",
        ],
        cellLoc="center",
        rowLoc="right",
        loc="center",
        bbox=(0.0, 0.05, 1.0, 0.85),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f6f6")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.6)
    axis.set_title(
        "RMSE summary (SV; overall pools all protocol observations)", fontsize=10
    )


def _plot_trajectory(
    axis,
    runs: list[NelderMeadRun],
    bm_predictions: dict[str, np.ndarray],
    npz,
    colors: dict[float, tuple[float, ...]],
) -> None:
    """Plot optimization trajectory and display total optimization time."""
    targets = {
        name: np.asarray(npz[f"{name}__release_sv"], dtype=np.float64)
        for name in DATASET_NAMES
    }
    bm_overall_rmse = _overall_rmse(bm_predictions, targets)

    # Plot BM reference RMSE baseline
    axis.axhline(
        bm_overall_rmse,
        color="#CC79A7",
        linestyle="--",
        linewidth=1.8,
        label=f"BM reference ({bm_overall_rmse:.3f})",
        zorder=2,
    )

    info_lines = []
    for run in runs:
        color = colors[run.strength]
        iters = np.asarray(run.iterations_trajectory, dtype=np.int64)
        rmses = np.asarray(run.rmse_trajectory, dtype=np.float64)
        formatted_time = _format_time(run.time_taken_seconds)

        axis.plot(
            iters,
            rmses,
            color=color,
            linestyle="-",
            linewidth=1.8,
            label=f"p={run.strength:g} ({formatted_time})",
            zorder=4,
        )
        axis.scatter(
            iters[0],
            rmses[0],
            color=color,
            marker="o",
            s=35,
            zorder=5,
        )
        axis.scatter(
            iters[-1],
            rmses[-1],
            color=color,
            marker="*",
            s=90,
            edgecolors=color,
            zorder=6,
        )

        feval_str = (
            f", {run.function_evaluations} feval"
            if run.function_evaluations > 0
            else ""
        )
        info_lines.append(
            f"p={run.strength:g}: {run.time_taken_seconds:.1f} s "
            f"({run.time_taken_seconds / 60:.1f} min)\n"
            f"  {run.iterations} iter{feval_str}"
        )

    axis.set_title("Optimization trajectory & duration", fontsize=10)
    axis.set_xlabel("Nelder-Mead iteration", fontsize=9)
    axis.set_ylabel("Discovery RMSE (SV)", fontsize=9)
    axis.grid(alpha=0.2)
    axis.set_ylim(bottom=0.0)

    axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=8.5,
    )

    summary_box_text = "Total Optimization Time:\n" + "\n".join(info_lines)
    # Position callout in upper-right below the legend where the plot area is empty
    axis.text(
        0.97,
        0.66,
        summary_box_text,
        transform=axis.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.95,
            linewidth=0.8,
        ),
        zorder=7,
    )


def _plot_parameter_set(
    model: str,
    parameter_set: str,
    runs: list[NelderMeadRun],
    bm_predictions: dict[str, np.ndarray],
    npz,
    output_path: Path,
    dpi: int,
) -> None:
    runs = sorted(runs, key=lambda run: run.strength)
    strengths = [run.strength for run in runs]
    if len(strengths) != len(set(strengths)):
        raise ValueError(
            f"duplicate perturbation strengths for {model} {parameter_set}"
        )
    color_values = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(runs)))
    colors = dict(zip(strengths, color_values, strict=True))

    fig = plt.figure(figsize=(19, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(0.14, 1.0, 0.72))
    legend_axis = fig.add_subplot(grid[0, :])
    protocol_axes = [fig.add_subplot(grid[1, index]) for index in range(4)]
    rmse_axis = fig.add_subplot(grid[2, :2])
    trajectory_axis = fig.add_subplot(grid[2, 2:])

    _add_legends(legend_axis, runs, colors)

    print(f"\n{model}\t{parameter_set}")
    print("dataset\tseries\tperturbation\trmse")
    for axis, name in zip(protocol_axes, DATASET_NAMES, strict=True):
        _plot_protocol(axis, npz, name, bm_predictions[name], runs, colors)
        release = np.asarray(npz[f"{name}__release_sv"], dtype=np.float64)
        print(
            f"{name}\tBM optimized reference\t-\t"
            f"{_rmse(bm_predictions[name], release):.8g}"
        )
        for run in runs:
            print(
                f"{name}\tNelder-Mead initial\t{run.strength:g}\t"
                f"{_rmse(run.initial[name], release):.8g}"
            )
            print(
                f"{name}\tNelder-Mead optimized\t{run.strength:g}\t"
                f"{_rmse(run.optimized[name], release):.8g}"
            )

    protocol_axes[0].set_ylabel("Release (SV)", fontsize=9)
    _plot_rmse_summary(rmse_axis, runs, bm_predictions, npz)
    _plot_trajectory(trajectory_axis, runs, bm_predictions, npz, colors)

    for run in runs:
        print(
            f"Run p={run.strength:g}: total time={run.time_taken_seconds:.2f}s, "
            f"iterations={run.iterations}, feval={run.function_evaluations}, "
            f"initial RMSE={run.initial_rmse:.6g}, final RMSE={run.final_rmse:.6g}"
        )

    fig.suptitle(
        f"{model} — {parameter_set}: Nelder-Mead optimization",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> int:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    data_path = (
        args.data.expanduser().resolve()
        if args.data is not None
        else reference_root / "dataset" / "data.npz"
    )
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    with np.load(data_path, allow_pickle=False) as npz:
        protocols = _load_protocols(data_path)
        runs = _load_runs(input_dir, protocols)
        groups: dict[tuple[str, str], list[NelderMeadRun]] = {}
        for run in runs:
            groups.setdefault((run.model, run.parameter_set), []).append(run)

        for (model, parameter_set), group in sorted(groups.items()):
            bm_path = _bm_export_path(reference_root, model, parameter_set)
            if not bm_path.is_file():
                raise FileNotFoundError(bm_path)
            bm_predictions = _load_bm_reference(bm_path, npz)
            output_path = (
                output_dir / f"{model}_{parameter_set}_optimization.png"
            )
            _plot_parameter_set(
                model,
                parameter_set,
                group,
                bm_predictions,
                npz,
                output_path,
                args.dpi,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
