"""Plot EDGAR optimization results for each Berkeley Madonna parameter set.

Each input JSONL contains one seed program initialized from a named Berkeley
Madonna (BM) parameter set at a particular perturbation strength. This script
creates one figure per parameter set comparing the four protocol rollouts.

Run from the EDGAR repository root::

    uv run python projects/synaptic_transmission/benchmark/optimization/plot_optimization_comparisons.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
EDGAR_ROOT = HERE.parents[3]
EDGAR_PROJECT = HERE.parents[1]
DEFAULT_REFERENCE_ROOT = Path(
    os.environ.get(
        "SYNAPTIC_TRANSMISSION_REFERENCE",
        "/home/rajah/projects/synaptic_transmission",
    )
)
DATASET_NAMES = tuple(f"dataset_{index:02d}" for index in range(1, 5))
RESULT_PATTERN = re.compile(
    r"^(?P<model>Neher|RP)_(?P<parameter_set>.+)_perturbation_"
    r"(?P<strength>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\.jsonl$"
)

sys.path.insert(0, str(EDGAR_PROJECT / "data_loader"))
sys.path.insert(0, str(EDGAR_ROOT))

from edgar.evolution.population import Population
from load_data import DEFAULT_DISCOVERY_EQUILIBRATION_MS, _make_split, apply_model


@dataclass(frozen=True)
class OptimizationRun:
    """One initial and optimized EDGAR rollout."""

    model: str
    parameter_set: str
    strength: float
    initial: dict[str, np.ndarray]
    optimized: dict[str, np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot protocol fits for the BM-initialized synaptic-transmission "
            "optimizations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=HERE,
        help="Directory containing the optimization JSONL files.",
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


def _batched_params(params: dict) -> dict[str, jax.Array]:
    """Convert one serialized parameter set to ``apply_model`` batch format."""
    batched = {}
    for name, value in params.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = array[None]
        if array.shape != (1,):
            raise ValueError(
                f"parameter {name!r} has shape {array.shape}, expected (1,)"
            )
        batched[name] = jnp.asarray(array)
    return batched


def _predict(model_fn, split: dict, params: dict, npz) -> dict[str, np.ndarray]:
    prediction = np.asarray(
        apply_model(model_fn, split, _batched_params(params))[0],
        dtype=np.float64,
    )
    return {
        name: prediction[index, : npz[f"{name}__time_ms"].size]
        for index, name in enumerate(DATASET_NAMES)
    }


def _result_metadata(path: Path) -> tuple[str, str, float] | None:
    match = RESULT_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return (
        match.group("model"),
        match.group("parameter_set"),
        float(match.group("strength")),
    )


def _load_runs(input_dir: Path, split: dict, npz) -> list[OptimizationRun]:
    runs = []
    for path in sorted(input_dir.glob("*.jsonl")):
        metadata = _result_metadata(path)
        if metadata is None:
            continue
        model, parameter_set, strength = metadata
        programs = list(Population.load(str(path)))
        if len(programs) != 1:
            raise ValueError(f"{path} contains {len(programs)} programs; expected one")

        program = programs[0]
        if program.params_init is None or program.params is None:
            raise ValueError(f"{path} does not contain initial and optimized parameters")
        model_fn = program.compile_model()
        runs.append(
            OptimizationRun(
                model=model,
                parameter_set=parameter_set,
                strength=strength,
                initial=_predict(model_fn, split, program.params_init, npz),
                optimized=_predict(model_fn, split, program.params, npz),
            )
        )

    if not runs:
        raise FileNotFoundError(
            f"No files matching {RESULT_PATTERN.pattern!r} found in {input_dir}"
        )
    return runs


def _bm_export_path(reference_root: Path, model: str, parameter_set: str) -> Path:
    prefix = "RP_" if model == "RP" else ""
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
        raise ValueError(f"{path}: expected four protocol groups, found {len(index_groups)}")

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


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _plot_protocol(
    axis,
    npz,
    name: str,
    bm_prediction: np.ndarray,
    runs: list[OptimizationRun],
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
        )
        axis.plot(
            time,
            run.optimized[name],
            color=color,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=2.5,
        )

    axis.set_title(f"{name.replace('dataset_', 'Dataset ')}\n{protocol}", fontsize=10)
    axis.set_xlabel("Protocol time (ms)")
    axis.grid(alpha=0.2)


def _add_legends(
    axis,
    runs: list[OptimizationRun],
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
            label="EDGAR initial",
        ),
        Line2D(
            [],
            [],
            color="#666666",
            linestyle="-",
            marker="o",
            markersize=3,
            linewidth=1.8,
            label="EDGAR optimized",
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


def _overall_rmse(
    predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> float:
    residuals = np.concatenate(
        [predictions[name] - targets[name] for name in DATASET_NAMES]
    )
    return float(np.sqrt(np.mean(residuals**2)))


def _plot_rmse_summary(
    axis,
    runs: list[OptimizationRun],
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
            [f"EDGAR initial, p={strength}", f"EDGAR optimized, p={strength}"]
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
        bbox=(0.19, 0.0, 0.62, 0.88),
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
    axis.set_title("RMSE summary (SV; overall pools all protocol observations)", fontsize=10)


def _plot_parameter_set(
    model: str,
    parameter_set: str,
    runs: list[OptimizationRun],
    bm_predictions: dict[str, np.ndarray],
    npz,
    output_path: Path,
    dpi: int,
) -> None:
    runs = sorted(runs, key=lambda run: run.strength)
    strengths = [run.strength for run in runs]
    if len(strengths) != len(set(strengths)):
        raise ValueError(f"duplicate perturbation strengths for {model} {parameter_set}")
    color_values = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(runs)))
    colors = dict(zip(strengths, color_values, strict=True))

    fig = plt.figure(figsize=(19, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(0.16, 1.0, 0.52))
    legend_axis = fig.add_subplot(grid[0, :])
    protocol_axes = [fig.add_subplot(grid[1, index]) for index in range(4)]
    rmse_axis = fig.add_subplot(grid[2, :])

    _add_legends(legend_axis, runs, colors)

    print(f"\n{model}\t{parameter_set}")
    print("dataset\tseries\tperturbation\trmse")
    for axis, name in zip(protocol_axes, DATASET_NAMES, strict=True):
        _plot_protocol(axis, npz, name, bm_predictions[name], runs, colors)
        release = np.asarray(npz[f"{name}__release_sv"], dtype=np.float64)
        print(f"{name}\tBM optimized reference\t-\t{_rmse(bm_predictions[name], release):.8g}")
        for run in runs:
            print(
                f"{name}\tEDGAR initial\t{run.strength:g}\t"
                f"{_rmse(run.initial[name], release):.8g}"
            )
            print(
                f"{name}\tEDGAR optimized\t{run.strength:g}\t"
                f"{_rmse(run.optimized[name], release):.8g}"
            )

    protocol_axes[0].set_ylabel("Release (SV)")
    _plot_rmse_summary(rmse_axis, runs, bm_predictions, npz)
    fig.suptitle(
        f"{model} — {parameter_set}: protocol fits",
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
        split = _make_split(npz, DATASET_NAMES, DEFAULT_DISCOVERY_EQUILIBRATION_MS)
        runs = _load_runs(input_dir, split, npz)
        groups: dict[tuple[str, str], list[OptimizationRun]] = {}
        for run in runs:
            groups.setdefault((run.model, run.parameter_set), []).append(run)

        for (model, parameter_set), group in sorted(groups.items()):
            bm_path = _bm_export_path(reference_root, model, parameter_set)
            if not bm_path.is_file():
                raise FileNotFoundError(bm_path)
            bm_predictions = _load_bm_reference(bm_path, npz)
            output_path = output_dir / f"{model}_{parameter_set}_optimization.png"
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
