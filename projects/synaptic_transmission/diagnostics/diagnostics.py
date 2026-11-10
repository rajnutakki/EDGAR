"""Held-out release fits, residuals, and parent-state equilibration plots.

Columns follow the discovery test split's leave-one-out fold order. Each model
gets an equilibration row (state colours are shared across models), followed by
release and absolute-residual rows with a shared time axis in each column.
"""

from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from edgar.projects.diagnostics import BaseDiagnostics

class Diagnostics(BaseDiagnostics):
    def plot_model_fits(
        self,
        data: dict[str, np.ndarray],
        programs: list[Any],
        rng: np.random.Generator,
        save_path: str = "",
        losses: list[float] | None = None,
        sample_losses: list[np.ndarray] | None = None,
        program_names: list[str] | None = None,
        params: list[dict] | None = None,
        title_prefix: str | None = None,
        apply_model_fn: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot every held-out protocol using its fold's fitted parameters.

        ``data`` must be a batched test split with one protocol per fold.
        ``params`` and ``program_names`` override program attributes, including
        for EDGAR's initial-versus-final fit comparison. State traces include the
        initial state at -equilibration_time and end immediately before the first
        stimulus; release predictions and residuals use observation indices only.
        ``apply_model_fn`` is the project's ``(model_fn, data, params)`` rollout
        hook supplied by EDGAR; the diagnostics module does not import project code.
        ``rng``, ``losses`` and ``sample_losses`` are accepted for the standard
        diagnostics interface; all held-out datasets are shown deterministically.
        """
        if save_path == "":
            raise ValueError("plot_model_fits requires a non-empty save_path")
        if not programs:
            return

        target = np.asarray(data["release"])
        if target.ndim != 3 or target.shape[1] != 1:
            raise ValueError("Expected a test split with one held-out protocol per fold")
        n_folds = target.shape[0]
        names = program_names if program_names is not None else [p.name for p in programs]
        parameter_sets = params if params is not None else [p.params for p in programs]
        if len(names) != len(programs) or len(parameter_sets) != len(programs):
            raise ValueError("params and program_names must match the number of programs")

        n_models = len(programs)
        model_colours = plt.get_cmap("tab10")
        state_colours: dict[str, Any] = {}
        fig, axes = plt.subplots(
            n_models + 2, n_folds, squeeze=False,
            figsize=(4.5 * n_folds, 2.7 * n_models + 5),
            gridspec_kw={"height_ratios": [1.2] * n_models + [1.4, 0.7]},
            layout="constrained",
        )
        try:
            for column in range(n_folds):
                axes[0, column].set_title(f"Dataset {column + 1} · held out")
                axes[-1, column].sharex(axes[-2, column])

            if apply_model_fn is None:
                raise ValueError("Synaptic diagnostics require EDGAR's apply_model_fn")
            for model_index, (program, fitted, name) in enumerate(
                zip(programs, parameter_sets, names)
            ):
                if fitted is None:
                    raise ValueError(f"No fitted parameters available for {name}")
                model_fn = program.compile_model()
                output = apply_model_fn(model_fn, data, fitted)
                release = np.asarray(output["release"])
                output_time = np.asarray(output["time"])
                dt = float(output_time[0, 0, 1] - output_time[0, 0, 0])
                state_keys = [key for key in output if key not in {"release", "time"}]
                for key in state_keys:
                    if key not in state_colours:
                        state_colours[key] = model_colours(len(state_colours) % 10)
                colour = model_colours(model_index % 10)

                for column in range(n_folds):
                    state_ax = axes[model_index, column]
                    equilibration = float(np.asarray(data["_equilibration_time"])[column, 0])
                    n_equilibration = int(round(equilibration / dt))
                    # Downsample display only, retaining both ends; simulation
                    # still uses every step of the scoring rollout.
                    indices = np.unique(np.linspace(
                        0, n_equilibration, min(n_equilibration + 1, 1500), dtype=int
                    ))
                    state_time = indices * dt - equilibration
                    fold_params = {key: value[column] for key, value in fitted.items()}
                    initial = model_fn.INITIAL_STATE(fold_params)
                    for key in state_keys:
                        # Rollout index zero is the state AFTER the first step.
                        trace = np.asarray(output[key])[column, 0]
                        values = np.concatenate((
                            np.asarray(initial[key]).reshape(1), trace[:n_equilibration]
                        ))
                        state_ax.plot(state_time, values[indices],
                                      color=state_colours[key], label=key, linewidth=1.3)
                    state_ax.set_xlabel("Time before first pulse (ms)")
                    state_ax.set_ylabel(f"{name:.20}\nState")
                    state_ax.legend(fontsize="small", ncol=2)

                    time = np.asarray(data["time"])[column, 0]
                    observation_index = np.asarray(data["_observation_index"])[column, 0]
                    observed = target[column, 0]
                    valid = (observation_index >= 0) & np.isfinite(time) & np.isfinite(observed)
                    time, observed = time[valid], observed[valid]
                    predicted = release[column, 0, observation_index[valid]]
                    release_ax, residual_ax = axes[-2, column], axes[-1, column]
                    if model_index == 0:
                        release_ax.plot(time, observed, "o-", color="black",
                                        markersize=3, linewidth=1, label="Data", zorder=5)
                    release_ax.plot(time, predicted, color=colour, label=f"{name:.20}", linewidth=1.5)
                    residual_ax.plot(time, np.abs(predicted - observed),
                                     color=colour, linewidth=1.3)
                    release_ax.set_ylabel("Release")
                    residual_ax.set_ylabel("|Release Residual|")
                    residual_ax.set_xlabel("Time from first pulse (ms)")
                    residual_ax.set_ylim(bottom=0)
                    release_ax.tick_params(labelbottom=False)

            for ax in axes.flat:
                ax.grid(alpha=0.25)
            axes[-2, 0].legend(fontsize="small")
            fig.suptitle(title_prefix or "Discovery held-out fits and equilibration")
            fig.savefig(save_path, bbox_inches="tight", dpi=120)
        finally:
            plt.close(fig)

def plot_model_fits(*args, **kwargs):
    """Module-level compatibility wrapper for Diagnostics.plot_model_fits."""
    return Diagnostics().plot_model_fits(*args, **kwargs)
