import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp


def plot_model_fits(
    data,
    programs,
    rng: np.random.Generator,
    save_path="",
    losses=None,
    sample_losses=None,
    program_names=None,
    params=None,
    title_prefix: str | None = None,
):
    """
    Plot observed target responses and overlaid model predictions for 9 random
    stimulus angles (population panel, 3x3), followed by single-cell tuning curves (3x3 panel)
    and trial-by-trial response tracking (3x3 panel) for 9 example cells.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Expected keys:
        - 'stimulus': array of shape (n_samples, n_trials)
        - 'response': array of shape (n_samples, n_trials, n_cells)
        - 'signal': array of shape (n_samples, n_trials, n_cells) (tuning curves)
    programs : list
        List of Program objects.
    save_path : str
        Output path for the plot.
    losses : list[float], optional
        List of scalar losses for each program.
    sample_losses : list[np.ndarray], optional
        List of per-sample losses for each program.
    program_names : list[str], optional
        List of names for each program.
    params : list[dict], optional
        List of parameter dicts for each program.
    title_prefix : str, optional
        Prefix for the figure title.
    """
    ### Plotting options ####
    model_colors = [
        "tab:orange",
        "tab:blue",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:olive",
    ]
    model_alphas = [0.8, 0.5, 0.3]
    data_color = "black"

    if save_path == "":
        raise ValueError("plot_model_fits requires a non-empty save_path")

    # 1. Resolve arguments
    if program_names is None:
        program_names = [p.name for p in programs]
    if losses is None:
        losses = [
            p.program_losses.discover.final if hasattr(p, "program_losses") else None
            for p in programs
        ]
    if params is None:
        params = [p.params for p in programs]

    # 2. Handle the sample dimension (take the first sample for plotting)
    sample_idx = 0
    stims = np.asarray(data["stimulus"][sample_idx]).reshape(-1)
    actual_response = np.asarray(data["response"][sample_idx])
    sig = np.asarray(data["signal"][sample_idx])  # (n_trials, n_cells)

    # 3. Calculate Preferred Angles using Vector Concentration
    # Use nansum because data might be masked (NaNs)
    complex_sum = np.nansum(sig * np.exp(2j * stims)[:, np.newaxis], axis=0)
    pref_angles = (np.angle(complex_sum) / 2.0) % np.pi

    # Filter out cells that are entirely NaN in the response (masked out)
    valid_cells = np.where(~np.all(np.isnan(actual_response), axis=0))[0]
    if len(valid_cells) == 0:
        valid_cells = np.arange(actual_response.shape[1])

    # Sort only the valid cells
    pref_angles_valid = pref_angles[valid_cells]
    sort_idx = np.argsort(pref_angles_valid)

    final_cell_idx = valid_cells[sort_idx]
    sorted_pref_angles = pref_angles[final_cell_idx]
    sorted_actual = actual_response[:, final_cell_idx]

    # 4. Pick 3 random trials and cells that actually have data
    valid_trials = np.where(~np.all(np.isnan(actual_response), axis=1))[0]
    if len(valid_trials) == 0:
        valid_trials = np.arange(len(stims))

    n_show = min(3, len(valid_trials))
    random_trials = rng.choice(valid_trials, size=n_show, replace=False)

    # 5. Compute predictions (both sorted and raw cell indexing)
    predictions_sorted = []
    predictions_raw = []

    for i, program in enumerate(programs):
        model_fn = (
            program.compile_model()
            if hasattr(program, "compile_model")
            else program["model"]
        )
        p_dict = params[i]

        # Slice params for the first sample
        def _slice_leaf(x):
            if (
                isinstance(x, (np.ndarray, jnp.ndarray))
                and x.ndim > 0
                and x.shape[0] > sample_idx
            ):
                return x[sample_idx]
            return x

        plot_params = jax.tree_util.tree_map(_slice_leaf, p_dict)

        # Model expects data dict where arrays don't have the sample dim
        single_sample_data = {
            k: v[sample_idx] if hasattr(v, "__getitem__") and len(v) > sample_idx else v
            for k, v in data.items()
        }

        y_pred = np.asarray(model_fn(single_sample_data, plot_params))
        predictions_raw.append(y_pred)

        y_pred_sorted = y_pred[
            :, final_cell_idx
        ]  # Use the same filtered/sorted indices
        predictions_sorted.append(y_pred_sorted)

    # Pick 3 cells with valid responses for individual tuning curves (Row 2)
    chosen_cells = rng.choice(valid_cells, size=min(3, len(valid_cells)), replace=False)

    # 6. Plotting (2x3 grid: Row 1 = Population Fits, Row 2 = Single-cell Tuning Curves)
    fig = plt.figure(figsize=(18, 16))
    outer_gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # --- ROW 1: POPULATION FITS (3 trials) ---
    for i, trial_idx in enumerate(random_trials):
        ax_slot = outer_gs[0, i]
        inner_gs = ax_slot.subgridspec(2, 1, height_ratios=[4, 1.2], hspace=0.08)

        ax1 = fig.add_subplot(inner_gs[0])
        ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)

        angle = stims[trial_idx]

        # Plot each model's prediction
        for j, y_pred_sorted in enumerate(predictions_sorted):
            label = program_names[j] if j < len(program_names) else f"Model {j + 1}"
            ax1.plot(
                sorted_pref_angles,
                y_pred_sorted[trial_idx],
                color=model_colors[j % len(model_colors)],
                linewidth=1.5,
                alpha=model_alphas[j % len(model_alphas)],
                label=label,
                zorder=5,
            )

            # Compute point-by-point squared error
            sq_err = (y_pred_sorted[trial_idx] - sorted_actual[trial_idx]) ** 2
            ax2.plot(
                sorted_pref_angles,
                sq_err,
                color=model_colors[j % len(model_colors)],
                linewidth=1.5,
                alpha=model_alphas[j % len(model_alphas)],
            )

        # Plot actual/observed responses as scatter
        ax1.scatter(
            sorted_pref_angles,
            sorted_actual[trial_idx],
            color=data_color,
            s=15,
            alpha=0.3,
            label="Observed",
            zorder=10,
        )

        # Stimulus angle vertical line
        ax1.axvline(
            angle % np.pi,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="Stimulus Angle",
            zorder=1,
        )
        ax2.axvline(angle % np.pi, color="red", linestyle="--", alpha=0.6, zorder=1)

        ax1.set_title(
            f"Population Response: Trial {trial_idx}\n(Stimulus: {angle:.2f} rad)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_ylabel("Response", fontsize=10)
        ax1.tick_params(labelbottom=False, labelsize=9)

        ax2.set_xlabel("Pref. Orientation (rad)", fontsize=10)
        ax2.set_ylabel("Squared Error", fontsize=9)
        ax2.tick_params(labelsize=9)
        if i == 0:
            ax1.legend(fontsize=8, loc="upper right")

    # --- ROW 2: SINGLE-CELL TUNING CURVES (3 cells) ---
    sort_idx = np.argsort(stims)
    for i, cell in enumerate(chosen_cells):
        ax_slot = outer_gs[1, i]
        inner_gs = ax_slot.subgridspec(2, 1, height_ratios=[4, 1.2], hspace=0.08)

        ax1 = fig.add_subplot(inner_gs[0])
        ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)

        # Plot each model's prediction
        for j, y_pred in enumerate(predictions_raw):
            label = program_names[j] if j < len(program_names) else f"Model {j + 1}"
            ax1.plot(
                stims[sort_idx],
                y_pred[sort_idx, cell],
                color=model_colors[j % len(model_colors)],
                linewidth=1.5,
                alpha=model_alphas[j % len(model_alphas)],
                label=label,
                zorder=5,
            )

            # Compute point-by-point squared error (MSE) relative to stimulus angle
            sq_err = (y_pred[:, cell] - actual_response[:, cell]) ** 2
            ax2.plot(
                stims[sort_idx],
                sq_err[sort_idx],
                color=model_colors[j % len(model_colors)],
                linewidth=1.5,
                alpha=model_alphas[j % len(model_alphas)],
            )

        # Plot actual/observed responses as scatter
        ax1.scatter(
            stims,
            actual_response[:, cell],
            color=data_color,
            s=20,
            alpha=0.3,
            label="Observed",
            zorder=10,
        )

        # Set titles and labels
        ax1.set_title(
            f"Tuning Curve: Cell {cell}\n(Pref. Orientation: {pref_angles[cell]:.2f} rad)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_ylabel("Response", fontsize=10)
        ax1.tick_params(labelbottom=False, labelsize=9)

        ax2.set_xlabel("Stimulus Angle (rad)", fontsize=10)
        ax2.set_ylabel("Squared Error", fontsize=9)
        ax2.tick_params(labelsize=9)
        if i == 0:
            ax1.legend(fontsize=8, loc="upper right")

    # Construct Title
    summary_parts = []
    for i in range(len(programs)):
        name = program_names[i]
        loss = losses[i]
        if loss is not None:
            summary_parts.append(f"{name} loss: {loss:.4f}")
        else:
            summary_parts.append(f"{name} loss: n/a")

    title = " | ".join(summary_parts)
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    else:
        title = f"Model Fits and Predictions vs Observed Target Responses\n{title}"

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, bbox_inches="tight", dpi=140)
    plt.close()
