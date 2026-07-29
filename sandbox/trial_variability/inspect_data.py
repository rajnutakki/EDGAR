# ruff: noqa: E402
import sys
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import jax

# Fix multiprocessing spawn issues where __main__ might lack __spec__
if not hasattr(sys.modules["__main__"], "__spec__"):
    import importlib.machinery

    sys.modules["__main__"].__spec__ = importlib.machinery.ModuleSpec(
        name="__main__", loader=None, origin=sys.argv[0] if sys.argv else None
    )

# Configure JAX memory allocation flags to avoid CUDA OOM errors during scoring
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

repo_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(repo_root / "projects" / "trial_variability" / "data_loader"))

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.evolution.population import Population
from edgar.scoring.scoring import score


def bin_trials_by_angle(theta, Y, bin_width_deg=2):
    """
    Bin trials by angle and pad with NaN to match the maximum number of repeats.

    Parameters:
    - theta: array of stimulus angles (radians)
    - Y: array of responses (trials x cells)
    - bin_width_deg: width of each bin in degrees

    Returns:
    - binned_data: array of shape (max_repeats, n_bins, n_cells)
    - bins: array of shape (n_bins,) representing bin centers
    """
    n_bins = int(round(360 / bin_width_deg))
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_cells = Y.shape[1]

    bins_trials = []
    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        mask = (theta >= low) & (theta < high)
        bins_trials.append(Y[mask])

    max_repeats = max(len(b) for b in bins_trials)
    padded_bins = []
    for b in bins_trials:
        if len(b) < max_repeats:
            padded = np.vstack([b, np.full((max_repeats - len(b), n_cells), np.nan)])
        else:
            padded = b
        padded_bins.append(padded)

    binned_data = np.stack(padded_bins, axis=1)
    print(f"Binned data shape: {binned_data.shape} (max_repeats, n_bins, n_cells)")
    return binned_data, bins


def load_and_score_seeds() -> Population | None:
    """Load the trial_variability config and task spec, then score seed programs.

    This function reproduces the scoring process of seed models using EDGAR's
    internal population, task spec, and evaluation modules.

    Returns:
        The populated and scored Population object, or None if config loading fails.
    """
    config_path = repo_root / "projects" / "trial_variability" / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return None

    config = Config.from_yaml(config_path)
    spec = TaskSpec.from_config(config)

    print("Loading task data for seed scoring...")
    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )
    if os.path.exists(
        repo_root / "sandbox" / "trial_variability" / "scored_seeds.jsonl"
    ):
        print("Scored seeds already exist. Loading...")
        population = Population.load(
            repo_root / "sandbox" / "trial_variability" / "scored_seeds.jsonl"
        )

    else:
        print("Scored seeds not found. Proceeding to score seed programs...")
        population = Population()
        for seed_p in spec.seed_programs:
            if not seed_p.code.model_jax:
                seed_p.code.model_jax = seed_p.code.model.replace(
                    "import numpy as np", "import jax.numpy as jnp"
                ).replace("np.", "jnp.")
            population.add(seed_p)

        print("Scoring seed programs...")
        score(
            population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover"
        )
        population.save(
            repo_root / "sandbox" / "trial_variability" / "scored_seeds.jsonl"
        )

        print("\n--- Seed Program Scores ---")
        for idx in range(len(population)):
            prog = population[idx]
            loss_val = prog.program_losses.discover.final
            print(f"Program {idx + 1} ({prog.name}): final loss = {loss_val}")

    return population, X_discover, X_validate


if __name__ == "__main__":
    # Load and score seed programs
    print("\n--- Loading and scoring seed programs ---")
    population, X_disc, X_val = load_and_score_seeds()
    model_predictions = [
        jax.vmap(p.compile_model(), in_axes=(0, 0))(X_val[1], p.params)[0]
        for p in population
    ]  # each element (n_trials, n_cells

    theta = X_val[1]["stimulus"][0]
    sort_idx = theta.argsort()
    theta_s = theta[sort_idx]

    binned_reponses = [
        bin_trials_by_angle(theta_s, mp, bin_width_deg=3) for mp in model_predictions
    ]

    # Make the plot
    fig = plt.figure(figsize=(16, 24))
    outer_gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    axes = np.empty((3, 2), dtype=object)
    axes_err = np.empty((3, 2), dtype=object)

    for r in range(3):
        for c in range(2):
            inner_gs = outer_gs[r, c].subgridspec(
                2, 1, height_ratios=[4, 1.2], hspace=0.08
            )
            axes[r, c] = fig.add_subplot(inner_gs[0])
            axes_err[r, c] = fig.add_subplot(inner_gs[1], sharex=axes[r, c])

    line_colors = ["red", "blue"]
    alphas = [0.8, 0.5]

    cell_idx = X_val[1]["response"].shape[1] // 2 + 1  # Start from middle cell

    # Calculate stats ignoring NaNs
    actual_response = X_val[1]["response"][0]
    for i, (Y_binned, theta_bins) in enumerate(binned_reponses):
        cell_responses = Y_binned[:, :, cell_idx]
        median_responses = np.nanmedian(cell_responses, axis=0)
        pct_5 = np.nanpercentile(cell_responses, 5, axis=0)
        pct_95 = np.nanpercentile(cell_responses, 95, axis=0)

        # --- Plot on Column 0, Row 0 (Unbinned Tuning Curves) ---
        mp = model_predictions[i]
        axes[0, 0].plot(
            theta_s,
            mp[sort_idx, cell_idx],
            color=line_colors[i],
            linewidth=1.5,
            alpha=alphas[i],
            label=f"Prediction (Model {i + 1})",
        )

        sq_err = (mp[:, cell_idx] - actual_response[:, cell_idx]) ** 2
        axes_err[0, 0].plot(
            theta_s,
            sq_err[sort_idx],
            color=line_colors[i],
            linewidth=1.5,
            alpha=alphas[i],
            label=f"Squared Error (Model {i + 1})",
        )

        # --- Plot on Column 1, Row 0 (Binned Responses) ---
        # 1. Plot binned median response
        axes[0, 1].plot(
            theta_bins,
            median_responses,
            color=line_colors[i],
            linewidth=1.5,
            label=f"Median Response (Model {i + 1})",
        )

        # 2. Plot shaded ribbon for 5th and 95th percentile responses
        axes[0, 1].fill_between(
            theta_bins,
            pct_5,
            pct_95,
            color=line_colors[i],
            alpha=0.15,
            label=f"5th-95th Percentile (Model {i + 1})",
        )

        # 3. Plot squared error on the panel below axes[0,1]
        axes_err[0, 1].plot(
            theta_s,
            sq_err[sort_idx],
            color=line_colors[i],
            linewidth=1.5,
            alpha=alphas[i],
            label=f"Squared Error (Model {i + 1})",
        )

    # --- Final touches for Column 0, Row 0 ---
    # Estimate density for theta vs actual_response[:, cell_idx]
    x_coords_00 = theta
    y_coords_00 = actual_response[:, cell_idx]
    xy_coords_00 = np.vstack([x_coords_00, y_coords_00])
    from scipy.stats import gaussian_kde

    density_00 = gaussian_kde(xy_coords_00)(xy_coords_00)

    axes[0, 0].scatter(
        x_coords_00,
        y_coords_00,
        c=density_00,
        cmap="viridis",
        s=15,
        alpha=0.8,
        label="Observed",
        zorder=10,
    )
    axes[0, 0].set_title(f"Tuning Curve: Cell {cell_idx}")
    axes[0, 0].set_ylabel("Response")
    axes[0, 0].legend()
    axes[0, 0].tick_params(labelbottom=False)

    axes_err[0, 0].set_xlabel("Theta (radians)")
    axes_err[0, 0].set_ylabel("Squared Error")

    # --- Final touches for Column 1, Row 0 ---
    axes[0, 1].scatter(
        x_coords_00,
        y_coords_00,
        c=density_00,
        cmap="viridis",
        s=15,
        alpha=0.8,
        label="Observed",
        zorder=10,
    )
    axes[0, 1].set_title(f"Binned Responses over Stimulus Angle (Cell {cell_idx})")
    axes[0, 1].set_ylabel("Normalized Response")
    axes[0, 1].legend()
    axes[0, 1].tick_params(labelbottom=False)

    axes_err[0, 1].set_xlabel("Theta (radians)")
    axes_err[0, 1].set_ylabel("Squared Error")

    # --- Plot on Column 0, Row 1 (axes[1,0] and axes_err[1,0]) ---
    cell_mid = actual_response.shape[1] // 2
    trial_idx = actual_response.shape[0] // 2 + 1
    sort_idx_cell = np.argsort(model_predictions[1][trial_idx, cell_mid:])
    n_cells = actual_response.shape[1] - cell_mid

    # Plot Model 2 (blue) response
    axes[1, 0].plot(
        np.arange(n_cells),
        model_predictions[1][trial_idx, cell_mid:][sort_idx_cell],
        color=line_colors[1],
        linewidth=1.5,
        alpha=alphas[1],
        label="Prediction (Model 2)",
    )

    # Plot Model 1 (red) response
    axes[1, 0].plot(
        np.arange(n_cells),
        model_predictions[0][trial_idx, cell_mid:][sort_idx_cell],
        color=line_colors[0],
        linewidth=1.5,
        alpha=alphas[0],
        label="Prediction (Model 1)",
    )

    # Plot Observed/Actual response with a colormap representing preferred orientation
    x_coords = np.arange(n_cells)
    y_coords = actual_response[trial_idx, cell_mid:][sort_idx_cell]

    # Preferred orientation of the cells (computed from observed responses across trials)
    sig_subset = actual_response[:, cell_mid:]
    complex_sum = np.nansum(sig_subset * np.exp(2j * theta)[:, np.newaxis], axis=0)
    pref_angles = (np.angle(complex_sum) / 2.0) % np.pi

    sc = axes[1, 0].scatter(
        x_coords,
        y_coords,
        c=pref_angles[sort_idx_cell],
        cmap="twilight",
        vmin=0,
        vmax=np.pi,
        s=15,
        alpha=0.8,
        label="Observed",
        zorder=10,
    )
    # Add layout-preserving colorbar across both subplots (axes[1,0] and axes_err[1,0]) to keep them aligned
    cbar = fig.colorbar(sc, ax=[axes[1, 0], axes_err[1, 0]], pad=0.02, aspect=25)
    cbar.set_label("Pref. Orientation (rad)")

    # Add an indicator line on the colorbar showing the stimulus angle for the trial (wrapped to [0, pi])
    stim_wrapped = theta[trial_idx] % np.pi
    cbar.ax.axhline(stim_wrapped, color="black", linewidth=2, linestyle="--")

    # Plot squared errors
    sq_err_m1 = (
        model_predictions[0][trial_idx, cell_mid:][sort_idx_cell]
        - actual_response[trial_idx, cell_mid:][sort_idx_cell]
    ) ** 2
    axes_err[1, 0].plot(
        np.arange(n_cells),
        sq_err_m1,
        color=line_colors[0],
        linewidth=1.5,
        alpha=alphas[0],
        label="Squared Error (Model 1)",
    )

    sq_err_m2 = (
        model_predictions[1][trial_idx, cell_mid:][sort_idx_cell]
        - actual_response[trial_idx, cell_mid:][sort_idx_cell]
    ) ** 2
    axes_err[1, 0].plot(
        np.arange(n_cells),
        sq_err_m2,
        color=line_colors[1],
        linewidth=1.5,
        alpha=alphas[1],
        label="Squared Error (Model 2)",
    )

    # Final touches for axes[1,0] and axes_err[1,0]
    axes[1, 0].set_title(
        f"Trial Response Profile: Trial {trial_idx}\n(Stimulus: {theta[trial_idx]:.2f} rad)"
    )
    axes[1, 0].set_ylabel("Response")
    axes[1, 0].legend()
    axes[1, 0].tick_params(labelbottom=False)

    axes_err[1, 0].set_xlabel("Cells (sorted by Model 2 prediction)")
    axes_err[1, 0].set_ylabel("Squared Error")

    # --- Plot on Column 1, Row 1 (axes[1,1] and axes_err[1,1]) ---
    response_T_test = actual_response[:, cell_mid:].T  # (n_cells, n_trials)
    numerator_vc = np.nansum(
        np.exp(2j * theta)[np.newaxis, :] * response_T_test, axis=1
    )
    denominator_vc = np.nansum(response_T_test, axis=1)
    conc = np.abs(numerator_vc / np.where(denominator_vc == 0, 1.0, denominator_vc))
    sort_idx_vc = np.argsort(conc)

    # Plot Model 2 (blue) response
    axes[1, 1].plot(
        np.arange(n_cells),
        model_predictions[1][trial_idx, cell_mid:][sort_idx_vc],
        color=line_colors[1],
        linewidth=1.5,
        alpha=alphas[1],
        label="Prediction (Model 2)",
    )

    # Plot Model 1 (red) response
    axes[1, 1].plot(
        np.arange(n_cells),
        model_predictions[0][trial_idx, cell_mid:][sort_idx_vc],
        color=line_colors[0],
        linewidth=1.5,
        alpha=alphas[0],
        label="Prediction (Model 1)",
    )

    # Plot Observed/Actual response with a density colormap
    x_coords_11 = np.arange(n_cells)
    y_coords_11 = actual_response[trial_idx, cell_mid:][sort_idx_vc]

    # Estimate density using gaussian_kde
    xy_coords_11 = np.vstack([x_coords_11, y_coords_11])
    density_11 = gaussian_kde(xy_coords_11)(xy_coords_11)

    axes[1, 1].scatter(
        x_coords_11,
        y_coords_11,
        c=density_11,
        cmap="viridis",
        s=15,
        alpha=0.8,
        label="Observed",
        zorder=10,
    )

    # Plot squared errors
    sq_err_m1_vc = (
        model_predictions[0][trial_idx, cell_mid:][sort_idx_vc]
        - actual_response[trial_idx, cell_mid:][sort_idx_vc]
    ) ** 2
    axes_err[1, 1].plot(
        np.arange(n_cells),
        sq_err_m1_vc,
        color=line_colors[0],
        linewidth=1.5,
        alpha=alphas[0],
        label="Squared Error (Model 1)",
    )

    sq_err_m2_vc = (
        model_predictions[1][trial_idx, cell_mid:][sort_idx_vc]
        - actual_response[trial_idx, cell_mid:][sort_idx_vc]
    ) ** 2
    axes_err[1, 1].plot(
        np.arange(n_cells),
        sq_err_m2_vc,
        color=line_colors[1],
        linewidth=1.5,
        alpha=alphas[1],
        label="Squared Error (Model 2)",
    )

    # Final touches for axes[1,1] and axes_err[1,1]
    axes[1, 1].set_title(
        f"Trial Response Profile (Sorted by VC):\nTrial {trial_idx} (Stimulus: {theta[trial_idx]:.2f} rad)"
    )
    axes[1, 1].set_ylabel("Response")
    axes[1, 1].legend()
    axes[1, 1].tick_params(labelbottom=False)

    axes_err[1, 1].set_xlabel("Cells (sorted by vector concentration)")
    axes_err[1, 1].set_ylabel("Squared Error")

    # --- Plot on Column 0, Row 2 (axes[2,0] and axes_err[2,0]) ---
    # Center the test-subset responses and perform SVD to find the PC loadings
    Y_test = actual_response[:, cell_mid:]
    Y_clean = np.where(np.isnan(Y_test), 0, Y_test)
    Y_centered = Y_clean - np.mean(Y_clean, axis=0)

    _, _, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    loadings_pc1 = Vt[0, :]
    sort_idx_pc1 = np.argsort(loadings_pc1)

    # Plot Model 2 (blue) response
    axes[2, 0].plot(
        np.arange(n_cells),
        model_predictions[1][trial_idx, cell_mid:][sort_idx_pc1],
        color=line_colors[1],
        linewidth=1.5,
        alpha=alphas[1],
        label="Prediction (Model 2)",
    )

    # Plot Model 1 (red) response
    axes[2, 0].plot(
        np.arange(n_cells),
        model_predictions[0][trial_idx, cell_mid:][sort_idx_pc1],
        color=line_colors[0],
        linewidth=1.5,
        alpha=alphas[0],
        label="Prediction (Model 1)",
    )

    # Plot Observed/Actual response with a density colormap
    x_coords_20 = np.arange(n_cells)
    y_coords_20 = actual_response[trial_idx, cell_mid:][sort_idx_pc1]

    # Estimate density using gaussian_kde
    xy_coords_20 = np.vstack([x_coords_20, y_coords_20])
    density_20 = gaussian_kde(xy_coords_20)(xy_coords_20)

    axes[2, 0].scatter(
        x_coords_20,
        y_coords_20,
        c=density_20,
        cmap="viridis",
        s=15,
        alpha=0.8,
        label="Observed",
        zorder=10,
    )

    # Plot squared errors
    sq_err_m1_pc1 = (
        model_predictions[0][trial_idx, cell_mid:][sort_idx_pc1]
        - actual_response[trial_idx, cell_mid:][sort_idx_pc1]
    ) ** 2
    axes_err[2, 0].plot(
        np.arange(n_cells),
        sq_err_m1_pc1,
        color=line_colors[0],
        linewidth=1.5,
        alpha=alphas[0],
        label="Squared Error (Model 1)",
    )

    sq_err_m2_pc1 = (
        model_predictions[1][trial_idx, cell_mid:][sort_idx_pc1]
        - actual_response[trial_idx, cell_mid:][sort_idx_pc1]
    ) ** 2
    axes_err[2, 0].plot(
        np.arange(n_cells),
        sq_err_m2_pc1,
        color=line_colors[1],
        linewidth=1.5,
        alpha=alphas[1],
        label="Squared Error (Model 2)",
    )

    # Final touches for axes[2,0] and axes_err[2,0]
    axes[2, 0].set_title(
        f"Trial Response Profile (Sorted by PC1):\nTrial {trial_idx} (Stimulus: {theta[trial_idx]:.2f} rad)"
    )
    axes[2, 0].set_ylabel("Response")
    axes[2, 0].legend()
    axes[2, 0].tick_params(labelbottom=False)

    axes_err[2, 0].set_xlabel("Cells (sorted by first PC loading)")
    axes_err[2, 0].set_ylabel("Squared Error")

    # --- Plot on Column 1, Row 2 (axes[2,1] and axes_err[2,1]) ---
    loadings_pc2 = Vt[1, :]
    sort_idx_pc2 = np.argsort(loadings_pc2)

    # Plot Model 2 (blue) response
    axes[2, 1].plot(
        np.arange(n_cells),
        model_predictions[1][trial_idx, cell_mid:][sort_idx_pc2],
        color=line_colors[1],
        linewidth=1.5,
        alpha=alphas[1],
        label="Prediction (Model 2)",
    )

    # Plot Model 1 (red) response
    axes[2, 1].plot(
        np.arange(n_cells),
        model_predictions[0][trial_idx, cell_mid:][sort_idx_pc2],
        color=line_colors[0],
        linewidth=1.5,
        alpha=alphas[0],
        label="Prediction (Model 1)",
    )

    # Plot Observed/Actual response with a density colormap
    x_coords_21 = np.arange(n_cells)
    y_coords_21 = actual_response[trial_idx, cell_mid:][sort_idx_pc2]

    # Estimate density using gaussian_kde
    xy_coords_21 = np.vstack([x_coords_21, y_coords_21])
    density_21 = gaussian_kde(xy_coords_21)(xy_coords_21)

    axes[2, 1].scatter(
        x_coords_21,
        y_coords_21,
        c=density_21,
        cmap="viridis",
        s=15,
        alpha=0.8,
        label="Observed",
        zorder=10,
    )

    # Plot squared errors
    sq_err_m1_pc2 = (
        model_predictions[0][trial_idx, cell_mid:][sort_idx_pc2]
        - actual_response[trial_idx, cell_mid:][sort_idx_pc2]
    ) ** 2
    axes_err[2, 1].plot(
        np.arange(n_cells),
        sq_err_m1_pc2,
        color=line_colors[0],
        linewidth=1.5,
        alpha=alphas[0],
        label="Squared Error (Model 1)",
    )

    sq_err_m2_pc2 = (
        model_predictions[1][trial_idx, cell_mid:][sort_idx_pc2]
        - actual_response[trial_idx, cell_mid:][sort_idx_pc2]
    ) ** 2
    axes_err[2, 1].plot(
        np.arange(n_cells),
        sq_err_m2_pc2,
        color=line_colors[1],
        linewidth=1.5,
        alpha=alphas[1],
        label="Squared Error (Model 2)",
    )

    # Final touches for axes[2,1] and axes_err[2,1]
    axes[2, 1].set_title(
        f"Trial Response Profile (Sorted by PC2):\nTrial {trial_idx} (Stimulus: {theta[trial_idx]:.2f} rad)"
    )
    axes[2, 1].set_ylabel("Response")
    axes[2, 1].legend()
    axes[2, 1].tick_params(labelbottom=False)

    axes_err[2, 1].set_xlabel("Cells (sorted by second PC loading)")
    axes_err[2, 1].set_ylabel("Squared Error")

    plt.savefig(str(repo_root) + "/sandbox/trial_variability/data_summary.png")
