# ruff: noqa: E402
import numpy as np
import sys
import os
import jax
from pathlib import Path

# Setup JAX and pathing
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

# Add absolute project root to sys.path for importing project stuff
project_root = str(Path(__file__).parent.parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.scoring.scoring import _get_params, _eval_loss, _optimize
from edgar.llm.utils import translate_to_jax
from projects.trial_variability.image_feedback.plot import plot_model_fits


def main():
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "projects" / "trial_variability" / "config.yaml"

    print(f"Loading config from: {config_path}")
    config = Config.from_yaml(config_path)
    spec = TaskSpec.from_config(config)

    # Load data
    print("Loading data...")
    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )

    # We will analyze both Seed Models in a loop
    for seed_idx, seed in enumerate(spec.seed_programs):
        model_num = seed_idx + 1
        print("\n" + "=" * 50)
        print(f"       ANALYZING SEED MODEL {model_num} ({seed.name})")
        print("=========================================")

        # Assign JAX model dynamically
        seed.code.model_jax = translate_to_jax(seed.code.model)

        # Compile Model and Param Est
        print(f"Compiling JAX model {model_num}...")
        model_fn = seed.compile_model()
        param_est_fn = seed.compile_param_est()

        # 1. EVALUATE WITH DEFAULT PARAMETERS
        print("\n--- 1. Evaluating with Default Parameters ---")
        params_default = _get_params(None, seed.default_params, X_discover[0])
        loss_default = _eval_loss(model_fn, spec.loss_fn, params_default, X_discover[1])
        print(f"Default Params Validation Loss: {loss_default:.4f}")

        # 2. EVALUATE WITH PARAMETER ESTIMATOR
        print("\n--- 2. Evaluating with Parameter Estimator ---")
        params_est = _get_params(param_est_fn, seed.default_params, X_discover[0])
        loss_est = _eval_loss(model_fn, spec.loss_fn, params_est, X_discover[1])
        print(f"Estimated Params Validation Loss: {loss_est:.4f}")

        # 3. EVALUATE AFTER OPTIMIZATION (STARTING FROM ESTIMATOR)
        print("\n--- 3. Optimizing starting from Estimator ---")
        params_opt = _optimize(
            model_fn,
            spec.loss_fn,
            params_est,
            X_discover[0],
            spec.scoring["gradient_descent"],
        )
        loss_opt = _eval_loss(model_fn, spec.loss_fn, params_opt, X_discover[1])
        print(f"Optimized from Est Validation Loss: {loss_opt:.4f}")

        # 4. EVALUATE AFTER OPTIMIZATION (STARTING FROM DEFAULT PARAMS)
        print("\n--- 4. Optimizing starting from Default Params ---")
        params_opt_default = _optimize(
            model_fn,
            spec.loss_fn,
            params_default,
            X_discover[0],
            spec.scoring["gradient_descent"],
        )
        loss_opt_default = _eval_loss(
            model_fn, spec.loss_fn, params_opt_default, X_discover[1]
        )
        print(f"Optimized from Default Validation Loss: {loss_opt_default:.4f}")

        # --- Print Summary Table ---
        print(f"\n================ SUMMARY MODEL {model_num} ================")
        print(f"1. Default Parameters:           Loss = {loss_default:.4f}")
        print(f"2. Parameter Estimator:          Loss = {loss_est:.4f}")
        print(f"3. Optimized (from Estimator):   Loss = {loss_opt:.4f}")
        print(f"4. Optimized (from Default):     Loss = {loss_opt_default:.4f}")
        print("=========================================")

        # --- DIAGNOSTIC PLOTS ---
        print(f"\nGenerating diagnostic plots for Model {model_num}...")
        import matplotlib.pyplot as plt

        sample_idx = 0
        theta_np = np.asarray(X_discover[0]["stimulus"][sample_idx])  # (n_trials,)
        y_np = np.asarray(X_discover[0]["response"][sample_idx])  # (n_trials, n_cells)

        # Generate predictions for all sets of parameters
        single_sample_data = {k: v[sample_idx] for k, v in X_discover[0].items()}

        # 1. Default
        default_params_single = jax.tree_util.tree_map(
            lambda x: x[sample_idx], params_default
        )
        y_pred_default = np.asarray(model_fn(single_sample_data, default_params_single))

        # 2. Estimated
        est_params_single = jax.tree_util.tree_map(lambda x: x[sample_idx], params_est)
        y_pred_est = np.asarray(model_fn(single_sample_data, est_params_single))

        # 3. Optimized (from Estimator)
        opt_params_single = jax.tree_util.tree_map(lambda x: x[sample_idx], params_opt)
        y_pred_opt = np.asarray(model_fn(single_sample_data, opt_params_single))

        # 4. Optimized (from Default)
        opt_default_params_single = jax.tree_util.tree_map(
            lambda x: x[sample_idx], params_opt_default
        )
        y_pred_opt_default = np.asarray(
            model_fn(single_sample_data, opt_default_params_single)
        )

        # Pick 4 cells to plot with valid responses
        valid_cell_indices = np.where(~np.isnan(y_np).all(axis=0))[0]
        rng = np.random.default_rng(42)
        chosen_cells = rng.choice(valid_cell_indices, size=4, replace=False)

        # Plot 1: Standard Population Fit Comparison Panel (Hacked to show all 4 parameter sets)
        mock_programs = [
            {"model": model_fn},
            {"model": model_fn},
            {"model": model_fn},
            {"model": model_fn},
        ]
        plot_model_fits(
            data=X_discover[0],
            programs=mock_programs,
            save_path=f"sandbox/trial_variability/fit_comparison_model{model_num}.png",
            losses=[loss_default, loss_est, loss_opt, loss_opt_default],
            params=[params_default, params_est, params_opt, params_opt_default],
            program_names=["Default", "Estimated", "Opt from Est", "Opt from Default"],
            title_prefix=f"Seed Model {model_num} Comparison Across Parameter Sets",
        )
        print(
            f"Saved standard population fit plot to sandbox/trial_variability/fit_comparison_model{model_num}.png"
        )

        # Plot 2: Tuning Curves (Response vs Stimulus Angle)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, cell in enumerate(chosen_cells):
            ax = axes.flatten()[idx]
            sort_idx = np.argsort(theta_np)

            # Actual responses (scatter because they are noisy trials)
            ax.scatter(
                theta_np,
                y_np[:, cell],
                color="gray",
                alpha=0.3,
                label="Actual Responses",
            )

            # Default Params (dotted orange)
            ax.plot(
                theta_np[sort_idx],
                y_pred_default[sort_idx, cell],
                color="orange",
                linestyle=":",
                linewidth=1.5,
                label="Default Params",
            )

            # Estimated Params (dashed blue)
            ax.plot(
                theta_np[sort_idx],
                y_pred_est[sort_idx, cell],
                color="tab:blue",
                linestyle="--",
                linewidth=2.0,
                label="Estimated Params",
            )

            # Optimized (from Estimator - solid red)
            ax.plot(
                theta_np[sort_idx],
                y_pred_opt[sort_idx, cell],
                color="red",
                linestyle="-",
                linewidth=2.5,
                label="Opt from Est",
            )

            # Optimized (from Default - solid green)
            ax.plot(
                theta_np[sort_idx],
                y_pred_opt_default[sort_idx, cell],
                color="tab:green",
                linestyle="-",
                linewidth=2.0,
                label="Opt from Default",
            )

            # Clip y-axis to match the actual data range with 10% padding
            cell_y = y_np[:, cell]
            valid_y = cell_y[~np.isnan(cell_y)]
            ymin, ymax = np.min(valid_y), np.max(valid_y)
            yrange = ymax - ymin if ymax > ymin else 1.0
            padding = 0.10 * yrange
            ax.set_ylim(ymin - padding, ymax + padding)

            ax.set_title(f"Cell {cell}")
            ax.set_xlabel("Stimulus Angle (radians)")
            ax.set_ylabel("Response")
            if idx == 0:
                ax.legend()

        plt.suptitle(
            f"Model {model_num} - Tuning Curves for Example Cells", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(
            f"sandbox/trial_variability/tuning_curves_example_cells_model{model_num}.png",
            dpi=150,
        )
        plt.close()
        print(
            f"Saved tuning curves plot to sandbox/trial_variability/tuning_curves_example_cells_model{model_num}.png"
        )

        # Plot 3: Response across different trials (Trial tracking)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for idx, cell in enumerate(chosen_cells):
            ax = axes.flatten()[idx]
            n_trials_to_show = min(150, len(theta_np))
            trials_idx = np.arange(n_trials_to_show)

            # Actual responses
            ax.plot(
                trials_idx,
                y_np[trials_idx, cell],
                color="gray",
                alpha=0.4,
                marker="o",
                linestyle="-",
                label="Actual",
            )

            # Default Params (dotted orange)
            ax.plot(
                trials_idx,
                y_pred_default[trials_idx, cell],
                color="orange",
                linestyle=":",
                label="Default",
            )

            # Estimated Params (dashed blue)
            ax.plot(
                trials_idx,
                y_pred_est[trials_idx, cell],
                color="tab:blue",
                linestyle="--",
                label="Estimated",
            )

            # Optimized (from Estimator - solid red)
            ax.plot(
                trials_idx,
                y_pred_opt[trials_idx, cell],
                color="red",
                linestyle="-",
                label="Opt from Est",
            )

            # Optimized (from Default - solid green)
            ax.plot(
                trials_idx,
                y_pred_opt_default[trials_idx, cell],
                color="tab:green",
                linestyle="-",
                label="Opt from Default",
            )

            # Clip y-axis to match the actual data range with 10% padding
            cell_y = y_np[trials_idx, cell]
            valid_y = cell_y[~np.isnan(cell_y)]
            ymin, ymax = np.min(valid_y), np.max(valid_y)
            yrange = ymax - ymin if ymax > ymin else 1.0
            padding = 0.10 * yrange
            ax.set_ylim(ymin - padding, ymax + padding)

            ax.set_title(f"Cell {cell}")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Response")
            if idx == 0:
                ax.legend()

        plt.suptitle(
            f"Model {model_num} - Trial-by-Trial Responses for Example Cells",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(
            f"sandbox/trial_variability/trial_responses_example_cells_model{model_num}.png",
            dpi=150,
        )
        plt.close()
        print(
            f"Saved trial-by-trial responses plot to sandbox/trial_variability/trial_responses_example_cells_model{model_num}.png"
        )


if __name__ == "__main__":
    main()
