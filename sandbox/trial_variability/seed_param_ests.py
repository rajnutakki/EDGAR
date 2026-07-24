# ruff: noqa: E402
import numpy as np
import sys
import os
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

        # Print estimated parameter averages (for setting default_params)
        print("\n--- Estimated Parameter Means (for setting default_params) ---")
        for k, v in params_est.items():
            mean_val = np.nanmean(v)
            print(f"  {k}: mean = {mean_val:.6f}")

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

        # Plot 1: Standard Population Fit Comparison Panel (Symmetrical 3x3 nested layout)
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
            rng=np.random.default_rng(),
        )
        print(
            f"Saved standard population fit plot to sandbox/trial_variability/fit_comparison_model{model_num}.png"
        )


if __name__ == "__main__":
    main()
