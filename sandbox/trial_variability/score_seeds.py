import asyncio
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.scoring.scoring import _get_params, _eval_loss, _optimize
from pathlib import Path
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()


async def main():
    # Use absolute path for robustness in sandbox
    project_root = Path(__file__).parent.parent.parent
    path = project_root / "projects" / "trial_variability" / "config.yaml"

    print(f"Loading config from: {path}")
    config = Config.from_yaml(path)
    spec = TaskSpec.from_config(config)

    # Load data
    print("Loading data...")
    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params
    )

    if isinstance(spec.loss_fn, tuple):
        loss_fn_train, loss_fn_test = spec.loss_fn
    else:
        loss_fn_train = loss_fn_test = spec.loss_fn

    for idx, seed in enumerate(spec.seed_programs):
        print("\n==========================================")
        print(f"Scoring {seed.name} (Seed {idx + 1}/{len(spec.seed_programs)})")
        print("==========================================")

        # Translate model code to JAX
        model_numpy = seed.code.model
        model_jax = model_numpy.replace(
            "import numpy as np", "import jax.numpy as jnp"
        ).replace("np.", "jnp.")
        seed.code.model_jax = model_jax

        print("\n--- Translated JAX Model ---")
        print(seed.code.model_jax)
        print("----------------------------\n")

        # Compile model & param estimator
        model_fn = seed.compile_model()
        param_est_fn = seed.compile_param_est()

        # Get initial parameters
        params_init = _get_params(param_est_fn, seed.default_params, X_discover[0])

        print("Initial Parameters:")
        for k, v in params_init.items():
            print(f"  {k}: shape {v.shape if hasattr(v, 'shape') else type(v)}")

        # Evaluate initial loss on test split (X_discover[1]) using loss_fn_test
        initial_loss = _eval_loss(model_fn, loss_fn_test, params_init, X_discover[1])
        print(f"\nInitial loss: {initial_loss:.4f}")

        # Optimize on train split (X_discover[0]) using loss_fn_train
        print("\nOptimizing parameters...")
        params = _optimize(
            model_fn,
            loss_fn_train,
            params_init,
            X_discover[0],
            spec.scoring["gradient_descent"],
        )

        print("\nOptimized Parameters:")
        for k, v in params.items():
            print(f"  {k}: shape {v.shape if hasattr(v, 'shape') else type(v)}")

        # Evaluate final loss on test split (X_discover[1]) using loss_fn_test
        final_loss = _eval_loss(model_fn, loss_fn_test, params, X_discover[1])
        print(f"\nFinal loss: {final_loss:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
