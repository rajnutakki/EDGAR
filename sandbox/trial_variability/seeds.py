import asyncio
import numpy as np
from edgar.io.config import Config, RetryConfig
from edgar.io.task_spec import TaskSpec
from edgar.evolution.population import Population
from edgar.llm.generate import translate_programs
from edgar.scoring.scoring import score
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

    # Initialize population with seeds
    pop = Population()
    for seed_prog in spec.seed_programs:
        # We need to make sure seed_prog has data for default_params resolution if needed
        # but TaskSpec.from_config already handles that.
        pop.add(seed_prog)

    print(f"Added {len(pop)} seed programs to population.")

    # JAX Translation
    print("Translating programs to JAX...")
    retry_config = RetryConfig(**spec.llms.get("retry", {}))
    await translate_programs(
        pop,
        spec.prompt_schemas.jax_model,
        spec.llms["jax_model_translator_llm"],
        retry_config=retry_config,
        max_tokens=spec.llms.get("max_tokens"),
        output_schema=spec.response_schemas.jax_model,
    )
    for i, p in enumerate(pop):
        print(f"Seed {i}")
        print(f"JAX translated code:\n {p.code.model_jax}\n")

    # Scoring
    print("Scoring programs...")
    score(
        pop,
        X_discover,
        X_eval,
        spec.scoring,
        spec.loss_fn,
        split="discover",
    )

    # Display results
    for i in range(len(pop)):
        p = pop[i]
        print(f"\n--- Seed {i}: {p.name} ---")
        if p.code.model_jax:
            print("JAX Translation: SUCCESS")
        else:
            print("JAX Translation: FAILED")

        print(f"Loss (init): {p.program_losses.discover.init}")
        print(f"Loss (final): {p.program_losses.discover.final}")

        if p.params:
            print(f"Optimized params: {list(p.params.keys())}")
            for k, v in p.params.items():
                print(f"  {k}: shape {v.shape}, mean {np.mean(v):.4f}")
        else:
            print("Optimized params: None")


if __name__ == "__main__":
    asyncio.run(main())
