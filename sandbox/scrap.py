from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from pathlib import Path
from edgar.evolution.population import Population
from edgar.evolution.island import seed
from edgar.llm.utils import translate_to_jax
from edgar.scoring.scoring import score
from edgar.scoring.utils import _compute_mean_loss
import sys
import os

if not hasattr(sys.modules["__main__"], "__spec__"):
    sys.modules["__main__"].__spec__ = None
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

if __name__ == "__main__":
    config = Config.from_yaml(Path("projects/synthetic_data/config.yaml"))
    spec = TaskSpec.from_config(config)
    X_discover, X_validate, X_eval = spec.load_data_fn(
            data_path=spec.io["data_path"], **spec.project_params
        )

    print("Discovery data shape:", X_discover[0]["y"].shape)
    population = Population()
    islands = seed(population, spec.seed_programs, n_islands = 1)
    #Fake translate
    for p in population:
        p.code.model_jax = translate_to_jax(p.code.model)

    score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")


    model_fn = population[0].compile_model()
    output = _compute_mean_loss(model_fn, spec.loss_fn, population[0].params, X_discover[0])