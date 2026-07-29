# Check the distribution of scores across different random seeds for the data split
from pathlib import Path
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.llm.utils import translate_to_jax
from edgar.scoring.scoring import score
from edgar.evolution.population import Population
from edgar.evolution.island import seed
import os
import asyncio
import sys
import matplotlib.pyplot as plt
import json

if not hasattr(sys.modules["__main__"], "__spec__"):
    sys.modules["__main__"].__spec__ = None
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()


async def score_seeds(random_seed: int = 42):
    project_root = Path(__file__).parent.parent.parent
    path = project_root / "projects" / "trial_variability" / "config.yaml"

    print(f"Loading config from: {path}")
    config = Config.from_yaml(path)
    spec = TaskSpec.from_config(config)
    print("Loading data...")
    spec.project_params.pop(
        "random_seed", None
    )  # Remove random_seed from project_params if present
    X_discover, X_validate, X_eval = spec.load_data_fn(
        data_path=spec.io["data_path"], **spec.project_params, random_seed=random_seed
    )

    # Do naive jax translation
    population = Population()
    for program in spec.seed_programs:
        program.code.model_jax = translate_to_jax(program.code.model)

    islands = seed(population, spec.seed_programs, n_islands=1)

    # Scoring
    # Discover scoring
    score(population, X_discover, X_eval, spec.scoring, spec.loss_fn, split="discover")
    # Validate scoring
    population.prepare_validation_scoring(islands=islands)
    score(population, X_validate, None, spec.scoring, spec.loss_fn, split="validate")
    return population


def main():
    scores = {
        "Seed 1": {
            "Discover": {"init": [], "final": []},
            "Validate": {"init": [], "final": []},
        },
        "Seed 2": {
            "Discover": {"init": [], "final": []},
            "Validate": {"init": [], "final": []},
        },
    }
    for rseed in [42, 43, 44, 45, 46]:
        print(f"\n\n\nScoring for random seed: {rseed}")
        pop = asyncio.run(score_seeds(random_seed=rseed))
        scores["Seed 1"]["Discover"]["init"].append(pop[0].program_losses.discover.init)
        scores["Seed 1"]["Discover"]["final"].append(
            pop[0].program_losses.discover.final
        )
        scores["Seed 1"]["Validate"]["init"].append(pop[0].program_losses.validate.init)
        scores["Seed 1"]["Validate"]["final"].append(
            pop[0].program_losses.validate.final
        )
        scores["Seed 2"]["Discover"]["init"].append(pop[1].program_losses.discover.init)
        scores["Seed 2"]["Discover"]["final"].append(
            pop[1].program_losses.discover.final
        )
        scores["Seed 2"]["Validate"]["init"].append(pop[1].program_losses.validate.init)
        scores["Seed 2"]["Validate"]["final"].append(
            pop[1].program_losses.validate.final
        )
    print("\n\n\nFinal Scores across different random seeds:")
    print(scores)
    return scores


if __name__ == "__main__":
    if not os.path.exists("seed_scores.json"):
        scores = main()
        with open("seed_scores.json", "w") as f:
            json.dump(scores, f)
    else:
        with open("seed_scores.json", "r") as f:
            scores = json.load(f)

    # Plot the results
    for seed, splits in scores.items():
        for split, losses in splits.items():
            plt.plot(losses["final"], label=f"{seed} - {split} - Final", marker="o")

    plt.xlabel("Data split random seed")
    plt.ylabel("Score")
    plt.title("Distribution of Scores across Different Random Seeds")
    plt.legend()
    plt.savefig("seed_score_distribution.png")
