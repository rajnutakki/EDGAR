#Take programs from a run and run optimizations with perturbed initial parameters to see how stable / unstable the optimization is.
from pathlib import Path
from edgar.evolution.population import Population
from edgar.io.config import Config
from edgar.io.task_spec import TaskSpec
from edgar.scoring.scoring import _get_params, _eval_loss, _optimize
import jax.numpy as jnp
import numpy as np
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_enable_command_buffer=").strip()

repo_root = Path(__file__).resolve().parent.parent.parent

def optimize_with_perturbations(program, data, loss_fn_train, loss_fn_test, epsilons):
    model_fn = program.compile_model()
    param_est_fn = program.compile_param_est()
    params_init = _get_params(param_est_fn, program.default_params, data[0])
    param_sets = [params_init]

    for epsilon in epsilons:
        perturbed_params = {k: v*np.random.normal(loc=1,size=v.shape,scale=epsilon) for k, v in params_init.items()}
        param_sets.append(perturbed_params)

    #Run optimization for each initial parameter set
    epsilons = [0] + list(epsilons)
    for i, params_init in enumerate(param_sets):
        print(f"Epsilon = {epsilons[i]:.6f}")
        initial_loss = _eval_loss(
                        model_fn, loss_fn_test, params_init, data[1]
                    )
        params_opt, _ = _optimize(
                        model_fn, loss_fn_train, params_init, data[0], spec.scoring["gradient_descent"]
                    )
        params_opt = params_opt[0]
        final_loss = _eval_loss(
                        model_fn, loss_fn_test, params_opt, data[1]
                    )
        print(f"Losses (no complexity penalty)- Initial: {initial_loss:.4f}, Final: {final_loss:.4f}")
        print(f"Losses (complexity penalty)- Initial: {initial_loss+spec.scoring['param_penalty_weight']*program.n_params:.4f}, Final: {final_loss+spec.scoring['param_penalty_weight']*program.n_params:.4f}")
        # print("Initial parameters:")
        # for k,v in params_init.items():
        #     print(f"- {k}: shape {v.shape}, mean {jnp.nanmean(v):.4f}, std: {jnp.nanstd(v):.4f}")
        # print("Optimized parameters:")
        # for k, v in params_opt.items():
        #     print(f"- {k}: shape {v.shape}, mean {jnp.nanmean(v):.4f}, std: {jnp.nanstd(v):.4f}")

if __name__ == "__main__":
    run_path = "/home/rajah/projects/trial_variability/run_output/tv-artifacts1/2026-07-31/16-06-04"
    pop = Population.load(f"{run_path}/population.jsonl")
    #spec = TaskSpec.from_config(Config.from_taskspec(f"{run_path}/task_spec.yaml"))
    spec = TaskSpec.from_config(Config.from_yaml(f"{repo_root}/projects/trial_variability/config.yaml"))
    seeds = [pop[0], pop[1]]
    pop = pop.get_sorted()
    top_programs = pop[:3]
    programs = seeds + top_programs

    X_discover, _, _ = spec.load_data_fn(data_path=spec.io["data_path"], **spec.project_params)
    loss_fn_train, loss_fn_test = spec.loss_fn

    for program in programs:
        print(f"Optimizing program: {program.name}")
        optimize_with_perturbations(program, X_discover, loss_fn_train, loss_fn_test, epsilons=[0.01, 0.1, 0.5, 1]) 


