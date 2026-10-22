# SciPy optimizer profiling (synaptic transmission)

`benchmarks/scipy_synaptic.py` uses seed model 0, the Neher
`NM_FSIN_4Pr_v3.csv` parameters, seed 11, a 30% perturbation, the discovery
train/test split with 30,000 ms equilibration, and Nelder-Mead. It measures a
full EDGAR `score()` (including spawn, initial/final evaluation, and per-sample
losses) and a direct `scipy.optimize.minimize` on the *same training objective*.
All runs used `JAX_PLATFORMS=cpu` and `uv run python benchmarks/scipy_synaptic.py
--maxiter N`; before = `fdbc434`, after = this branch, each in a fresh process.
The direct objective is intentionally not jitted as a whole, matching the
original comparison script. Times are wall seconds from one run each, not
statistical estimates.

| Max iterations | Before score | After score | Direct SciPy before / after | Train loss (before / after / direct) |
|---:|---:|---:|---:|---:|
| 20 | 24.43 s | 2.19 s | 14.99 / 14.79 s | 0.22776441 / 0.22776441 / 0.22776444 |
| 100 | 97.61 s | 3.58 s | 56.76 / 55.67 s | 0.11876548 / 0.11876550 / 0.11876549 |

The old objective rebuilt batched parameter pytrees in Python and invoked
`evaluate_scalar_loss` without a JIT at each function evaluation **and** each
iteration callback. Although the project `apply_model` was JIT-compiled, loss
operations (gather, masking, reductions) and scalar conversion still dispatched
separately every time. The new objective JIT-compiles unflattening, model
application and loss/reduction together, passing sample data as dynamic inputs
so samples with identical shapes reuse the executable. A test counts one trace
across two samples and checks their independent fitted parameters.

The final `score_final_loss` is evaluated on the held-out test protocols;
the trajectory and direct SciPy losses above are on the training protocols.
The change does not avoid one-off compilation per model/shape, worker startup,
or the initial/final/fingerprint scoring phases. Other backends, models and
dataset shapes can have different speedups; JIT-compile time is included here.
