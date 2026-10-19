#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCHMARK_DIR}/../../../.." && pwd)"
SCORER="${BENCHMARK_DIR}/score_seeds_BM_params.py"

cd "${REPO_ROOT}"

run_score() {
    local seed="$1"
    local model="$2"
    local parameter_set="$3"
    local strength="$4"
    local output="${BENCHMARK_DIR}/${model}_${parameter_set}_perturbation_${strength}.jsonl"

    echo "Scoring ${model}, ${parameter_set}, perturbation ${strength}"
    time uv run python "${SCORER}" \
        --seed "${seed}" \
        --parameter-set "${parameter_set}" \
        --strength "${strength}" \
        --output "${output}"
}

# Seed 1: Neher model, both applicable parameter sets.
# for parameter_set in NM_FSIN_4Pr_v3 NM_FSIN_4Pr_v9; do
#     for strength in 0.0 0.1 0.3; do
#         run_score 1 Neher "${parameter_set}" "${strength}"
#     done
# done
for parameter_set in NM_FSIN_4Pr_v3; do
    for strength in 0.0 0.3; do
        run_score 1 Neher "${parameter_set}" "${strength}"
    done
done

# # Seed 2: replacement-pool model. The HighPv parameter set is not applicable.
# for strength in 0.0 0.1 0.3; do
#     run_score 2 RP MZM_FSIN_4Prot_v11b "${strength}"
# done
#done