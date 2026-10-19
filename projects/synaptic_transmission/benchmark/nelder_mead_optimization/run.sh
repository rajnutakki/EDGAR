#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCHMARK_DIR}/../../../.." && pwd)"
SCORER="${BENCHMARK_DIR}/optimize_seeds_BM_params.py"

cd "${REPO_ROOT}"

run_score() {
    local seed="$1"
    local model="$2"
    local parameter_set="$3"
    local strength="$4"

    echo "Scoring ${model}, ${parameter_set}, perturbation ${strength}"
    uv run python "${SCORER}" \
        --seed "${seed}" \
        --parameter-set "${parameter_set}" \
        --strength "${strength}"
}


for parameter_set in NM_FSIN_4Pr_v9; do
    for strength in 0.3; do
        run_score 1 Neher "${parameter_set}" "${strength}"
    done
done

for parameter_set in MZM_FSIN_4Prot_v11b; do
    for strength in 0.3; do
        run_score 2 RP "${parameter_set}" "${strength}"
    done
done

