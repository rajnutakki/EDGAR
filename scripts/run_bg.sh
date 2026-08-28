#!/usr/bin/env bash
# Run an EDGAR project in the background, logging output to ~/projects/<name>/run_output/out.log

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <project_name>" >&2
    exit 1
fi

PROJECT_NAME="$1"
CONFIG="projects/${PROJECT_NAME}/config.yaml"
OUT_DIR="${HOME}/projects/${PROJECT_NAME}/run_output"
LOG="${OUT_DIR}/out.log"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config not found at ${CONFIG}" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

nohup uv run edgar run "$CONFIG" > "$LOG" 2>&1 &

echo "Started PID $! — logging to ${LOG}"
