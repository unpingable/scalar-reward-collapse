#!/usr/bin/env bash
# Fire-and-forget sweep runner for bandit/AB experiments.
# Usage: ./scripts/run_bandit_sweep.sh [sweep_config.json] [output_dir]

set -euo pipefail

SWEEP_CONFIG="${1:-}"
OUTPUT_DIR="${2:-runs/sweep}"

echo "=== Scalar Reward Collapse — Bandit Sweep ==="
echo "Output: ${OUTPUT_DIR}"

if [ -n "${SWEEP_CONFIG}" ]; then
    echo "Config: ${SWEEP_CONFIG}"
    python -m scalar_collapse sweep --sweep-config "${SWEEP_CONFIG}" --output "${OUTPUT_DIR}"
else
    echo "Using default sweep config"
    python -m scalar_collapse sweep --output "${OUTPUT_DIR}"
fi

echo ""
echo "=== Sweep complete ==="
echo "Results in: ${OUTPUT_DIR}/sweep_summary.json"
