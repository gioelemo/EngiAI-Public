#!/usr/bin/env bash
# Generate baseline metric CSVs for the HPC training benchmark.
#
# These baselines are from the OFFICIAL pre-trained EngiOpt models (100 epochs)
# downloaded from WandB. They serve as the reference for agent vs baseline
# comparison plots.
#
# Algorithms:
#   - cgan_cnn_2d:       official 100-epoch cGAN model (baselines for seeds 1-10)
#   - diffusion_2d_cond: official 100-epoch diffusion model (baselines for seeds 1-10)
#
# Note: diffusion artifacts live under the "engibench" WandB entity and need
# --wandb-entity engibench to resolve correctly.
#
# Usage:
#   ./benchmarks/problems/hpc_train_beams2d/data/baseline/generate_baselines.sh
#
# Prerequisites:
#   - engiopt package installed (pip install -e .)
#   - WandB credentials configured (wandb login)
#   - Internet access to download model weights from WandB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# ── Configuration ────────────────────────────────────────────────────────────
SEEDS=(1 2 3 4 5 6 7 8 9 10)
ALGORITHMS=("cgan_cnn_2d" "diffusion_2d_cond")
PROBLEM_ID="beams2d"
N_SAMPLES=50
SIGMA=10.0
WANDB_PROJECT="engiopt"
WANDB_ENTITY="engibench"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "HPC Training Baseline Generation"
echo "============================================================"
echo "  Seeds:        ${SEEDS[*]}"
echo "  Algorithms:   ${ALGORITHMS[*]}"
echo "  Problem:      ${PROBLEM_ID}"
echo "  N samples:    ${N_SAMPLES}"
echo "  Sigma:        ${SIGMA}"
echo "  WandB entity: ${WANDB_ENTITY}"
echo "  Output dir:   ${SCRIPT_DIR}"
echo "============================================================"

cd "${PROJECT_ROOT}"

FAILED=()

for ALGO in "${ALGORITHMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUTPUT_CSV="${SCRIPT_DIR}/${ALGO}_seed${SEED}_metrics.csv"

        echo ""
        echo "------------------------------------------------------------"
        echo "  Algorithm: ${ALGO}  |  Seed: ${SEED}"
        echo "  Output:    ${OUTPUT_CSV}"
        echo "------------------------------------------------------------"

        # Remove existing file to avoid append duplicates
        rm -f "${OUTPUT_CSV}"

        if python -m "engiopt.${ALGO}.evaluate_${ALGO}" \
            --problem-id "${PROBLEM_ID}" \
            --seed "${SEED}" \
            --n-samples "${N_SAMPLES}" \
            --sigma "${SIGMA}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-entity "${WANDB_ENTITY}" \
            --output-csv "${OUTPUT_CSV}"; then
            echo "  -> OK: ${OUTPUT_CSV}"
        else
            echo "  -> FAILED: ${ALGO} seed=${SEED}"
            FAILED+=("${ALGO}_seed${SEED}")
        fi
    done
done

echo ""
echo "============================================================"
echo "  Baseline Generation Complete"
echo "============================================================"

# List generated files
echo ""
echo "Generated files:"
ls -la "${SCRIPT_DIR}"/*.csv 2>/dev/null || echo "  (none)"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: The following baselines failed:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}"
    done
    echo ""
    echo "This may happen if official models are not yet available"
    echo "on WandB for some algorithm/seed combinations."
    exit 1
fi

echo ""
echo "All baselines generated successfully!"
