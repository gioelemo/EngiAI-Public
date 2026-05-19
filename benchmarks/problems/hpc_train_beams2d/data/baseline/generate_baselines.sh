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
#   ./benchmarks/problems/hpc_train_beams2d/data/baseline/generate_baselines.sh --parallel 4
#   ./benchmarks/problems/hpc_train_beams2d/data/baseline/generate_baselines.sh --parallel 6
#
# Prerequisites:
#   - engiopt package installed (pip install -e .)
#   - WandB credentials configured (wandb login)
#   - Internet access to download model weights from WandB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# ── Parse arguments ──────────────────────────────────────────────────────────
MAX_PARALLEL=1  # Default: sequential (safe)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel|-j)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--parallel N]"
            exit 1
            ;;
    esac
done

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
echo "  Max parallel: ${MAX_PARALLEL}"
echo "  Output dir:   ${SCRIPT_DIR}"
echo "============================================================"

cd "${PROJECT_ROOT}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Track PIDs and their job names using parallel indexed arrays (bash 3.2 compat)
PID_LIST=()
NAME_LIST=()
FAILED=()
SUCCEEDED=0
TOTAL=$(( ${#ALGORITHMS[@]} * ${#SEEDS[@]} ))

# Reap finished jobs from PID_LIST, updating SUCCEEDED/FAILED
reap_finished() {
    local new_pids=()
    local new_names=()
    local i=0
    while [ $i -lt ${#PID_LIST[@]} ]; do
        local pid="${PID_LIST[$i]}"
        local name="${NAME_LIST[$i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            if wait "$pid"; then
                echo "  [DONE] ${name}"
                SUCCEEDED=$((SUCCEEDED + 1))
            else
                echo "  [FAIL] ${name}"
                FAILED+=("${name}")
            fi
        else
            new_pids+=("$pid")
            new_names+=("$name")
        fi
        i=$((i + 1))
    done
    PID_LIST=("${new_pids[@]+"${new_pids[@]}"}")
    NAME_LIST=("${new_names[@]+"${new_names[@]}"}")
}

# Wait until fewer than MAX_PARALLEL jobs are running
wait_for_slot() {
    while [ ${#PID_LIST[@]} -ge ${MAX_PARALLEL} ]; do
        sleep 1
        reap_finished
    done
}

# Wait for all remaining jobs to finish
wait_all() {
    while [ ${#PID_LIST[@]} -gt 0 ]; do
        sleep 1
        reap_finished
    done
}

LAUNCHED=0
for ALGO in "${ALGORITHMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUTPUT_CSV="${SCRIPT_DIR}/${ALGO}_seed${SEED}_metrics.csv"
        LOG_FILE="${LOG_DIR}/${ALGO}_seed${SEED}.log"
        JOB_NAME="${ALGO}_seed${SEED}"

        # Remove existing file to avoid append duplicates
        rm -f "${OUTPUT_CSV}"

        # Wait for a free slot
        wait_for_slot

        LAUNCHED=$((LAUNCHED + 1))
        echo "[${LAUNCHED}/${TOTAL}] Launching ${JOB_NAME} (log: ${LOG_FILE})"

        # Launch in background
        python -m "engiopt.${ALGO}.evaluate_${ALGO}" \
            --problem-id "${PROBLEM_ID}" \
            --seed "${SEED}" \
            --n-samples "${N_SAMPLES}" \
            --sigma "${SIGMA}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-entity "${WANDB_ENTITY}" \
            --output-csv "${OUTPUT_CSV}" \
            > "${LOG_FILE}" 2>&1 &

        PID_LIST+=($!)
        NAME_LIST+=("${JOB_NAME}")
    done
done

# Wait for remaining jobs
echo ""
echo "All jobs launched. Waiting for remaining ${#PID_LIST[@]} to finish..."
wait_all

echo ""
echo "============================================================"
echo "  Baseline Generation Complete"
echo "  Succeeded: ${SUCCEEDED}/${TOTAL}"
echo "  Failed:    ${#FAILED[@]}/${TOTAL}"
echo "============================================================"

# List generated files
echo ""
echo "Generated files:"
ls -la "${SCRIPT_DIR}"/*.csv 2>/dev/null || echo "  (none)"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: The following baselines failed:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f} (see ${LOG_DIR}/${f}.log)"
    done
    echo ""
    echo "This may happen if official models are not yet available"
    echo "on WandB for some algorithm/seed combinations."
    exit 1
fi

echo ""
echo "All baselines generated successfully!"
