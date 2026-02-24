#!/usr/bin/env bash
# Run HPC training benchmark suite for multiple models and prompt styles.
#
# Usage:
#   ./benchmarks/evaluations/run_hpc_benchmark_suite.sh hpc-train-cgan hpc-train-diff
#   ./benchmarks/evaluations/run_hpc_benchmark_suite.sh hpc-train-natural-cgan hpc-train-natural-diff
#
# Algorithm-specific prompt styles ensure data stays separate per algorithm:
#   hpc-train-cgan / hpc-train-natural-cgan     -> cgan_cnn_2d
#   hpc-train-diff / hpc-train-natural-diff      -> diffusion_2d_cond
#
# Pipeline per (style, model):
#   1. Generate prompts (10 seeds x 100 epochs, algorithm from style)
#   2. Run agent evaluation via evaluate_agent.py (repeated RUNS for variance)
#   3. Extract data from Weave via extract_data.py
#   4. Compute HPC metrics via compute_hpc_metrics.py
#   5. Generate plots via run_all.py
#
# Note: HPC prompts are 10 seeds at a fixed 100 epochs, per algorithm.
# Multiple RUNS give statistical variance from LLM non-determinism.
# run_full_benchmark.py auto-detects this and passes --run instead of --seed
# to evaluate_agent.py, so Weave traces are named "run_N" (not "seed_N").

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODELS=(
    "google_genai:gemini-3-flash-preview"
    "openai:gpt-5-mini"
)
RUNS="1"
SAMPLES=10
PROBLEM="hpc_train_beams2d"
RAG_STATUS="no_rag"
# ─────────────────────────────────────────────────────────────────────────────

# Prompt styles from CLI args (each encodes the algorithm)
if [ $# -eq 0 ]; then
    echo "Usage: $0 <prompt-style> [<prompt-style> ...]"
    echo "  e.g. $0 hpc-train-cgan hpc-train-diff"
    echo "  e.g. $0 hpc-train-natural-cgan hpc-train-natural-diff"
    exit 1
fi
PROMPT_STYLES=("$@")

echo "============================================================"
echo "HPC Training Benchmark Suite"
echo "  Problem:       ${PROBLEM}"
echo "  Models:        ${MODELS[*]}"
echo "  Prompt styles: ${PROMPT_STYLES[*]}"
echo "  Runs:          ${RUNS}"
echo "  Samples:       ${SAMPLES}"
echo "  RAG status:    ${RAG_STATUS}"
echo "============================================================"

for STYLE in "${PROMPT_STYLES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  PROMPT STYLE: ${STYLE}"
    echo "============================================================"

    # Step 1: Generate prompts (algorithm is encoded in the style)
    echo ""
    echo "------------------------------------------------------------"
    echo "  Generating prompts: style=${STYLE}"
    echo "------------------------------------------------------------"
    python benchmarks/problems/hpc_train_beams2d/generate_prompts.py \
        --style "${STYLE}"

    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "  Model: ${MODEL}  |  Style: ${STYLE}"
        echo "------------------------------------------------------------"

        # Step 2: Run agent evaluation (--seeds passed to run_full_benchmark.py
        # which auto-converts to --run for HPC problems)
        python benchmarks/evaluations/run_full_benchmark.py \
            --problem "${PROBLEM}" \
            --seeds ${RUNS} \
            --samples "${SAMPLES}" \
            --prompt-style "${STYLE}" \
            --model "${MODEL}" \
            --skip-prompt-generation

        # Step 3: Extract data from Weave
        python benchmarks/evaluations/extract_data.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --rag-status "${RAG_STATUS}" \
            --model "${MODEL}" \
            --limit 10000

        # Step 4: Compute HPC metrics (agent vs baseline)
        python benchmarks/evaluations/compute_hpc_metrics.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --rag-status "${RAG_STATUS}" \
            --model "${MODEL}"

    done

    # Step 5: Generate plots (scans all models for this style)
    echo ""
    echo "------------------------------------------------------------"
    echo "  Generating plots: style=${STYLE}"
    echo "------------------------------------------------------------"
    python benchmarks/evaluations/plots/run_all.py \
        --problem "${PROBLEM}" \
        --prompt-style "${STYLE}" \
        --rag-status "${RAG_STATUS}"

done

echo ""
echo "============================================================"
echo "  All done!"
echo "============================================================"
