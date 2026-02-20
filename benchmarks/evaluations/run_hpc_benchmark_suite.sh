#!/usr/bin/env bash
# Run HPC training benchmark suite for multiple models and prompt styles.
#
# Usage:
#   ./benchmarks/evaluations/run_hpc_benchmark_suite.sh hpc-train
#   ./benchmarks/evaluations/run_hpc_benchmark_suite.sh hpc-train hpc-train-natural
#
# Prompt styles are passed as positional arguments.
# Models are configured below.
#
# Pipeline per (style, model):
#   1. Generate prompts (once per style)
#   2. Run agent evaluation via evaluate_agent.py
#   3. Extract data from Weave via extract_data.py
#   4. Compute HPC metrics via compute_hpc_metrics.py
#   5. Generate plots via run_all.py

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODELS=(
    "openai:gpt-5-mini"
    "google_genai:gemini-3-flash-preview"
)
PROBLEM="hpc_train_beams2d"
RAG_STATUS="no_rag"
# HPC prompts are fixed (3 configs) — seed/samples are ignored by
# generate_prompts.py but required by evaluate_agent.py CLI.
SEED=0
SAMPLES=3
# ─────────────────────────────────────────────────────────────────────────────

# Prompt styles from CLI args
if [ $# -eq 0 ]; then
    echo "Usage: $0 <prompt-style> [<prompt-style> ...]"
    echo "  e.g. $0 hpc-train hpc-train-natural"
    exit 1
fi
PROMPT_STYLES=("$@")

echo "============================================================"
echo "HPC Training Benchmark Suite"
echo "  Problem:       ${PROBLEM}"
echo "  Models:        ${MODELS[*]}"
echo "  Prompt styles: ${PROMPT_STYLES[*]}"
echo "  RAG status:    ${RAG_STATUS}"
echo "============================================================"

for STYLE in "${PROMPT_STYLES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  PROMPT STYLE: ${STYLE}"
    echo "============================================================"

    # Step 1: Generate prompts (once per style — prompts are model-independent)
    echo ""
    echo "------------------------------------------------------------"
    echo "  Generating prompts for style: ${STYLE}"
    echo "------------------------------------------------------------"
    python benchmarks/problems/hpc_train_beams2d/generate_prompts.py \
        --style "${STYLE}"

    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "  Model: ${MODEL}  |  Style: ${STYLE}"
        echo "------------------------------------------------------------"

        # Step 2: Run agent evaluation
        python benchmarks/evaluations/run_full_benchmark.py \
            --problem "${PROBLEM}" \
            --seeds ${SEED} \
            --samples "${SAMPLES}" \
            --prompt-style "${STYLE}" \
            --model "${MODEL}" \
            --agent-only \
            --skip-prompt-generation

        # Step 3: Extract data from Weave
        python benchmarks/evaluations/extract_data.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --rag-status "${RAG_STATUS}" \
            --model "${MODEL}" \
            --limit 2000

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
    echo "  Generating plots for style: ${STYLE}"
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
