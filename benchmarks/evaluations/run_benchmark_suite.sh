#!/usr/bin/env bash
# Run full benchmark suite for multiple models and prompt styles.
#
# Usage:
#   ./benchmarks/evaluations/run_benchmark_suite.sh workflow-random
#   ./benchmarks/evaluations/run_benchmark_suite.sh natural workflow-random workflow-conditional
#   ./benchmarks/evaluations/run_benchmark_suite.sh full natural workflow workflow-random
#
# Prompt styles are passed as positional arguments.
# Models, seeds, and samples are configured below.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODELS=(
    "openai:gpt-5-mini"
    "google_genai:gemini-3-flash-preview"
    "ollama:qwen3:4b-instruct-2507-q8_0"
)
SEEDS="1 2 3"
SAMPLES=5
PROBLEM="beams2d"
RAG_STATUS="no_rag"
# ─────────────────────────────────────────────────────────────────────────────

# Prompt styles from CLI args
if [ $# -eq 0 ]; then
    echo "Usage: $0 <prompt-style> [<prompt-style> ...]"
    echo "  e.g. $0 natural workflow-random workflow-conditional"
    exit 1
fi
PROMPT_STYLES=("$@")

echo "============================================================"
echo "Benchmark Suite"
echo "  Problem:       ${PROBLEM}"
echo "  Models:        ${MODELS[*]}"
echo "  Prompt styles: ${PROMPT_STYLES[*]}"
echo "  Seeds:         ${SEEDS}"
echo "  Samples:       ${SAMPLES}"
echo "  RAG status:    ${RAG_STATUS}"
echo "============================================================"

for STYLE in "${PROMPT_STYLES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  PROMPT STYLE: ${STYLE}"
    echo "============================================================"

    FIRST_MODEL=true

    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "  Model: ${MODEL}  |  Style: ${STYLE}"
        echo "------------------------------------------------------------"

        # Step 1: Run evaluation (generate prompts only for the first model)
        EXTRA_ARGS=""
        if [ "${FIRST_MODEL}" = true ]; then
            FIRST_MODEL=false
        else
            EXTRA_ARGS="--skip-prompt-generation"
        fi

        python benchmarks/evaluations/run_full_benchmark.py \
            --problem "${PROBLEM}" \
            --seeds ${SEEDS} \
            --samples "${SAMPLES}" \
            --prompt-style "${STYLE}" \
            --model "${MODEL}" \
            --agent-only \
            ${EXTRA_ARGS}

        # Step 2: Extract design data from Weave to JSON
        python benchmarks/evaluations/extract_data.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --rag-status "${RAG_STATUS}" \
            --model "${MODEL}" \
            --limit 500

        # Step 3: Compute global metrics
        python benchmarks/evaluations/compute_global_metrics.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --rag-status "${RAG_STATUS}" \
            --model "${MODEL}"

    done

    # Step 4: Generate plots (scans all models for this style)
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
