#!/usr/bin/env bash
# Run workflow benchmark suite for multiple models and prompt styles.
#
# Usage:
#   ./benchmarks/evaluations/run_workflow_benchmark_suite.sh workflow-random
#   ./benchmarks/evaluations/run_workflow_benchmark_suite.sh --problem photonics2d workflow-random workflow-conditional
#   ./benchmarks/evaluations/run_workflow_benchmark_suite.sh full natural workflow-random workflow-conditional
#
# Options:
#   --problem <name>  Problem to evaluate (default: beams2d). E.g. beams2d, photonics2d
#
# Prompt styles are passed as positional arguments (after optional flags).
# Models, seeds, and samples are configured below.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODELS=(
    "openai:gpt-5-mini"
    "google_genai:gemini-3-flash-preview"
    "ollama:qwen3:4b-instruct-2507-q8_0"
    "ollama:qwen3.5:4b-q8_0"

)
SEEDS="1 2 3"
SAMPLES=5
PROBLEM="beams2d"
RAG_STATUS="no_rag"
# ─────────────────────────────────────────────────────────────────────────────

# Parse optional --problem flag
while [[ $# -gt 0 && "$1" == --* ]]; do
    case "$1" in
        --problem)
            PROBLEM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Prompt styles from remaining CLI args
if [ $# -eq 0 ]; then
    echo "Usage: $0 [--problem <name>] <prompt-style> [<prompt-style> ...]"
    echo "  e.g. $0 natural workflow-random workflow-conditional"
    echo "  e.g. $0 --problem photonics2d workflow-random workflow-conditional"
    exit 1
fi
PROMPT_STYLES=("$@")

echo "============================================================"
echo "Workflow Benchmark Suite"
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
            ${EXTRA_ARGS}

        # Step 2: Extract design data from Weave to JSON
        python benchmarks/evaluations/extract_data.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --rag-status "${RAG_STATUS}" \
            --model "${MODEL}" \
            --limit 10000

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
