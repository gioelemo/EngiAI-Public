#!/usr/bin/env bash
# Run RAG benchmark suite for multiple models.
#
# Evaluates each model twice (RAG on via MMORE, RAG off) to measure how much
# document retrieval improves parameter accuracy.
#
# Usage:
#   ./benchmarks/evaluations/run_rag_benchmark_suite.sh
#
# Prerequisites:
#   - MMORE eval service running on port 8001 (make mmore-eval-up)
#   - Indexed papers:
#       file_id='engibench_paper'  (EngiBench paper)
#       file_id='soptx_paper'     (SOPTX paper, arXiv:2505.02438)
#
# Pipeline:
#   1. Generate prompts (fixed, 4 handcrafted prompts)
#   2. For each model × run: evaluate with --mmore (RAG on) and --no-mmore (RAG off)
#   3. Extract data for both RAG statuses
#   4. Generate plots (scans all models)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MODELS=(
    "openai:gpt-5-mini"
    "google_genai:gemini-3-flash-preview"
)
RUNS=(1 2 3)
PROBLEM="rag_beams2d"
STYLE="rag-eval"
MMORE_RAG_URL="${MMORE_RAG_URL:-http://localhost:8001}"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "RAG Benchmark Suite"
echo "  Problem:       ${PROBLEM}"
echo "  Models:        ${MODELS[*]}"
echo "  Prompt style:  ${STYLE}"
echo "  Runs:          ${RUNS[*]}"
echo "  MMORE URL:     ${MMORE_RAG_URL}"
echo "============================================================"

# Step 1: Generate prompts (fixed set, only need to do once)
echo ""
echo "------------------------------------------------------------"
echo "  Generating prompts: style=${STYLE}"
echo "------------------------------------------------------------"
python benchmarks/problems/rag_beams2d/generate_prompts.py --style "${STYLE}"

# Step 2: Run evaluations per model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Model: ${MODEL}"
    echo "============================================================"

    # RAG ON
    for RUN in "${RUNS[@]}"; do
        echo ""
        echo "  ${MODEL} — RAG ON — run ${RUN}"
        echo "------------------------------------------------------------"
        MMORE_RAG_URL="${MMORE_RAG_URL}" \
        python benchmarks/evaluations/evaluate_agent.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --model "${MODEL}" \
            --run "${RUN}" \
            --mmore
    done

    # RAG OFF
    for RUN in "${RUNS[@]}"; do
        echo ""
        echo "  ${MODEL} — RAG OFF — run ${RUN}"
        echo "------------------------------------------------------------"
        MMORE_RAG_URL="${MMORE_RAG_URL}" \
        python benchmarks/evaluations/evaluate_agent.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --model "${MODEL}" \
            --run "${RUN}" \
            --no-mmore
    done

done

# Step 3: Extract data per model and RAG status
echo ""
echo "------------------------------------------------------------"
echo "  Extracting data"
echo "------------------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    for RAG_STATUS in "rag" "no_rag"; do
        echo "  Extracting: ${MODEL} — ${RAG_STATUS}"
        python benchmarks/evaluations/extract_data.py \
            --problem "${PROBLEM}" \
            --prompt-style "${STYLE}" \
            --rag-status "${RAG_STATUS}" \
            --model "${MODEL}" \
            --limit 2000
    done
done

# Step 4: Generate plots (scans all models, both RAG statuses)
echo ""
echo "------------------------------------------------------------"
echo "  Generating plots"
echo "------------------------------------------------------------"
python benchmarks/evaluations/plots/run_all.py \
    --problem "${PROBLEM}" \
    --prompt-style "${STYLE}"

echo ""
echo "============================================================"
echo "  All done!"
echo "============================================================"
