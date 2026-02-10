"""
Generate RAG-dependent evaluation prompts for the rag_beams2d problem.

Unlike other problem prompt generators, these prompts are handcrafted and do NOT
sample from a HuggingFace dataset. Each prompt requires the agent to look up
factual information (e.g., from the EngiBench paper) before choosing design
parameters — testing whether RAG access improves parameter accuracy.

The --samples and --seed CLI arguments are accepted for compatibility with
run_full_benchmark.py but are otherwise ignored (prompts are fixed).

Usage:
    python benchmarks/problems/rag_beams2d/generate_prompts.py --style rag
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Prompt style configuration
# ---------------------------------------------------------------------------

PROMPT_STYLES: dict[str, dict] = {
    "rag": {
        "description": "RAG-dependent: requires searching indexed documents",
        # WITH RAG the optimal sequence includes searching first
        "optimal_tool_calls": [
            {"name": "search_documents", "count": 1},
            {"name": "optimize_design", "count": 1},
            {"name": "simulate_design", "count": 1},
            {"name": "render_design", "count": 1},
        ],
        "optimal_call_count": 4,
        "success_criteria": "rag_parameter_accuracy",
    },
}

# ---------------------------------------------------------------------------
# Handcrafted prompts with known expected answers
# ---------------------------------------------------------------------------
# Each entry defines:
#   prompt            - The text sent to the agent
#   conditions        - Expected parameter values (for scoring)
#   metadata          - Context for scorers
#   target            - Ground truth (empty here — no HF design to compare)
#
# expected_volfrac is the midpoint of the range stated in the EngiBench paper.
# expected_volfrac_tolerance is the ±window within which the agent's choice is
# considered correct (accounts for reasonable rounding by the LLM).
# ---------------------------------------------------------------------------

RAG_PROMPTS: list[dict] = [
    {
        # -------------------------------------------------------------------
        # Prompt 0 — Volume fraction range from the EngiBench paper
        # The beams_2d_50_100_v0 dataset spans volfrac in [0.2, 0.5].
        # Midpoint = 0.35.
        # -------------------------------------------------------------------
        "prompt": (
            "According to the EngiBench paper, what are the typical volume fraction "
            "ranges used in their beams2d experiments?\n\n"
            "Based on what you find in the paper, generate a 2D beam design using the "
            "middle of that range. Use default values for all other parameters."
        ),
        "conditions": {
            "expected_volfrac": 0.35,
            "volfrac_range_low": 0.2,
            "volfrac_range_high": 0.5,
            "expected_volfrac_tolerance": 0.05,
        },
        "metadata": {
            "knowledge_source": "EngiBench paper",
            "rag_query_hint": "beams2d volume fraction range",
            "parameter_tested": "volfrac",
            "expected_reasoning": (
                "Agent should search for EngiBench paper, find volfrac range 0.2-0.5, "
                "compute midpoint 0.35, and call optimize_design with volfrac=0.35."
            ),
        },
        "target": {},
    },
]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def create_rag_prompts(
    style: str = "rag",
    seed: int | None = None,  # noqa: ARG001 — kept for CLI compat
    samples: int | None = None,  # noqa: ARG001 — kept for CLI compat
) -> list[dict]:
    """Return the list of RAG evaluation prompts in evaluation-ready format."""
    if style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown prompt style '{style}'. Available: {list(PROMPT_STYLES.keys())}"
        )

    style_config = PROMPT_STYLES[style]
    prompts = []

    for i, raw in enumerate(RAG_PROMPTS):
        prompts.append(
            {
                "prompt": raw["prompt"],
                "prompt_style": style,
                "conditions": raw["conditions"],
                "metadata": {
                    **raw["metadata"],
                    "prompt_style": style,
                    "prompt_style_description": style_config["description"],
                    "success_criteria": style_config["success_criteria"],
                },
                "optimal_tool_calls": style_config["optimal_tool_calls"],
                "optimal_call_count": style_config["optimal_call_count"],
                "target": raw["target"],
                "example_id": i,
                "dataset_split": "test",
            }
        )

    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RAG evaluation prompts for rag_beams2d"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="rag",
        choices=list(PROMPT_STYLES.keys()),
        help="Prompt style to generate (default: rag)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed (ignored — prompts are fixed; kept for CLI compatibility)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples (ignored — all prompts are always generated)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split label used in the output filename (default: test)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG_BEAMS2D PROMPT GENERATION")
    print("=" * 60)
    print()
    print(f"Style:  {args.style}")
    print(f"Seed:   {args.seed} (ignored — prompts are fixed)")
    print()

    prompts = create_rag_prompts(style=args.style, seed=args.seed)
    n = len(prompts)

    output_dir = Path(__file__).parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"rag_beams2d_prompts_{n}_samples_{args.split}_{args.style}_seed{args.seed}.json"
    )

    with output_file.open("w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {n} prompt(s).")
    print(f"Saved to: {output_file}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
