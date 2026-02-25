"""
Generate RAG-dependent evaluation prompts for the rag_beams2d problem.

Unlike other problem prompt generators, these prompts are handcrafted and do NOT
sample from a HuggingFace dataset. Each prompt requires the agent to look up
factual information (e.g., from the EngiBench paper) before choosing design
parameters — testing whether RAG access improves parameter accuracy.

The --samples and --seed CLI arguments are accepted for compatibility with
run_full_benchmark.py but are otherwise ignored (prompts are fixed).

Usage:
    python benchmarks/problems/rag_beams2d/generate_prompts.py --style rag-eval
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
    "rag-eval": {
        "description": "RAG-dependent: requires searching indexed documents",
        # WITH RAG the optimal sequence includes searching first
        "optimal_tool_calls": [
            {"name": "search_documents", "count": 1},
            {"name": "optimize_design", "count": 1},
            {"name": "render_design", "count": 1},
        ],
        "optimal_call_count": 3,
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
# expected_volfrac is taken from the EngiBench paper's API example, which shows
# problem.conditions as (("volfrac", 0.35), ("forcedist", 0.0), ...).
# expected_volfrac_tolerance is the ±window considered correct (accounts for
# reasonable rounding by the LLM).
# NOTE: The scorer gates volfrac_accuracy credit on rag_tool_called=True.
# Even if the agent reaches the correct volfrac via get_problem_details instead
# of the paper, it earns no parameter accuracy credit.
# ---------------------------------------------------------------------------

RAG_PROMPTS: list[dict] = [
    {
        # -------------------------------------------------------------------
        # Prompt 0 — Default volume fraction from the EngiBench paper
        #
        # The EngiBench paper's API walkthrough (Section 3.1) shows the
        # Beams2D default conditions as (("volfrac", 0.35), ("forcedist", 0.0), ...).
        #
        # The expected answer (0.35) IS in the paper and IS also returned by
        # get_problem_details. The scorer distinguishes the two cases:
        # volfrac_accuracy only contributes to rag_benefit_score when the
        # agent actually called search_documents (rag_tool_called=True).
        #
        # RAG-on:  searches paper → finds 0.35 → score ~1.0
        # RAG-off: skips search → rag_called=False → eff_volfrac gated → score = 0.0
        # -------------------------------------------------------------------
        "prompt": (
            "The EngiBench paper documents the default design conditions for the "
            "Beams2D problem in its API walkthrough.\n\n"
            "Search the paper to find the default volume fraction (volfrac) listed "
            "for the Beams2D problem. Then generate a 2D beam design using exactly "
            "that volume fraction. Use default values for all other parameters (do NOT "
            "ask for clarification — proceed directly with defaults)."
        ),
        "conditions": {
            "expected_volfrac": 0.35,
            "expected_volfrac_tolerance": 0.05,
        },
        "metadata": {
            "knowledge_source": "EngiBench paper Section 3.1",
            "rag_query_hint": "EngiBench Beams2D default conditions volfrac",
            "parameter_tested": "volfrac",
            "expected_reasoning": (
                "Agent should search the EngiBench paper, find the API example showing "
                'problem.conditions = (("volfrac", 0.35), ...), and call '
                "optimize_design with volfrac=0.35. "
                "The scorer gates volfrac_accuracy on rag_tool_called=True, so agents "
                "that skip search and use get_problem_details do not earn parameter credit."
            ),
        },
        "target": {},
    },
    {
        # -------------------------------------------------------------------
        # Prompt 1 — Both parameters from the paper's API example (harder)
        #
        # The EngiBench paper's API walkthrough (Section 3.1) shows a custom
        # design configuration with volfrac=0.7 and forcedist=0.3.
        #
        # BOTH values differ from the problem defaults (volfrac=0.35,
        # forcedist=0.0), so the agent cannot cheat via get_problem_details.
        # The agent must find the specific code block in the paper and extract
        # both values correctly.
        #
        # Scoring uses two-parameter mode (dynamic weights):
        #   effective_volfrac_accuracy   0.40 (gated on rag_called)
        #   effective_forcedist_accuracy 0.40 (gated on rag_called)
        #   rag_tool_called              0.20
        #
        # RAG-on:  searches paper → finds both 0.7 and 0.3 → score ~1.0
        # RAG-off: calls get_problem_details → gets (0.35, 0.0) (both wrong)
        #          → parameter scores gated → rag_called=False → score = 0.0
        # -------------------------------------------------------------------
        "prompt": (
            "In the EngiBench paper's Section 3.1 API walkthrough, a code example "
            "runs a Beams2D optimization using non-default design conditions. "
            "Search the paper to find both the volume fraction and force distribution "
            "from that example, then generate a 2D beam design with those exact "
            "values. Use default values for all other parameters and do not ask for "
            "clarification."
        ),
        "conditions": {
            "expected_volfrac": 0.7,
            "expected_volfrac_tolerance": 0.05,
            "expected_forcedist": 0.3,
            "expected_forcedist_tolerance": 0.05,
        },
        "metadata": {
            "knowledge_source": "EngiBench paper Section 3.1",
            "rag_query_hint": "EngiBench API example desired_conds volfrac forcedist beams2d",
            "parameter_tested": "volfrac+forcedist",
            "expected_reasoning": (
                "Agent should search the EngiBench paper, find the API example showing "
                'desired_conds = {"volfrac": 0.7, "forcedist": 0.3}, and call '
                "optimize_design with volfrac=0.7 and forcedist=0.3. "
                "Both values differ from get_problem_details defaults (0.35, 0.0) so "
                "the agent cannot cheat. Scorer gates both parameter scores on "
                "rag_tool_called=True."
            ),
        },
        "target": {},
    },
    {
        # -------------------------------------------------------------------
        # Prompt 2 — Volume fraction + filter radius from the SOPTX paper's
        #            2D cantilever beam benchmark (post-cutoff source)
        #
        # He et al. (2025) "SOPTX: A High-Performance Multi-Backend Framework
        # for Topology Optimization" (arXiv:2505.02438) benchmarks their
        # framework on a 2D cantilever beam with:
        #   volfrac = 0.4, rmin = 6.0 (nelx=160, nely=100)
        #
        # BOTH values differ from the EngiBench Beams2D defaults
        # (volfrac=0.35, rmin≈3.5), so the agent cannot cheat via
        # get_problem_details. rmin=6.0 is an atypical value that LLMs
        # would not know from training data — the paper was submitted
        # May 2025, beyond all current model knowledge cutoffs.
        #
        # Scoring uses rmin two-parameter mode (dynamic weights):
        #   effective_volfrac_accuracy 0.40 (gated on rag_called)
        #   effective_rmin_accuracy    0.40 (gated on rag_called)
        #   rag_tool_called            0.20
        #
        # RAG-on:  searches paper → finds volfrac=0.4, rmin=6.0 → score ~1.0
        # RAG-off: skips search → parameters gated → rag_called=False → score = 0.0
        # -------------------------------------------------------------------
        "prompt": (
            "The SOPTX paper by He et al. (2025) benchmarks its topology optimization "
            "framework on a 2D cantilever beam problem.\n\n"
            "Search the paper to find both the volume fraction (volfrac) and the "
            "filter radius (rmin) used for that 2D cantilever benchmark. Then generate "
            "a 2D beam design using exactly those values. Use default values for all "
            "other parameters and do not ask for clarification."
        ),
        "conditions": {
            "expected_volfrac": 0.4,
            "expected_volfrac_tolerance": 0.05,
            "expected_rmin": 6.0,
            "expected_rmin_tolerance": 0.5,
        },
        "metadata": {
            "knowledge_source": "SOPTX paper (arXiv:2505.02438, He et al. 2025)",
            "rag_query_hint": "SOPTX 2D cantilever beam volume fraction filter radius benchmark",
            "parameter_tested": "volfrac+rmin",
            "expected_reasoning": (
                "Agent should search the SOPTX paper (arXiv:2505.02438), find the "
                "2D cantilever beam benchmark specifying volfrac=0.4 and rmin=6.0, "
                "and call optimize_design with those exact values. "
                "Both differ from EngiBench defaults and rmin=6.0 is atypical — "
                "no LLM would guess it from training data. "
                "Scorer gates both parameter scores on rag_tool_called=True."
            ),
        },
        "target": {},
    },
    {
        # -------------------------------------------------------------------
        # Prompt 3 — Mixed sources: three non-default parameters from two papers
        #
        # The agent must combine information from multiple RAG sources:
        #   volfrac = 0.7   (EngiBench paper API example, non-default)
        #   forcedist = 0.3 (EngiBench paper API example, non-default)
        #   rmin = 6.0      (SOPTX paper 2D cantilever benchmark, post-cutoff)
        #
        # This is the hardest prompt: the agent needs to query TWO papers
        # and ALL three values differ from get_problem_details defaults
        # (volfrac=0.35, forcedist=0.0, rmin≈3.5). Additionally rmin=6.0
        # is from a post-cutoff paper so LLMs cannot guess it.
        #
        # Scoring uses triple-parameter mode (dynamic weights):
        #   effective_volfrac_accuracy   0.30 (gated on rag_called)
        #   effective_forcedist_accuracy 0.30 (gated on rag_called)
        #   effective_rmin_accuracy      0.30 (gated on rag_called)
        #   rag_tool_called              0.10
        #
        # RAG-on:  searches both papers → finds all three → score ~1.0
        # RAG-off: no search → all params wrong → rag_called=False → score = 0.0
        # -------------------------------------------------------------------
        "prompt": (
            "Generate a 2D beam design combining parameters from multiple sources:\n\n"
            "1. Use the volume fraction and force distribution from the EngiBench "
            "paper's API walkthrough example (the non-default values shown in the "
            "code snippet).\n"
            "2. Use the filter radius from the SOPTX paper by He et al. (2025) for "
            "their 2D cantilever beam benchmark.\n\n"
            "Search the relevant papers to find each value, then generate a 2D beam "
            "design using exactly those three parameters. Use default values for all "
            "other parameters and do not ask for clarification."
        ),
        "conditions": {
            "expected_volfrac": 0.7,
            "expected_volfrac_tolerance": 0.05,
            "expected_forcedist": 0.3,
            "expected_forcedist_tolerance": 0.05,
            "expected_rmin": 6.0,
            "expected_rmin_tolerance": 0.5,
        },
        "metadata": {
            "knowledge_source": "EngiBench paper + SOPTX paper (arXiv:2505.02438)",
            "rag_query_hint": "EngiBench API example volfrac forcedist, SOPTX cantilever rmin",
            "parameter_tested": "volfrac+forcedist+rmin",
            "expected_reasoning": (
                "Agent should: (1) search EngiBench paper for API example showing "
                "desired_conds = {volfrac: 0.7, forcedist: 0.3}, "
                "(2) search SOPTX paper for 2D cantilever benchmark rmin=6.0. "
                "Then call optimize_design with volfrac=0.7, forcedist=0.3, rmin=6.0. "
                "All three differ from defaults so the agent cannot cheat. "
                "Scorer gates all three parameter scores on rag_tool_called=True."
            ),
        },
        # Two papers → two search calls needed
        "optimal_tool_calls": [
            {"name": "search_documents", "count": 2},
            {"name": "optimize_design", "count": 1},
            {"name": "render_design", "count": 1},
        ],
        "optimal_call_count": 4,
        "target": {},
    },
]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def create_rag_prompts(
    style: str = "rag-eval",
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
                    "problem_type": "rag_beams2d",
                    "prompt_style": style,
                    "prompt_style_description": style_config["description"],
                    "success_criteria": style_config["success_criteria"],
                },
                # Per-prompt overrides take precedence over style defaults
                "optimal_tool_calls": raw.get(
                    "optimal_tool_calls", style_config["optimal_tool_calls"]
                ),
                "optimal_call_count": raw.get(
                    "optimal_call_count", style_config["optimal_call_count"]
                ),
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
        default="rag-eval",
        choices=list(PROMPT_STYLES.keys()),
        help="Prompt style to generate (default: rag-eval)",
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
        output_dir / f"rag_beams2d_prompts_{n}_samples_{args.split}_{args.style}.json"
    )

    with output_file.open("w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {n} prompt(s).")
    print(f"Saved to: {output_file}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
