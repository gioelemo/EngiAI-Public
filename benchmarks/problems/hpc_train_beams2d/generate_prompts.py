"""
Generate HPC cGAN training evaluation prompts for the hpc_train_beams2d problem.

These prompts are handcrafted and do NOT sample from a HuggingFace dataset.
Each prompt instructs the agent to train a cGAN model on the Euler HPC cluster,
then download the model, generate designs, and simulate them.

The 3 prompts vary seed and epochs to test different training configurations:
  Prompt 0: seed=1, epochs=20  (~10-20min training)
  Prompt 1: seed=2, epochs=50  (~30-60min training)
  Prompt 2: seed=3, epochs=100 (~1-2h training)

Usage:
    python benchmarks/problems/hpc_train_beams2d/generate_prompts.py --style hpc-train
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
    "hpc-train": {
        "description": "HPC training workflow: train cGAN, generate designs, simulate",
        "optimal_tool_calls": [
            {"name": "generate_training_command", "count": 1},
            {"name": "submit_slurm_job", "count": 1},
            {"name": "monitor_job_until_complete", "count": 1},
            {"name": "download_wandb_model", "count": 1},
            {"name": "sample_designs_from_model", "count": 1},
            {"name": "simulate_design", "count": 5},
        ],
        "optimal_call_count": 10,
        "success_criteria": "hpc_workflow_completion",
    },
}

# ---------------------------------------------------------------------------
# Fixed design conditions for evaluation (same across all prompts)
# ---------------------------------------------------------------------------

DESIGN_CONDITIONS = [
    {"volfrac": 0.30, "rmin": 2.0, "forcedist": 0.0, "overhang_constraint": False},
    {"volfrac": 0.40, "rmin": 2.5, "forcedist": 0.2, "overhang_constraint": False},
    {"volfrac": 0.50, "rmin": 3.0, "forcedist": 0.5, "overhang_constraint": False},
    {"volfrac": 0.60, "rmin": 3.5, "forcedist": 0.8, "overhang_constraint": False},
    {"volfrac": 0.35, "rmin": 2.0, "forcedist": 0.0, "overhang_constraint": False},
]

# ---------------------------------------------------------------------------
# Training configurations (seed x epochs)
# ---------------------------------------------------------------------------

TRAINING_CONFIGS = [
    {"seed": 1, "epochs": 20},
    {"seed": 2, "epochs": 50},
    {"seed": 3, "epochs": 100},
]


def _build_conditions_text(conditions: list[dict]) -> str:
    """Format conditions list as prompt text."""
    lines = []
    for i, cond in enumerate(conditions):
        lines.append(
            f"     Design {i}: volfrac={cond['volfrac']:.2f}, "
            f"rmin={cond['rmin']:.1f}, "
            f"forcedist={cond['forcedist']:.1f}, "
            f"overhang_constraint={cond['overhang_constraint']}"
        )
    return "\n".join(lines)


def _build_prompt(seed: int, epochs: int) -> str:
    """Build the full prompt text for a training configuration."""
    conditions_text = _build_conditions_text(DESIGN_CONDITIONS)
    return (
        f"Train a cGAN CNN 2D generative model for the Beams2D topology optimization "
        f"problem on the Euler HPC cluster, then evaluate it by generating and "
        f"simulating designs.\n\n"
        f"Step 1: Generate Training Script\n"
        f"   - Use the generate_training_command tool with:\n"
        f"     algorithm: cgan_cnn_2d\n"
        f"     problem_id: beams2d\n"
        f"     epochs: {epochs}\n"
        f"     seed: {seed}\n\n"
        f"Step 2: Submit to HPC\n"
        f"   - Submit the generated SLURM script to the Euler cluster\n\n"
        f"Step 3: Monitor Training\n"
        f"   - Monitor the job until it completes\n"
        f"   - Use check_interval=120 and max_checks=200 for the monitoring\n\n"
        f"Step 4: Download Trained Model\n"
        f"   - Download the trained generator model from WandB\n"
        f"   - Use algorithm=cgan_cnn_2d, problem_id=beams2d, seed={seed}, "
        f"model_type=generator\n\n"
        f"Step 5: Generate Designs\n"
        f"   - Generate 5 designs using the downloaded model with these conditions:\n"
        f"{conditions_text}\n\n"
        f"Step 6: Simulate Each Design\n"
        f"   - Simulate each of the 5 generated designs to get compliance values\n"
        f"   - Report the compliance value for each design\n\n"
        f"Complete all steps in order. Do not ask for clarification."
    )


# ---------------------------------------------------------------------------
# Handcrafted prompts
# ---------------------------------------------------------------------------

HPC_TRAIN_PROMPTS: list[dict] = [
    {
        "prompt": _build_prompt(cfg["seed"], cfg["epochs"]),
        "conditions": {
            "seed": cfg["seed"],
            "epochs": cfg["epochs"],
            "algorithm": "cgan_cnn_2d",
            "problem_id": "beams2d",
            "design_conditions": DESIGN_CONDITIONS,
        },
        "metadata": {
            "training_config": {
                "seed": cfg["seed"],
                "epochs": cfg["epochs"],
                "algorithm": "cgan_cnn_2d",
                "problem_id": "beams2d",
            },
            "expected_workflow_steps": [
                "generate_training_command",
                "submit_slurm_job",
                "monitor_job_until_complete",
                "download_wandb_model",
                "sample_designs_from_model",
                "simulate_design",
            ],
            "n_design_conditions": len(DESIGN_CONDITIONS),
        },
        "target": {},
    }
    for cfg in TRAINING_CONFIGS
]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def create_hpc_train_prompts(
    style: str = "hpc-train",
    seed: int | None = None,  # noqa: ARG001 — kept for CLI compat
    samples: int | None = None,  # noqa: ARG001 — kept for CLI compat
) -> list[dict]:
    """Return the list of HPC training evaluation prompts."""
    if style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown prompt style '{style}'. Available: {list(PROMPT_STYLES.keys())}"
        )

    style_config = PROMPT_STYLES[style]
    prompts = []

    for i, raw in enumerate(HPC_TRAIN_PROMPTS):
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
        description="Generate HPC training evaluation prompts for hpc_train_beams2d"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="hpc-train",
        choices=list(PROMPT_STYLES.keys()),
        help="Prompt style to generate (default: hpc-train)",
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
    print("HPC_TRAIN_BEAMS2D PROMPT GENERATION")
    print("=" * 60)
    print()
    print(f"Style:  {args.style}")
    print(f"Seed:   {args.seed} (ignored — prompts are fixed)")
    print()

    prompts = create_hpc_train_prompts(style=args.style, seed=args.seed)
    n = len(prompts)

    output_dir = Path(__file__).parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"hpc_train_beams2d_prompts_{n}_samples_{args.split}_{args.style}.json"
    )

    with output_file.open("w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {n} prompt(s):")
    for i, p in enumerate(prompts):
        cfg = p["conditions"]
        print(f"  Prompt {i}: seed={cfg['seed']}, epochs={cfg['epochs']}")
    print(f"\nSaved to: {output_file}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
