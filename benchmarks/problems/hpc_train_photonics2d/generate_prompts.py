"""
Generate HPC training evaluation prompts for the hpc_train_photonics2d problem.

These prompts are handcrafted and do NOT sample from a HuggingFace dataset.
Each prompt instructs the agent to train a generative model on the Euler HPC
cluster, then evaluate it using the standard EngiOpt evaluation script.

The prompts form a grid of 10 seeds for a single algorithm (fixed at 100 epochs
to match the official baselines). The algorithm is encoded in the prompt style:

  hpc-train-cgan      -> cgan_cnn_2d
  hpc-train-diff      -> diffusion_2d_cond

  Seeds:      1..10
  Epochs:     100 (fixed — matches official baselines)

Usage:
    python benchmarks/problems/hpc_train_photonics2d/generate_prompts.py --style hpc-train-cgan
    python benchmarks/problems/hpc_train_photonics2d/generate_prompts.py --style hpc-train-diff
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Algorithm constants
# ---------------------------------------------------------------------------

ALGORITHMS = ["cgan_cnn_2d", "diffusion_2d_cond"]

ALGORITHM_DISPLAY_NAMES: dict[str, str] = {
    "cgan_cnn_2d": "cGAN CNN 2D",
    "diffusion_2d_cond": "conditional diffusion 2D",
}

ALGO_SHORT: dict[str, str] = {"cgan_cnn_2d": "cgan", "diffusion_2d_cond": "diff"}

SEEDS = list(range(1, 11))  # 1..10
EPOCHS = 100  # Fixed — matches official baselines

# ---------------------------------------------------------------------------
# Prompt style configuration
# ---------------------------------------------------------------------------

_BASE_TOOL_CALLS = [
    {"name": "generate_training_command", "count": 1},
    {"name": "submit_slurm_job", "count": 1},
    {"name": "monitor_job_until_complete", "count": 1},
    {"name": "evaluate_model", "count": 1},
]

PROMPT_STYLES: dict[str, dict] = {
    "hpc-train-cgan": {
        "description": "HPC training workflow: train cGAN on photonics2d, evaluate with EngiOpt metrics",
        "algorithm": "cgan_cnn_2d",
        "natural": False,
        "optimal_tool_calls": _BASE_TOOL_CALLS,
        "optimal_call_count": 4,
        "success_criteria": "hpc_workflow_completion",
    },
    "hpc-train-diff": {
        "description": "HPC training workflow: train diffusion on photonics2d, evaluate with EngiOpt metrics",
        "algorithm": "diffusion_2d_cond",
        "natural": False,
        "optimal_tool_calls": _BASE_TOOL_CALLS,
        "optimal_call_count": 4,
        "success_criteria": "hpc_workflow_completion",
    },
    "hpc-train-natural-cgan": {
        "description": "Natural language HPC training (cGAN on photonics2d) — no tool names or explicit steps",
        "algorithm": "cgan_cnn_2d",
        "natural": True,
        "optimal_tool_calls": _BASE_TOOL_CALLS,
        "optimal_call_count": 4,
        "success_criteria": "hpc_workflow_completion",
    },
    "hpc-train-natural-diff": {
        "description": "Natural language HPC training (diffusion on photonics2d) — no tool names or explicit steps",
        "algorithm": "diffusion_2d_cond",
        "natural": True,
        "optimal_tool_calls": _BASE_TOOL_CALLS,
        "optimal_call_count": 4,
        "success_criteria": "hpc_workflow_completion",
    },
}

# ---------------------------------------------------------------------------
# Training configurations (seed x algorithm, fixed epochs=100)
# ---------------------------------------------------------------------------


class TrainingConfig(TypedDict):
    seed: int
    epochs: int
    algorithm: str


def get_training_configs(algorithm: str) -> list[TrainingConfig]:
    """Build training configs for the given algorithm (10 seeds, fixed epochs)."""
    return [{"seed": seed, "epochs": EPOCHS, "algorithm": algorithm} for seed in SEEDS]


# Default configs used by plotting and other importers.
TRAINING_CONFIGS: list[TrainingConfig] = get_training_configs(ALGORITHMS[0])


def _build_prompt(seed: int, epochs: int, algorithm: str) -> str:
    """Build the explicit step-by-step prompt (hpc-train style)."""
    display_name = ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)
    return (
        f"Train a {display_name} generative model for the Photonics2D inverse design "
        f"problem on the Euler HPC cluster, then evaluate it against the dataset "
        f"baseline using the standard EngiOpt evaluation script.\n\n"
        f"Step 1: Generate Training Script\n"
        f"   - Use the generate_training_command tool with:\n"
        f"     algorithm: {algorithm}\n"
        f"     problem_id: photonics2d\n"
        f"     epochs: {epochs}\n"
        f"     seed: {seed}\n\n"
        f"Step 2: Submit to HPC\n"
        f"   - Submit the generated SLURM script to the Euler cluster\n\n"
        f"Step 3: Monitor Training\n"
        f"   - Monitor the job until it completes\n"
        f"   - Use check_interval=30 and max_checks=200 for the monitoring\n\n"
        f"Step 4: Evaluate Trained Model\n"
        f"   - Use the evaluate_model tool to evaluate the trained model\n"
        f"     against the dataset baseline:\n"
        f"     problem_id: photonics2d\n"
        f"     algorithm: {algorithm}\n"
        f"     seed: {seed}\n"
        f"     n_samples: 50\n"
        f"   - This downloads the model from WandB, generates designs, and\n"
        f"     computes metrics (IOG, COG, FOG, MMD, DPP, violation rate)\n"
        f"   - Report the evaluation metrics from the output\n\n"
        f"Complete all steps in order. Do not ask for clarification."
    )


def _build_natural_prompt(seed: int, epochs: int, algorithm: str) -> str:
    """Build a natural-language prompt (hpc-train-natural style).

    No tool names, no step numbers, no monitoring/evaluation parameters.
    The agent must infer the full workflow from context.
    """
    display_name = ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)
    return (
        f"Train a {display_name} model for the Photonics2D inverse design problem "
        f"on the Euler HPC cluster with seed {seed} and {epochs} epochs. "
        f"Use the available tools to generate the SLURM training script -- do not "
        f"write or modify any scripts manually. "
        f"Submit the job and wait for it to finish. Then use the model evaluation "
        f"tool to evaluate the trained model against the dataset -- it will "
        f"download the model from WandB automatically. Report the metrics.\n\n"
        f"Do not ask for clarification."
    )


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def create_hpc_train_prompts(
    style: str = "hpc-train-cgan",
    seed: int | None = None,  # noqa: ARG001 — kept for CLI compat
    samples: int | None = None,  # noqa: ARG001 — kept for CLI compat
) -> list[dict]:
    """Return the list of HPC training evaluation prompts.

    The algorithm is determined by the style (e.g. hpc-train-cgan -> cgan_cnn_2d).
    """
    if style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown prompt style '{style}'. Available: {list(PROMPT_STYLES.keys())}"
        )

    style_config = PROMPT_STYLES[style]
    algorithm = style_config["algorithm"]
    is_natural = style_config["natural"]
    configs = get_training_configs(algorithm)
    prompts = []

    for i, cfg in enumerate(configs):
        if is_natural:
            prompt_text = _build_natural_prompt(
                cfg["seed"], cfg["epochs"], cfg["algorithm"]
            )
        else:
            prompt_text = _build_prompt(cfg["seed"], cfg["epochs"], cfg["algorithm"])

        prompts.append(
            {
                "prompt": prompt_text,
                "prompt_style": style,
                "conditions": {
                    "seed": cfg["seed"],
                    "epochs": cfg["epochs"],
                    "algorithm": cfg["algorithm"],
                    "problem_id": "photonics2d",
                },
                "metadata": {
                    "problem_type": "hpc_train_photonics2d",
                    "training_config": {
                        "seed": cfg["seed"],
                        "epochs": cfg["epochs"],
                        "algorithm": cfg["algorithm"],
                        "problem_id": "photonics2d",
                    },
                    "prompt_style": style,
                    "prompt_style_description": style_config["description"],
                    "success_criteria": style_config["success_criteria"],
                    "expected_workflow_steps": [
                        "generate_training_command",
                        "submit_slurm_job",
                        "monitor_job_until_complete",
                        "evaluate_model",
                    ],
                },
                "optimal_tool_calls": style_config["optimal_tool_calls"],
                "optimal_call_count": style_config["optimal_call_count"],
                "target": {},
                "example_id": i,
                "dataset_split": "test",
            }
        )

    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HPC training evaluation prompts for hpc_train_photonics2d"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="hpc-train-cgan",
        choices=list(PROMPT_STYLES.keys()),
        help="Prompt style (encodes algorithm). Default: hpc-train-cgan",
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

    style_config = PROMPT_STYLES[args.style]
    algorithm = style_config["algorithm"]

    print("=" * 60)
    print("HPC_TRAIN_PHOTONICS2D PROMPT GENERATION")
    print("=" * 60)
    print()
    print(f"Style:     {args.style}")
    print(f"Algorithm: {algorithm}")
    print(f"Seeds:     {SEEDS}")
    print(f"Epochs:    {EPOCHS} (fixed)")
    print()

    prompts = create_hpc_train_prompts(style=args.style, seed=args.seed)
    n = len(prompts)

    output_dir = Path(__file__).parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"hpc_train_photonics2d_prompts_{n}_samples_{args.split}_{args.style}.json"
    )

    with output_file.open("w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {n} prompt(s):")
    for i, p in enumerate(prompts):
        cfg = p["conditions"]
        print(
            f"  Prompt {i}: seed={cfg['seed']}, epochs={cfg['epochs']}, algo={cfg['algorithm']}"
        )
    print(f"\nSaved to: {output_file}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
