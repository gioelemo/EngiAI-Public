# Beams 2D Problem Benchmarks

This directory contains problem-specific scripts and data for evaluating the engineering agent on 2D beam topology optimization tasks.

## Dataset

Uses the [IDEALLab/beams_2d_50_100_v0](https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0) dataset, which contains optimal 2D beam designs with varying:
- Volume fractions (material constraints)
- Minimum feature sizes (rmin)
- Load distributions (forcedist)
- Target compliance values

## Scripts

### `generate_prompts.py`

Transforms beam design examples from the HuggingFace dataset into natural language prompts for agent evaluation.

**Usage:**
```bash
conda activate engiai
cd benchmarks/problems/beams2d
python generate_prompts.py --samples 5 --style full
```

Run `python generate_prompts.py --help` for the full list of options,
including the supported `--style` values.

**Output:** `data/generated/beams2d_prompts_{samples}_samples_{split}_{style}_seed{seed}.json`
(file-name pattern defined in [benchmarks/shared/problem_registry.py](../../shared/problem_registry.py))

**What it does:**
- Loads beam design examples from the HuggingFace dataset
- Converts technical parameters to natural language
- Creates evaluation prompts with ground truth targets
- Saves the prompts locally as JSON under `data/generated/`

### `validate_prompts.py`

Validates that generated prompts are well-formed and complete.

**Usage:**
```bash
python validate_prompts.py
```

**Output:** `data/validated/validation_report.json`

**Checks:**
- Required fields present
- Parameter ranges valid
- Ground truth data complete
- Prompt quality and clarity

### Dataset Exploration

Interactive exploration of the beam dataset to understand the data distribution and characteristics.

**Usage:**
```bash
# Using problem name (recommended)
python -m benchmarks.shared.explore_dataset --problem beams2d

# Show more examples
python -m benchmarks.shared.explore_dataset --problem beams2d --num-samples 5
```

## Data Directory

```
data/
├── generated/       # Generated prompt datasets
├── validated/       # Validation reports
└── raw/            # Raw data samples (preserved for reference)
```

## Evaluation

To evaluate the agent on this problem, use the unified evaluation script:

```bash
cd benchmarks/evaluations
python evaluate_agent.py --problem beams2d --model openai:gpt-4.1 --samples 5
```

See [benchmarks/evaluations/README.md](../../evaluations/README.md) for more details on evaluation.

## Adding New Problem Types

Problems are registered in a single source of truth:
[benchmarks/shared/problem_registry.py](../../shared/problem_registry.py).
Adding an entry to the `PROBLEMS` dict auto-exposes the new problem in every
evaluation CLI (`--problem <name>`) — no manual wiring in `evaluate_agent.py`
is required.

To add a new problem type (e.g., a hypothetical `newproblem`):

1. Create a new directory: `benchmarks/problems/newproblem/`
2. Copy and adapt `generate_prompts.py` from this directory
3. Create the `data/` subdirectory structure
4. Register the problem by adding a `ProblemConfig` entry to the `PROBLEMS`
   dict in [benchmarks/shared/problem_registry.py](../../shared/problem_registry.py)
5. Create a README documenting the problem and dataset
