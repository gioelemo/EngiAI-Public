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
conda activate engineer-assistant
cd benchmarks/problems/beams2d
python generate_prompts.py
```

**Output:** `data/generated/beam_prompts_N_samples.json`

**What it does:**
- Loads beam design examples from HuggingFace
- Converts technical parameters to natural language
- Creates evaluation prompts with ground truth targets
- Publishes to Weave for tracking (if enabled)

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

### `explore_dataset.py`

Interactive exploration of the beam dataset to understand the data distribution and characteristics.

**Usage:**
```bash
python explore_dataset.py
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
python evaluate_agent.py --problem beams2d --model gpt-4o --samples 5
```

See [benchmarks/evaluations/README.md](../../evaluations/README.md) for more details on evaluation.

## Adding New Problem Types

To add a new problem type (e.g., thermoelastic2d):

1. Create a new directory: `benchmarks/problems/thermoelastic2d/`
2. Copy and adapt the scripts from this directory
3. Create the `data/` subdirectory structure
4. Add problem configuration to `benchmarks/evaluations/evaluate_agent.py`
5. Create a README documenting the problem and dataset
