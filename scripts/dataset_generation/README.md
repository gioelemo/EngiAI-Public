# Beam Prompt Dataset Generation

This directory contains scripts for generating benchmark datasets for Weave evaluation using beam design data from HuggingFace.

## Dataset Source

**HuggingFace Dataset**: [IDEALLab/beams_2d_50_100_v0](https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0)
- **Rows**: ~3.88k
- **Features**: optimal_design, volfrac, rmin, forcedist, overhang_constraint, c, optimization_history

## Workflow

### 1. Exploration ✅
```bash
python scripts/dataset_generation/explore_beams_dataset.py
```

Explores the dataset structure and saves sample data to `data/datasets/beam_prompts/raw/`.

### 2. Prompt Generation ✅
```bash
python scripts/dataset_generation/generate_beam_prompts.py
```

Generates natural language prompts from beam conditions:
- Transforms numerical parameters into readable descriptions
- Creates structured prompts for benchmarking
- Saves locally to `data/datasets/beam_prompts/generated/`
- Publishes to Weave for tracking and evaluation

**Output Example:**
```
Design a 2D beam structure with the following constraints:
- Volume fraction: 23.8% (use only 23.8% of available material)
- Minimum feature size (rmin): 3.5
- Load condition: a uniformly distributed force
- Target: Minimize compliance (maximize stiffness)

The beam should be designed on a 50x100 grid.
```

### 3. Validation ✅
```bash
python scripts/dataset_generation/validate_prompts.py
```

Validates that generated prompts are correct and consistent:
- **Numerical accuracy**: Prompt text matches numerical conditions
- **Parameter ranges**: Values are within expected bounds
- **Completeness**: All required fields are present
- **Consistency**: Descriptions align with parameter values

Saves validation report to `data/datasets/beam_prompts/validated/validation_report.json`

**Current Results:** 100% validation success (50/50 prompts passed all checks)

### 4. Evaluation (TODO)
Run engineering agent on prompts and compare outputs to targets.

## Directory Structure

```
data/datasets/beam_prompts/
├── raw/          # Downloaded HF data samples
├── generated/    # Generated prompts
└── validated/    # Validated datasets ready for Weave
```

## Requirements

```bash
pip install datasets  # HuggingFace datasets library
```

Already available in the project environment.
