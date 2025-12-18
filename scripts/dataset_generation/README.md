# Beam Prompt Dataset Generation

This directory contains scripts for generating benchmark datasets for Weave evaluation using beam design data from HuggingFace.

## Dataset Source

**HuggingFace Dataset**: [IDEALLab/beams_2d_50_100_v0](https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0)
- **Rows**: ~3.88k
- **Features**: optimal_design, volfrac, rmin, forcedist, overhang_constraint, c, optimization_history

## Workflow

### 1. Exploration (Current)
```bash
python scripts/dataset_generation/explore_beams_dataset.py
```

Explores the dataset structure and saves sample data to `data/datasets/beam_prompts/raw/`.

### 2. Generation (TODO)
Generate prompts from beam conditions.

### 3. Validation (TODO)
Validate generated prompts match the beam conditions.

### 4. Upload to Weave (TODO)
Publish validated dataset to Weave for benchmarking.

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
