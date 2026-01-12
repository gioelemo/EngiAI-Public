# Photonics 2D Problem

This folder contains files for the **photonics2d** topology optimization problem.

## Problem Description

Photonics 2D is a topology optimization problem focused on designing optical structures that maximize the overlap between a target field distribution and the simulated field. The optimization balances:

- **Objective**: Maximize `total_overlap` (overlap integral between target and simulated fields)
- **Design variables**: Binary material distribution (0 = air, 1 = material)
- **Conditions** (parameters, not constraints):
  - `lambda1`: First eigenvalue/wavelength parameter
  - `lambda2`: Second eigenvalue/wavelength parameter
  - `blur_radius`: Gaussian blur radius for smoothing

## Dataset

- **HuggingFace**: `IDEALLab/photonics_2d` (or similar dataset name)
- **Splits**: train, val, test
- **Fields**:
  - `design`: 2D binary array (material distribution)
  - `total_overlap`: Target overlap value
  - `conditions`: Dict with `lambda1`, `lambda2`, `blur_radius`

## Files

- `generate_prompts.py`: Script to generate problem prompts for all dataset splits
- `README.md`: This file

## Usage

### Generate Prompts

```bash
python benchmarks/problems/photonics2d/generate_prompts.py
```

This will create prompt files in `data/datasets/photonics2d_prompts/`:
- `train_prompts.json`
- `val_prompts.json`
- `test_prompts.json`

### Evaluate with Generic Scorer

```bash
python benchmarks/evaluations/evaluate_agent.py \
    --problem photonics2d \
    --scorers generic \
    --samples 5
```

## Differences from Beams2D

1. **Objective**: `total_overlap` (maximize) instead of `compliance` (minimize)
2. **No constraints**: All fields (`lambda1`, `lambda2`, `blur_radius`) are parameters, not constraints
3. **Domain**: Optical field optimization instead of structural mechanics
4. **Evaluation**: Focuses on field overlap rather than structural performance

## Generic Scorer Support

Photonics2D is fully supported by the generic scorer (`score_design_generic`) via the problem registry in `benchmarks/shared/problem_registry.py`:

```python
"photonics2d": ProblemConfig(
    objectives=[
        ObjectiveConfig(
            name="total_overlap",
            field_name="total_overlap",
            target_field="total_overlap",
            direction="maximize",
            relative_error_threshold=0.2,
            aliases=["overlap", "field_overlap"],
        )
    ],
    conditions=[
        ConditionConfig("lambda1", "lambda1", "parameter"),
        ConditionConfig("lambda2", "lambda2", "parameter"),
        ConditionConfig("blur_radius", "blur_radius", "parameter"),
    ],
    dataset_name="IDEALLab/photonics_2d",
    design_field="design",
)
```

No custom scorer code needed!
