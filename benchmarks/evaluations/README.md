# Agent Evaluation Framework

This directory contains the unified evaluation infrastructure for benchmarking the engineering agent across different problem types and LLM models.

## Overview

The evaluation framework uses [Weave](https://wandb.ai/site/weave) to track and compare agent performance across:
- Multiple problem types (beams2d, photonics2d, thermoelastic2d)
- Multiple LLM models (GPT-4o, Claude, etc.)
- Different model configurations (temperature, etc.)

## Quick Start

### Basic Evaluation

Evaluate the agent on beams2d with default settings:

```bash
conda activate engineer-assistant
cd benchmarks/evaluations
python evaluate_agent.py --problem beams2d --samples 5
```

### Compare Multiple Models

Evaluate different models on the same problem:

```bash
# Evaluate GPT-4o
python evaluate_agent.py --problem beams2d --model gpt-4o --samples 10

# Evaluate Claude Sonnet
python evaluate_agent.py --problem beams2d --model claude-3-5-sonnet-20241022 --samples 10
```

### Adjust Parameters

```bash
python evaluate_agent.py \
  --problem beams2d \
  --model gpt-4o \
  --samples 20 \
  --temperature 0.5 \
  --split test \
  --scorers all
```

### Scorer Options

- `--scorers generic` - Compute per-design metrics only (default)
- `--scorers engibench` - Compute per-design metrics + global metrics (MMD, DPP, RVC, optimality gaps)
- `--scorers all` - Same as engibench

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--problem` | Problem type to evaluate | `beams2d` |
| `--model` | LLM model name | From config |
| `--samples` | Number of samples to evaluate | `5` |
| `--temperature` | Model temperature | From config |
| `--split` | Dataset split (train/val/test) | `test` |
| `--scorers` | Scorer set (generic/engibench/all) | `generic` |

## Results Organization

Results are automatically organized by model and problem type:

```
results/
├── {model-name}/
│   └── {problem-type}/
│       └── comparisons/
│           ├── comparison_example_0.png
│           ├── comparison_example_1.png
│           └── ...
```

Example:
```
results/
├── gpt-4o/
│   └── beams2d/
│       └── comparisons/
└── claude-3-5-sonnet-20241022/
    └── beams2d/
        └── comparisons/
```

## Evaluation Metrics

### Generic Scorer

All problems now use the generic scorer (`score_design_generic`) which provides:

- **Design Quality Metrics**
  - IoU (Intersection over Union) - Topology overlap
  - Pixel accuracy - Element-wise match
  - MSE (Mean Squared Error) - Density field error
- **Constraint Checking** - Based on problem configuration
  - Volume fraction constraints (beams2d, photonics2d)
- **Objective Evaluation** - Based on problem configuration
  - Compliance scoring (beams2d)
  - Total overlap scoring (photonics2d)

The generic scorer is configuration-driven via `benchmarks.shared.problem_registry`, which defines objectives, constraints, and weights for each problem type.

See [../problems/beams2d/SCORING_METRICS.md](../problems/beams2d/SCORING_METRICS.md) for detailed beams2d metric definitions.

### EngiBench Global Metrics

Global metrics computed after evaluation completes (use `--scorers engibench` or `--scorers all`):

- **MMD** (Maximum Mean Discrepancy) - Measures how similar the distribution of generated designs is to the dataset distribution (lower is better)
- **DPP Diversity** - Design variability using Determinantal Point Process kernel (higher is better)
- **RVC** (Ratio of Violated Constraints) - Fraction of designs violating at least one constraint (lower is better, 0-1 range)
- **IOG** (Initial Optimality Gap) - Average gap between initial design objective and optimal objective
- **COG** (Cumulative Optimality Gap) - Average gap accumulated across all optimization iterations
- **FOG** (Final Optimality Gap) - Average gap between final design objective and optimal objective

These metrics use all generated designs vs. the full ground truth dataset.

## Weave Integration

All evaluations are tracked in Weave for easy comparison and visualization:

1. Results are automatically uploaded to Weave dashboard
2. View metrics, traces, and comparisons in the web UI
3. Compare performance across models and problems
4. Track improvements over time

Configure Weave in your `.env`:
```bash
USE_WEAVE=true
WEAVE_PROJECT="your-entity/your-project"
```

## Adding New Problem Types

To add support for a new problem type:

1. **Create the problem directory:** `benchmarks/problems/{problem_name}/`
   - Add `generate_prompts.py` to create evaluation prompts from the HuggingFace dataset
   - Create data structure: `data/generated/` for generated prompts
   - Optionally add `validate_prompts.py` for validation (see beams2d example)

2. **Add problem configuration to the central registry** (`benchmarks/shared/problem_registry.py`):

```python
from benchmarks.shared.problem_config import (
    ConditionConfig,
    ObjectiveConfig,
    ProblemConfig,
)

PROBLEMS = {
    "your_problem": ProblemConfig(
        name="your_problem",
        dataset_name="IDEALLab/your_dataset_name",
        design_field="optimal_design",
        tool_name="optimize_design",
        objectives=[
            ObjectiveConfig(
                name="your_objective",
                field_name="final_objective_value",
                target_field="objective_value",
                direction="minimize",  # or "maximize"
                relative_error_threshold=0.2,
                aliases=["obj", "objective"],
            )
        ],
        conditions=[
            ConditionConfig(
                name="your_parameter",
                field_name="parameter_name",
                constraint_type="equality",  # or "none" for non-constraint parameters
                tolerance=0.01,
            )
        ],
        design_metrics_weights={
            "iou": 0.4,
            "pixel_accuracy": 0.25,
            "constraint_match": 0.15,
            "objective_match": 0.2,
        },
        prompt_file_template="your_problem_prompts_50_samples_{split}.json",
    ),
}
```

3. **That's it!** The problem is now automatically available:
   - `evaluate_agent.py` will automatically include it (built from registry)
   - The generic scorer will use your problem configuration
   - No code changes needed in evaluation scripts

See `benchmarks/shared/problem_registry.py` for complete examples of beams2d, photonics2d, and thermoelastic2d.

## Troubleshooting

### "File not found" errors
- Ensure you've run `generate_prompts.py` for the problem type first
- Check that the prompt file name matches the configuration

### Import errors
- Activate the conda environment: `conda activate engineer-assistant`
- Ensure project root is accessible (scripts handle this automatically)

### Weave errors
- Check that `USE_WEAVE=true` in `.env`
- Verify `WEAVE_PROJECT` is set correctly
- Ensure you have the `weave` package installed

### Design extraction failures
- Check that `src/tools/engibench.py` includes `optimized_design` field in return dict
- Verify the agent is calling `optimize_design` with a `config` parameter
- Review debug output for parsing errors

## Performance Considerations

- Evaluations run in parallel for speed
- Each example gets a unique thread_id to avoid state sharing
- Matplotlib uses Agg backend to prevent threading issues
- Results are cached locally to avoid re-computation
- **No Docker containers required**: Evaluation automatically skips Prusa MCP and MMORE services
  - `SKIP_MCP=true` - Skips Prusa MCP server connection
  - `SKIP_MMORE=true` - Skips MMORE Docker container requirement
  - This allows running benchmarks without any external services
