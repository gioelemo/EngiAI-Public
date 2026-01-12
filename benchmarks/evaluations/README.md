# Agent Evaluation Framework

This directory contains the unified evaluation infrastructure for benchmarking the engineering agent across different problem types and LLM models.

## Overview

The evaluation framework uses [Weave](https://wandb.ai/site/weave) to track and compare agent performance across:
- Multiple problem types (currently beams2d, with more planned)
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

- `--scorers legacy` - Use generic scorer (for backward compatibility)
- `--scorers generic` - Use generic scorer that works for all problems
- `--scorers engibench` - Compute global MMD metric only
- `--scorers all` - Use both generic scorer and global MMD metric

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--problem` | Problem type to evaluate | `beams2d` |
| `--model` | LLM model name | From config |
| `--samples` | Number of samples to evaluate | `5` |
| `--temperature` | Model temperature | From config |
| `--split` | Dataset split (train/val/test) | `test` |
| `--scorers` | Scorer set (legacy/generic/engibench/all) | `legacy` |

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

- **MMD** (Maximum Mean Discrepancy) - Measures how similar the distribution of generated designs is to the dataset distribution
- Uses all generated designs vs. full ground truth dataset
- Lower MMD = better match to dataset distribution

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
   - Include `scorers.py` with problem-specific scoring functions
   - Add `generate_prompts.py` and `validate_prompts.py`
   - Create data structure: `data/{generated,validated,raw}`

2. **Add problem configuration to `evaluate_agent.py`:**

```python
from benchmarks.problems.your_problem.scorers import score_your_metric

PROBLEM_CONFIGS = {
    "beams2d": {...},
    "your_problem": {
        "dataset_name": "huggingface/dataset-name",
        "prompt_file": "{problem}_prompts_50_samples_{split}.json",
        "scorers": [
            score_your_metric,  # Problem-specific scorer
        ],
    },
}
```

3. **Update the `--problem` choices** in the argument parser

## Custom Scorers

To add custom scorers for specific problem types:

1. **Create scorer in problem directory** (`benchmarks/problems/{problem}/scorers.py`)
2. **Use the `@weave.op()` decorator** for Weave tracking
3. **Follow the scorer signature:** `(output, target, metadata) -> dict`
4. **Import and add to `PROBLEM_CONFIGS`** in `evaluate_agent.py`

Example scorer:
```python
import weave
from typing import Any

@weave.op()
def score_custom_metric(
    output: dict[str, Any],
    target: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score custom metric for your problem type."""
    # Extract data from output
    result_value = output.get("result")
    target_value = target.get("expected")

    # Compute score
    score = compute_similarity(result_value, target_value)

    return {
        "score": score,
        "result_value": result_value,
        "target_value": target_value,
    }
```

See [../problems/beams2d/scorers.py](../problems/beams2d/scorers.py) for a complete example.

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
