# Agent Evaluation Framework

This directory contains the unified evaluation infrastructure for benchmarking the engineering agent across different problem types and LLM models.

## Overview

The evaluation framework uses [Weave](https://wandb.ai/site/weave) to track and compare agent performance across:
- Multiple problem types (beams2d, thermoelastic2d, etc.)
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
  --temperature 0.5
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--problem` | Problem type to evaluate | `beams2d` |
| `--model` | LLM model name | From config |
| `--samples` | Number of samples to evaluate | `5` |
| `--temperature` | Model temperature | From config |

## Results Organization

Results are automatically organized by model and problem type:

```
results/
├── gpt-4o/
│   ├── beams2d/
│   │   └── comparisons/
│   │       ├── comparison_example_0.png
│   │       ├── comparison_example_1.png
│   │       └── ...
│   └── thermoelastic2d/
│       └── comparisons/
└── claude-3-5-sonnet-20241022/
    ├── beams2d/
    └── thermoelastic2d/
```

## Evaluation Metrics

### Qualitative Scorers

- **`score_constraint_accuracy`** - Does agent mention all constraint values?
- **`score_target_awareness`** - Does agent reference target compliance?
- **`score_understands_tradeoffs`** - Does agent understand material/performance tradeoffs?
- **`score_provides_actionable_guidance`** - Does agent give concrete steps?
- **`score_no_contradictions`** - Does agent avoid incorrect statements?

### Quantitative Scorers

- **`score_design_match`** - How similar is the agent's design to ground truth?
  - IoU (Intersection over Union)
  - Pixel accuracy
  - MSE (Mean Squared Error)
  - Volume fraction error

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

1. Create the problem directory: `benchmarks/problems/{problem_name}/`
2. Add problem configuration to `evaluate_agent.py`:

```python
PROBLEM_CONFIGS = {
    "beams2d": {...},
    "your_problem": {
        "dataset_name": "huggingface/dataset-name",
        "prompt_file": "your_prompts.json",
        "scorers": [
            score_constraint_accuracy,
            score_design_match,
            # Add problem-specific scorers
        ],
    },
}
```

3. Update the `--problem` choices in the argument parser

## Custom Scorers

To add custom scorers for specific problem types:

1. Create scorer functions in `benchmarks/shared/scorers.py`
2. Use the `@weave.op()` decorator
3. Follow the signature: `(prompt, conditions, output, target, metadata) -> dict`
4. Add to problem configuration in `PROBLEM_CONFIGS`

Example:
```python
@weave.op()
def score_custom_metric(
    prompt: str,
    conditions: dict[str, Any],
    output: dict[str, Any],
    target: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    # Your scoring logic
    return {"score": 0.85, "details": "..."}
```

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
