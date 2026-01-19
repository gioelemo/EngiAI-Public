# Tool Usage Analysis

This module provides tools to extract and visualize agent tool usage patterns from Weave evaluation traces.

## Overview

The tool usage analysis helps you understand:
- Which tools are most frequently used by agents
- How tool usage varies across different models (GPT-4.1 vs GPT-5.1)
- Correlation between tool usage and performance metrics
- Tool usage patterns across different problems (beams2d vs photonics2d)

## Workflow

### 1. Extract Tool Usage from Weave

After running evaluations with `evaluate_agent.py`, extract tool usage statistics from Weave traces:

```bash
python benchmarks/evaluations/extract_tool_usage.py \
  --project YOUR_WEAVE_PROJECT \
  --model openai:gpt-5.1 \
  --problem beams2d \
  --seed 1
```

**Arguments:**
- `--project`: Your Weave project name (e.g., `entity/project`)
- `--model`: Model identifier (e.g., `openai:gpt-5.1`, `openai:gpt-4.1`)
- `--problem`: Problem type (`beams2d` or `photonics2d`)
- `--seed`: Random seed used in evaluation (optional)
- `--output`: Custom output path (optional, defaults to results directory)
- `--limit`: Maximum traces to fetch (default: 100)

This will create a CSV file at:
```
benchmarks/evaluations/results/{model}/{problem}/tool_usage.csv
```

### 2. Generate Visualizations

Once tool usage data is extracted, generate visualizations:

```bash
# Generate only tool usage plots
python benchmarks/evaluations/plots/plot_tool_usage.py

# Or generate all plots including tool usage
python benchmarks/evaluations/plots/run_all.py
```

## Generated Visualizations

### 1. Tool Usage Frequency
**File:** `tool_usage_frequency.pdf`

Horizontal bar chart showing total usage count for each tool across all models and examples.

### 2. Tool Usage by Model
**File:** `tool_usage_by_model.pdf`

Side-by-side comparison of:
- Average total tool calls per example
- Average unique tools used per example

Grouped by model and problem type, with error bars showing standard deviation.

### 3. Tool Usage Heatmap
**File:** `tool_usage_heatmap.pdf`

Heatmap showing which tools are used by which models, with color intensity representing usage frequency.

### 4. Tool Usage vs Performance
**File:** `tool_usage_vs_performance.pdf`

Four scatter plots correlating tool usage with performance:
- Total tool calls vs Overall score
- Unique tools vs Overall score
- Total tool calls vs IoU
- Unique tools vs IoU

Includes trend lines and correlation coefficients.

## Data Format

The tool usage CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `example_id` | Example identifier |
| `model_id` | Model name (e.g., `openai:gpt-5.1`) |
| `problem_id` | Problem type (e.g., `beams2d`) |
| `seed` | Random seed used |
| `total_tools` | Total number of tool calls |
| `unique_tools` | Number of unique tools used |
| `tool_{name}` | Count for specific tool (e.g., `tool_optimize_design`) |

## Example Usage

### Extract data for multiple models

```bash
# GPT-5.1
python benchmarks/evaluations/extract_tool_usage.py \
  --project my-org/my-project \
  --model openai:gpt-5.1 \
  --problem beams2d

# GPT-4.1
python benchmarks/evaluations/extract_tool_usage.py \
  --project my-org/my-project \
  --model openai:gpt-4.1 \
  --problem beams2d
```

### Generate all visualizations

```bash
python benchmarks/evaluations/plots/run_all.py
```

This will generate 10 visualizations total (6 quality metrics + 4 tool usage).

## Integration with Existing Plots

Tool usage visualizations are fully integrated with the existing plotting infrastructure:

- Use the same styling configuration (`PLOT_STYLE`)
- Output to the same directory (`benchmarks/evaluations/plots/figures/`)
- Can correlate tool usage with performance metrics
- Included in `run_all.py` for batch generation

## Troubleshooting

### No tool usage data found

If you see this error:
```
❌ No tool usage data found. Please run extract_tool_usage.py first.
```

Make sure you've:
1. Run evaluations with `evaluate_agent.py`
2. Extracted tool usage with `extract_tool_usage.py`
3. Used the correct project name and model identifiers

### Missing correlation plots

If tool usage vs performance plots are skipped:
```
⚠️  No design metrics found, skipping correlation plots
```

This means design-level metrics aren't available. Run quality metric computations first:
```bash
python benchmarks/evaluations/compute_output_quality_design_stats.py
```

## Notes

- Tool usage extraction requires access to Weave traces
- The `tool_calls_info` field must be present in evaluation outputs
- Tool usage is tracked per example, not per seed/run
- Multiple runs with the same example_id will show separate rows

## See Also

- [Evaluation Guide](../README.md) - Overall evaluation workflow
- [Plot Generation](README.md) - Other visualization options
- [Weave Documentation](https://docs.wandb.ai/weave/) - Weave tracing system
