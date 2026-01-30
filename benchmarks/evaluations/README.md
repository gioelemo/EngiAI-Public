# Agent Evaluation Framework

This directory contains the unified evaluation infrastructure for benchmarking the engineering agent across different problem types and LLM models.

## Overview

The evaluation framework uses [Weave](https://wandb.ai/site/weave) to track and compare agent performance across:
- Multiple problem types (beams2d, photonics2d, thermoelastic2d)
- Multiple LLM models (GPT-4o, Gemini, etc.)
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
| `--scorers` | Scorer set (generic/engibench/all/task_completion/tool_use) | `generic` |
| `--seed` | Random seed for optimization | `None` |
| `--output-csv` | Custom CSV output path | Auto-generated |
| `--prompt-style` | Prompt style (full/approximate/natural/workflow) | `full` |
| `--mmore` | Enable MMORE RAG system | Disabled |
| `--no-mmore` | Disable MMORE RAG system (default) | - |

## Results Organization

Results are automatically organized by model and problem type:

```
results/
├── {model-name}/
│   └── {problem-type}/
│       ├── metrics.csv           # Global metrics (MMD, DPP, RVC, IOG, COG, FOG) across seeds
│       ├── design_metrics.csv    # Per-design metrics (IoU, accuracy, etc.) for each example
│       └── comparisons/          # Comparison visualizations
│           ├── seed_1/           # Per-seed comparisons (if using seeds)
│           │   ├── comparison_example_0.png
│           │   └── ...
│           ├── seed_2/
│           └── ...
```

Example:
```
results/
├── openai_gpt-4.1/
│   └── beams2d/
│       ├── metrics.csv
│       └── comparisons/
│           ├── seed_1/
│           ├── seed_2/
│           └── seed_3/
```

## Seed-Based Evaluation

For reproducible benchmarking and statistical analysis (matching EngiOpt paper methodology):

### Single Seed Evaluation

```bash
python evaluate_agent.py \
  --problem beams2d \
  --samples 50 \
  --scorers engibench \
  --seed 1
```

Output:
- Metrics saved to: `results/models/{model}/beams2d/{prompt_style}/{rag_status}/metrics.csv`
- Comparisons saved to: `results/models/{model}/beams2d/{prompt_style}/{rag_status}/comparisons/seed_1/`

### Multiple Seeds for Statistics

Run with multiple seeds to collect statistical data:

```bash
# Run 10 seeds (EngiOpt paper methodology)
for seed in {1..10}; do
  python evaluate_agent.py \
    --problem beams2d \
    --samples 50 \
    --scorers engibench \
    --seed $seed
done
```

Each run appends a row to the CSV file with the seed value tracked.

### Computing Statistics

After running multiple seeds, compute mean ± std:

```bash
python compute_metrics_stats.py results/openai_gpt-4.1/beams2d/metrics.csv
```

Output example:
```
============================================================
Metrics Statistics for results/openai_gpt-4.1/beams2d/metrics.csv
============================================================

Number of runs: 10

COG : 1.399069e+08 ± 1.671826e+08
MMD : 1.252433e-01 ± 1.019108e-01
RVC : 6.720000e-01 ± 1.589409e-01
DPP : 3.375223e-19 ± 1.064283e-18

============================================================
```

### CSV Metrics Format

The CSV file contains (compatible with EngiOpt paper format):
- `iog` - Initial Optimality Gap
- `cog` - Cumulative Optimality Gap
- `fog` - Final Optimality Gap
- `mmd` - Maximum Mean Discrepancy
- `dpp` - Determinantal Point Process diversity
- `rvc` - Ratio of Violated Constraints
- `seed` - Random seed used
- `problem_id` - Problem identifier
- `model_id` - Model name
- `n_samples` - Number of samples
- `sigma` - Kernel bandwidth (default: 10.0)

## Per-Design Metrics Analysis

In addition to global metrics, per-design metrics are automatically saved for granular analysis.

### Design Metrics CSV Format

The `design_metrics.csv` file contains per-example metrics:
- `seed` - Random seed used
- `example_id` - Problem instance identifier
- `problem_id` - Problem type
- `model_id` - Model name
- `overall_score` - Weighted overall score (0-1)
- `iou` - Intersection over Union (topology overlap)
- `pixel_accuracy` - Element-wise accuracy
- `mse` - Mean Squared Error (density field)
- `constraint_score` - Constraint satisfaction score
- `objective_score` - Objective value match score
- Problem-specific metrics (volume fraction, compliance, etc.)

### Computing Per-Design Statistics

After running multiple seeds, compute per-design statistics:

```bash
python compute_design_stats.py results/openai_gpt-4.1/beams2d/design_metrics.csv
```

Output example:
```
============================================================
Per-Design Metrics Statistics for results/openai_gpt-4.1/beams2d/design_metrics.csv
============================================================

Number of seeds: 10
Number of examples per seed: 50
Total rows: 500

GLOBAL STATISTICS (across all seeds and examples)
------------------------------------------------------------
overall_score       : 0.654321 ± 0.123456
iou                 : 0.721234 ± 0.098765
pixel_accuracy      : 0.876543 ± 0.054321
mse                 : 0.012345 ± 0.006789
constraint_score    : 0.891234 ± 0.076543
objective_score     : 0.543210 ± 0.165432

PER-SEED AGGREGATED STATISTICS
------------------------------------------------------------
overall_score       : 0.654321 ± 0.045678
iou                 : 0.721234 ± 0.032109
...

PER-EXAMPLE VARIANCE (how consistent are results across seeds?)
------------------------------------------------------------
overall_score       : mean std = 0.098765
iou                 : mean std = 0.087654
...
```

### Use Cases for Per-Design Metrics

1. **Identify Problem Instances**: Find which specific problems the agent struggles with
2. **Stability Analysis**: Measure consistency across seeds for each problem
3. **Failure Mode Analysis**: Debug systematic failures on certain problem types
4. **Model Comparison**: Compare per-design performance between different models
5. **Statistical Significance**: Compute confidence intervals for design-level metrics

## Evaluation Metrics

### Generic Scorer

All problems now use the generic scorer (`score_output_quality_visual`) which provides:

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

## MMORE RAG Evaluation

To evaluate the agent **with** the MMORE RAG system enabled (for retrieval-augmented generation), use the dedicated evaluation service that runs on port 8001 to avoid conflicts with the main application.

### Setup

A separate Docker Compose file is provided for running MMORE during evaluations:

```bash
# Start the MMORE evaluation service (runs on port 8001)
make mmore-eval-up

# Check service status
make mmore-eval-status

# View logs
make mmore-eval-logs
```

### Running Evaluations with MMORE

Use the dedicated make command that sets the correct `MMORE_RAG_URL`:

```bash
# Run evaluation with MMORE enabled
make mmore-eval-run ARGS="--problem beams2d --samples 5 --scorers all --seed 1 --prompt full"
```

Or manually set the environment variable:

```bash
MMORE_RAG_URL=http://localhost:8001 python benchmarks/evaluations/evaluate_agent.py \
  --problem beams2d \
  --samples 5 \
  --scorers all \
  --seed 1 \
  --prompt full \
  --mmore
```

### MMORE Evaluation Commands

| Command | Description |
|---------|-------------|
| `make mmore-eval-up` | Start MMORE RAG service for evaluation (port 8001) |
| `make mmore-eval-down` | Stop MMORE RAG evaluation service |
| `make mmore-eval-rebuild` | Rebuild and restart MMORE RAG evaluation service |
| `make mmore-eval-logs` | Show logs for MMORE RAG evaluation service |
| `make mmore-eval-status` | Show status of MMORE RAG evaluation service |
| `make mmore-eval-run ARGS="..."` | Run evaluation with MMORE enabled |

### Comparing MMORE vs Non-MMORE

Run evaluations with and without MMORE to compare RAG impact:

```bash
# Without MMORE (default)
python evaluate_agent.py --problem beams2d --samples 10 --seed 1 --prompt full

# With MMORE
make mmore-eval-run ARGS="--problem beams2d --samples 10 --seed 1 --prompt full"
```

Results are automatically tagged with `mmore_on` or `mmore_off` in Weave trace names for easy comparison.

### Service Architecture

The evaluation MMORE service runs independently from the main application stack:

| Service | Container Name | Host Port | Internal Port |
|---------|---------------|-----------|---------------|
| Main MMORE | `mmore-rag-service` | 8000 | 8000 |
| Eval MMORE | `mmore-rag-eval` | 8001 | 8000 |

Both services use separate Docker networks and volumes, so they can run simultaneously without conflicts.

## Performance Considerations

- Evaluations run in parallel for speed
- Each example gets a unique thread_id to avoid state sharing
- Matplotlib uses Agg backend to prevent threading issues
- Results are cached locally to avoid re-computation
- **No Docker containers required by default**: Evaluation automatically skips Prusa MCP and MMORE services
  - `SKIP_MCP=true` - Skips Prusa MCP server connection
  - `SKIP_MMORE=true` - Skips MMORE Docker container requirement (default)
  - Use `--mmore` flag to enable MMORE RAG (requires `mmore-eval` service running)
