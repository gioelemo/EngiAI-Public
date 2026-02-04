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

# Step 1: Run evaluation (saves to Weave)
python evaluate_agent.py --problem beams2d --samples 5

# Step 2: Extract data from Weave to JSON
python extract_data.py --problem beams2d

# Step 3: Compute global metrics
python compute_global_metrics.py --problem beams2d
```

Or use the complete pipeline:

```bash
python run_full_benchmark.py --problem beams2d --samples 5 --seeds 1 --agent-only
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

- `--scorers all` - All scorers: output_quality + task_completion + tool_use (default)
- `--scorers output_quality` - Per-design quality metrics only
- `--scorers task_completion` - Task completion checker only
- `--scorers tool_use` - Tool usage efficiency only

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--problem` | Problem type to evaluate | `beams2d` |
| `--model` | LLM model name | From config |
| `--samples` | Number of samples to evaluate | `10` |
| `--temperature` | Model temperature | From config |
| `--split` | Dataset split (train/val/test) | `test` |
| `--scorers` | Scorer set (output_quality/all/task_completion/tool_use) | `all` |
| `--seed` | Random seed for optimization | `None` |
## Results Organization

Results are automatically organized by model and problem type:

```
results/
├── models/                      # LLM agent results (JSON format)
│   └── {model-name}/
│       └── {problem-type}/
│           └── {prompt-style}/
│               └── {rag-status}/
│                   ├── designs_seed_1.json           # Per-design metrics
│                   ├── designs_seed_2.json
│                   ├── global_metrics_seed_1.json    # Global metrics
│                   ├── global_metrics_seed_2.json
│                   └── comparisons/          # Comparison visualizations
│                       ├── seed_1/           # Per-seed comparisons
│                       │   ├── comparison_example_0.png
│                       │   └── ...
│                       └── seed_2/
└── baselines/                   # Baseline method results (CSV format)
    └── cgan_cnn_2d/
        └── {problem}/
            └── output_quality_global_metrics.csv
```


## Seed-Based Evaluation

For reproducible benchmarking and statistical analysis (matching EngiOpt paper methodology):

### Single Seed Evaluation

```bash
python run_full_benchmark.py \
  --problem beams2d \
  --samples 50 \
  --seeds 1 \
  --agent-only
```

This runs the complete pipeline:
1. Evaluates agent and saves to Weave
2. Extracts per-design data to `designs_seed_1.json`
3. Computes global metrics to `global_metrics_seed_1.json`
4. Saves comparisons to `comparisons/seed_1/`

### Multiple Seeds for Statistics

Run with multiple seeds to collect statistical data:

```bash
# Run 10 seeds (EngiOpt paper methodology)
python run_full_benchmark.py \
  --problem beams2d \
  --samples 50 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --agent-only
```

Each seed creates separate JSON files: `designs_seed_N.json` and `global_metrics_seed_N.json`.

### Analyzing Results

After running multiple seeds, generate visualizations:

```bash
python plots/run_all.py --problem beams2d --model {your-model}
```

This creates:
- Global metrics plots (MMD, DPP, RVC, optimality gaps)
- Per-design metrics analysis
- Tool usage statistics
- Token and latency analysis

Or compare against baselines:

```bash
python compare_results.py --problem beams2d --model {your-model}
```

### JSON Metrics Format

Global metrics files (`global_metrics_seed_N.json`) contain:
- `iog` - Initial Optimality Gap
- `cog` - Cumulative Optimality Gap
- `fog` - Final Optimality Gap
- `mmd` - Maximum Mean Discrepancy
- `dpp` - Determinantal Point Process diversity
- `rvc` - Ratio of Violated Constraints
- `seed` - Random seed used
- `problem` - Problem identifier
- `model` - Model name
- `prompt_style` - Prompt style
- `rag_status` - RAG system status
- `n_samples` - Number of samples

Per-design metrics files (`designs_seed_N.json`) contain arrays with individual design evaluations.

## Per-Design Metrics Analysis

Per-design metrics are automatically saved to JSON files for granular analysis.

### Design Metrics JSON Format

The `designs_seed_N.json` files contain arrays with per-example metrics:
- `seed` - Random seed used
- `example_id` - Problem instance identifier
- `problem` - Problem type
- `model` - Model name
- `overall_score` - Weighted overall score (0-1)
- `iou` - Intersection over Union (topology overlap)
- `pixel_accuracy` - Element-wise accuracy
- `mse` - Mean Squared Error (density field)
- `constraint_score` - Constraint satisfaction score
- `objective_score` - Objective value match score
- `design_found` - Whether design was extracted
- `constraint_violations` - Number of violations
- `total_tokens` - Token usage
- `latency_ms` - Response latency
- `total_tools` - Tool calls made
- `unique_tools` - Unique tools used
- Problem-specific metrics (volume fraction, compliance, etc.)

### Analyzing Per-Design Data

Use the plotting tools to analyze per-design metrics:

```bash
python plots/run_all.py --problem beams2d --model {your-model}
```

This generates comprehensive analysis including:
- Distribution of scores across examples
- Correlation between metrics
- Success rate analysis
- Failure mode identification

### Use Cases for Per-Design Metrics

1. **Identify Problem Instances**: Find which specific problems the agent struggles with
2. **Stability Analysis**: Measure consistency across seeds for each problem
3. **Failure Mode Analysis**: Debug systematic failures on certain problem types
4. **Model Comparison**: Compare per-design performance between different models
5. **Statistical Significance**: Compute confidence intervals for design-level metrics

## Evaluation Metrics

### Output Quality Scorer

All problems use the output quality scorer (`score_output_quality`) which provides:

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
make mmore-eval-run ARGS="--problem beams2d --samples 5 --scorers all --seed 1 --prompt-style full"
```

Or manually set the environment variable:

```bash
MMORE_RAG_URL=http://localhost:8001 python benchmarks/evaluations/evaluate_agent.py \
  --problem beams2d \
  --samples 5 \
  --scorers all \
  --seed 1 \
  --prompt-style full \
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
python evaluate_agent.py --problem beams2d --samples 10 --seed 1 --prompt-style full

# With MMORE
make mmore-eval-run ARGS="--problem beams2d --samples 10 --seed 1 --prompt-style full"
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
