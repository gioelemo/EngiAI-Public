# Engineer Assistant Benchmarks

This directory contains benchmarking and evaluation infrastructure for the engineer assistant multi-agent system.

## Overview

The benchmarks evaluate the agent's performance on engineering design tasks across:
- **Multiple problem types** (beams2d, photonics2d, thermoelastic2d)
- **Multiple LLM models** (GPT-4o, Claude Sonnet, etc.)
- **Different configurations** (temperature, parameters, etc.)

Results are tracked using [Weave](https://wandb.ai/site/weave) for easy comparison and visualization.

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── shared/                      # Shared infrastructure
│   ├── problem_registry.py     # Central problem definitions
│   ├── explore_dataset.py      # Generic dataset explorer
│   ├── generic_scorer.py       # Universal topology optimizer scorer
│   ├── engibench_scorers.py   # Global MMD, DPP, RVC, optimality gap metrics
│   ├── metrics.py             # MMD, DPP, optimality gap computation utilities
│   └── utils.py               # Shared helper functions
├── problems/                    # Problem-specific prompt generation
│   ├── beams2d/                # 2D beam topology optimization
│   │   ├── README.md
│   │   ├── generate_prompts.py
│   │   ├── validate_prompts.py  # (beams2d only)
│   │   └── data/
│   │       ├── generated/      # Generated prompts
│   │       ├── validated/      # Validation reports
│   │       └── raw/           # Raw data samples (for exploration)
│   ├── photonics2d/            # 2D photonics optimization
│   │   ├── README.md
│   │   ├── generate_prompts.py
│   │   └── data/
│   │       └── generated/      # Generated prompts
│   └── thermoelastic2d/        # 2D thermoelastic multi-objective
│       ├── README.md
│       ├── generate_prompts.py
│       └── data/
│           └── generated/      # Generated prompts
└── evaluations/                # Unified evaluation framework
    ├── README.md
    ├── evaluate_agent.py      # Main evaluation script
    ├── compute_metrics_stats.py  # Compute global metrics statistics
    ├── compute_design_stats.py   # Compute per-design metrics statistics
    └── results/               # Results organized by model and problem
        ├── models/                # LLM agent results
        │   └── {model-name}/
        │       └── {problem-type}/
        │           └── {prompt-style}/
        │               └── {rag-status}/
        │                   ├── output_quality_global_metrics.csv   # Global metrics (MMD, DPP, RVC, gaps)
        │                   ├── output_quality_design_metrics.csv   # Per-design metrics (IoU, accuracy, etc.)
        │                   └── comparisons/          # Design comparison images
        │                       ├── seed_1/           # Per-seed comparisons
        │                       ├── seed_2/
        │                       └── ...
        └── baselines/         # Baseline method results (CGAN, etc.)
```

## Quick Start

### 1. Generate Prompts for a Problem

First, generate prompts from the dataset for the problem you want to evaluate:

```bash
conda activate engineer-assistant
cd benchmarks/problems/beams2d
python generate_prompts.py
```

### 2. (Optional) Validate Prompts

Ensure the generated prompts are well-formed (currently only available for beams2d):

```bash
python validate_prompts.py
```

### 3. Evaluate Agent

Run the evaluation on your chosen problem and model:

```bash
cd ../../evaluations
python evaluate_agent.py --problem beams2d --model gpt-4o --samples 5
```

### 4. View Results

- **Local:** Check `evaluations/results/models/{model}/{problem}/{prompt_style}/{rag_status}/comparisons/seed_N/` for comparison images
- **Weave Dashboard:** View full metrics, traces, and comparisons in the Weave UI

## Seed-Based Evaluation

To collect statistics across multiple optimization runs (matching the EngiOpt paper methodology), run evaluations with different seeds:

### Single Seed Evaluation

```bash
python evaluate_agent.py \
  --problem beams2d \
  --samples 50 \
  --scorers engibench \
  --seed 1
```

This saves metrics to `results/models/{model}/{problem}/{prompt_style}/{rag_status}/output_quality_global_metrics.csv` and comparison images to `results/models/{model}/{problem}/{prompt_style}/{rag_status}/comparisons/seed_1/`.

### Multiple Seeds for Statistical Analysis

Run the same evaluation with different seeds to collect statistics:

```bash
# Run 10 seeds matching EngiOpt paper methodology
for seed in {1..10}; do
  python evaluate_agent.py \
    --problem beams2d \
    --samples 50 \
    --scorers engibench \
    --seed $seed
done
```

Each run appends metrics to the same CSV file.

### Compute Statistics

After running multiple seeds, compute mean ± standard deviation:

```bash
cd benchmarks/evaluations
python compute_output_quality_global_stats.py results/models/openai_gpt-4.1/beams2d/full/no_rag/output_quality_global_metrics.csv
```

Output:
```
============================================================
Metrics Statistics for results/models/openai_gpt-4.1/beams2d/full/no_rag/output_quality_global_metrics.csv
============================================================

Number of runs: 10

COG : 1.399069e+08 ± 1.671826e+08
MMD : 1.252433e-01 ± 1.019108e-01
RVC : 6.720000e-01 ± 1.589409e-01
DPP : 3.375223e-19 ± 1.064283e-18

============================================================
```

### Metrics Output Format

The CSV file contains the following columns (compatible with EngiOpt paper format):
- `iog` - Initial Optimality Gap
- `cog` - Cumulative Optimality Gap
- `fog` - Final Optimality Gap
- `mmd` - Maximum Mean Discrepancy
- `dpp` - Determinantal Point Process diversity
- `rvc` - Ratio of Violated Constraints
- `seed` - Random seed used
- `problem_id` - Problem type
- `model_id` - Model identifier
- `n_samples` - Number of samples evaluated
- `sigma` - Kernel bandwidth for MMD/DPP (default: 10.0)

## Available Problems

### Beams 2D

**Dataset:** [IDEALLab/beams_2d_50_100_v0](https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0)

**Description:** 2D beam topology optimization with varying volume fractions, minimum feature sizes, and load distributions.

**Documentation:** [problems/beams2d/README.md](problems/beams2d/README.md)

**Evaluation:**
```bash
python evaluations/evaluate_agent.py --problem beams2d --samples 10
```

### Photonics 2D

**Dataset:** [IDEALLab/photonics_2d_120_120_v0](https://huggingface.co/datasets/IDEALLab/photonics_2d_120_120_v0)

**Description:** Optical topology optimization for maximizing field overlap in photonic structures.

**Documentation:** [problems/photonics2d/README.md](problems/photonics2d/README.md)

**Evaluation:**
```bash
python evaluations/evaluate_agent.py --problem photonics2d --samples 10
```

### Thermoelastic 2D

**Dataset:** [IDEALLab/thermoelastic_2d_v0](https://huggingface.co/datasets/IDEALLab/thermoelastic_2d_v0)

**Description:** Multi-objective topology optimization combining structural stiffness, thermal performance, and material efficiency.

**Documentation:** [problems/thermoelastic2d/README.md](problems/thermoelastic2d/README.md)

**Evaluation:**
```bash
python evaluations/evaluate_agent.py --problem thermoelastic2d --samples 10
```

### Future Problems

- **3D Structures** - Three-dimensional topology optimization
- **Multi-Physics** - Additional combined physics problems

## Comparing Models

Evaluate multiple models on the same problem to compare performance:

```bash
# Evaluate GPT-4o with 10 seeds
for seed in {1..10}; do
  python evaluations/evaluate_agent.py \
    --problem beams2d \
    --model gpt-4o \
    --samples 50 \
    --scorers engibench \
    --seed $seed
done

# Evaluate Claude Sonnet with 10 seeds
for seed in {1..10}; do
  python evaluations/evaluate_agent.py \
    --problem beams2d \
    --model claude-3-5-sonnet-20241022 \
    --samples 50 \
    --scorers engibench \
    --seed $seed
done

# Compare statistics
cd benchmarks/evaluations
python compute_output_quality_global_stats.py results/models/openai_gpt-4o/beams2d/full/no_rag/output_quality_global_metrics.csv
python compute_output_quality_global_stats.py results/models/anthropic_claude-3-5-sonnet-20241022/beams2d/full/no_rag/output_quality_global_metrics.csv

# View detailed comparison in Weave dashboard
```

Results are automatically organized by model in `results/models/{model}/{problem}/{prompt_style}/{rag_status}/` for easy comparison.

## Evaluation Metrics

The benchmarks use two types of scorer systems:

### Generic Scorer

All problems use the generic scorer (`score_output_quality_visual`) which evaluates design quality based on problem configuration:

- **Design Quality Metrics**
  - **IoU** (Intersection over Union) - Topology overlap
  - **Pixel Accuracy** - Element-wise accuracy
  - **MSE** (Mean Squared Error) - Density field error
- **Constraint Checking** - Configured per problem (e.g., volume fraction)
- **Objective Evaluation** - Configured per problem (e.g., compliance for beams2d, overlap for photonics2d)

The generic scorer is configuration-driven via `benchmarks.shared.problem_registry`.

See [problems/beams2d/SCORING_METRICS.md](problems/beams2d/SCORING_METRICS.md) for detailed metric definitions.

### Global Metrics (EngiBench)

Global metrics computed after evaluation completes (via `--scorers engibench` or `--scorers all`):

- **MMD** (Maximum Mean Discrepancy) - Similarity between generated designs and dataset distribution (lower is better)
- **DPP Diversity** - Design variability using Determinantal Point Process (higher is better)
- **RVC** (Ratio of Violated Constraints) - Fraction of designs violating constraints (lower is better)
- **IOG** (Initial Optimality Gap) - Average gap at start of optimization
- **COG** (Cumulative Optimality Gap) - Average gap accumulated during optimization
- **FOG** (Final Optimality Gap) - Average gap at end of optimization

### Per-Design Metrics

In addition to global metrics, every evaluation automatically saves per-design metrics to `design_metrics.csv` for granular analysis:

**Automatic metrics tracked:**
- `overall_score` - Weighted overall design score
- `iou` - Intersection over Union (topology similarity)
- `pixel_accuracy` - Element-wise accuracy
- `mse` - Mean squared error (density field)
- `constraint_score` - Constraint satisfaction score
- `objective_score` - Objective value match score
- Problem-specific metrics (volume fraction, compliance, etc.)

**Analyze per-design metrics:**
```bash
python compute_output_quality_design_stats.py results/models/openai_gpt-4.1/beams2d/full/no_rag/output_quality_design_metrics.csv
```

**Use cases:**
- Identify which problem instances cause failures
- Measure consistency across seeds for individual examples
- Compare per-design performance between models
- Debug systematic failure modes
- Compute confidence intervals for design-level metrics

See [evaluations/README.md](evaluations/README.md#per-design-metrics-analysis) for detailed documentation.

## Weave Integration

If Weave is enabled (`USE_WEAVE=true` in `.env`):
- Prompts are published to Weave datasets
- Evaluations are tracked with full lineage
- Results viewable in Weave dashboard with interactive charts
- Easy comparison across models and problems

Configure Weave in your `.env`:
```bash
USE_WEAVE=true
WEAVE_PROJECT="your-entity/your-project"
```

## Adding New Problem Types

To add a new problem type:

1. **Create problem directory:**
   ```bash
   mkdir -p benchmarks/problems/your_problem/data/{generated,validated,raw}
   ```

2. **Add to problem registry:**
   Edit `shared/problem_registry.py` and add your problem:
   ```python
   "your_problem": ProblemConfig(
       dataset_name="IDEALLab/your_dataset_name",
       objectives=[...],
       conditions=[...],
       design_metrics_weights={...},
   )
   ```

3. **Explore the dataset:**
   ```bash
   python -m benchmarks.shared.explore_dataset --problem your_problem
   ```

4. **Create problem scripts:**
   - `generate_prompts.py` - Generate prompts from dataset
   - `validate_prompts.py` - Validate prompt quality

5. **Document the problem:**
   Create `problems/your_problem/README.md` with dataset info and usage

6. **Test the workflow:**
   ```bash
   cd problems/your_problem
   python generate_prompts.py
   python validate_prompts.py
   cd ../../evaluations
   python evaluate_agent.py --problem your_problem --samples 5
   ```

## Adding Custom Scorers

To create problem-specific scorers:

1. **Add scorer function to problem directory** (e.g., `problems/your_problem/scorers.py`):
   ```python
   @weave.op()
   def score_your_metric(
       output: dict[str, Any],
       target: dict[str, Any],
       metadata: dict[str, Any],
   ) -> dict[str, Any]:
       # Your scoring logic
       # Extract data from output, compare with target
       return {"score": 0.85, "details": "..."}
   ```

2. **Add to problem configuration** in `evaluations/evaluate_agent.py`:
   ```python
   from benchmarks.problems.your_problem.scorers import score_your_metric

   PROBLEM_CONFIGS = {
       "your_problem": {
           "scorers": [score_your_metric],
           ...
       }
   }
   ```

3. **(Optional) Export from problem's `__init__.py`** for easier imports

## Requirements

All benchmarks use the main project dependencies from `pyproject.toml`:
- `weave` - Evaluation tracking and visualization
- `datasets` - HuggingFace dataset access
- `matplotlib` - Visualizations
- `numpy` - Numerical operations
- `langchain` - Agent framework
- `langgraph` - Agent orchestration

## Troubleshooting

### "File not found" errors
- Ensure you've run `generate_prompts.py` for the problem type first
- Check that you're in the correct directory

### Import errors
- Activate the conda environment: `conda activate engineer-assistant`
- Ensure project root is in `PYTHONPATH` (scripts handle this automatically)

### Weave errors
- Check that `USE_WEAVE=true` in `.env`
- Verify `WEAVE_PROJECT` is set correctly
- Ensure you have the `weave` package installed

### Design extraction failures
- Check that `src/tools/engibench.py` includes `optimized_design` field in return dict
- Verify the agent is calling `optimize_design` with a `config` parameter
- Review debug output for parsing errors

## Notes

### Design Comparison Methodology

The pixel-wise design comparison (via the generic scorer) measures how similar the agent's design is to the ground truth. Large differences are expected because:

1. **Multiple Local Optima** - Topology optimization has many valid solutions
2. **Sensitivity to Initialization** - Different random seeds produce different designs
3. **Algorithmic Variations** - Different optimizers may converge to different solutions

**Alternative Evaluation Approaches:**
- Compare objective values (compliance) rather than pixel-wise similarity
- Use same random seeds as ground truth for reproducibility
- Evaluate structural quality metrics (connectivity, symmetry)
- Assess constraint satisfaction rather than design similarity

### Parallel Evaluation

- The evaluation runs examples in parallel for speed
- Each example gets a unique `thread_id` to avoid state sharing
- Design extraction from message history ensures no global cache pollution
- Matplotlib uses Agg backend to prevent threading issues

### Debug Output

The evaluation script provides detailed debug output showing:
- Parameter extraction from prompts
- Design parsing from tool messages
- Scorer computations and intermediate values

This is helpful for diagnosing issues and understanding agent behavior.

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@software{engineer_assistant_benchmarks,
  title = {Engineer Assistant Benchmarks},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/engineer-assistant}
}
```
