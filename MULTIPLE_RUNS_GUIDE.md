# Managing Multiple Evaluation Runs - Complete Guide

## Overview

The evaluation system already supports running multiple evaluations with the same model using **seeds** for reproducibility. Each seed creates an independent run with different random sampling.

## Directory Structure

Results are automatically organized by:
```
benchmarks/evaluations/results/
├── models/
│   └── {model_name}/
│       └── {problem}/
│           └── {prompt_style}/
│               └── {rag_status}/
│                   ├── output_quality_design_metrics.csv  # All runs combined
│                   ├── output_quality_global_metrics.csv  # Aggregated stats
│                   └── run_metadata.json                   # Tracking info
└── baselines/
    └── cgan_cnn_2d/
        └── {problem}/
            └── output_quality_global_metrics.csv
```

Where:
- `{model_name}`: e.g., `openai_gpt-4o`, `anthropic_claude-3-5-sonnet`
- `{problem}`: e.g., `beams2d`, `thermoelastic2d`, `photonics2d`
- `{prompt_style}`: e.g., `full`, `approximate`, `natural`, `workflow`
- `{rag_status}`: `rag` (with MMORE) or `no_rag` (without)

## Method 1: Using run_full_benchmark.py (Recommended)

### Basic Usage

Run multiple seeds sequentially:
```bash
# Run seeds 1-5 with 10 samples each
python benchmarks/evaluations/run_full_benchmark.py \
    --problem beams2d \
    --seeds 1 2 3 4 5 \
    --n_samples 10 \
    --model openai:gpt-4o

# Run with different prompt style
python benchmarks/evaluations/run_full_benchmark.py \
    --problem beams2d \
    --seeds 1 2 3 \
    --n_samples 20 \
    --model openai:gpt-4o \
    --prompt-style approximate

# Run with MMORE RAG enabled
python benchmarks/evaluations/run_full_benchmark.py \
    --problem beams2d \
    --seeds 1 2 3 4 5 \
    --n_samples 10 \
    --model openai:gpt-4o \
    --mmore

# Run only agent (skip CGAN baseline)
python benchmarks/evaluations/run_full_benchmark.py \
    --problem beams2d \
    --seeds 1 2 3 4 5 \
    --n_samples 10 \
    --agent-only
```

### Features
- ✅ Automatically generates prompts for each seed
- ✅ Runs evaluations sequentially
- ✅ Tracks success/failure per seed
- ✅ Aggregates results automatically
- ✅ Provides summary at the end

## Method 2: Manual Individual Runs

Run evaluations one at a time:
```bash
# Generate prompts first
python benchmarks/problems/beams2d/generate_prompts.py \
    --samples 10 \
    --seed 1 \
    --style full

# Run evaluation
python benchmarks/evaluations/evaluate_agent.py \
    --problem beams2d \
    --model openai:gpt-4o \
    --samples 10 \
    --seed 1 \
    --scorers generic

# Repeat for other seeds
python benchmarks/problems/beams2d/generate_prompts.py --samples 10 --seed 2 --style full
python benchmarks/evaluations/evaluate_agent.py --problem beams2d --model openai:gpt-4o --samples 10 --seed 2
```

## Method 3: Parallel Runs (Advanced)

For faster execution, run multiple seeds in parallel:

```bash
#!/bin/bash
# run_parallel.sh

MODEL="openai:gpt-4o"
PROBLEM="beams2d"
SAMPLES=10
SEEDS=(1 2 3 4 5)

# Run all seeds in parallel
for seed in "${SEEDS[@]}"; do
    (
        echo "Starting seed $seed"
        python benchmarks/problems/$PROBLEM/generate_prompts.py \
            --samples $SAMPLES \
            --seed $seed \
            --style full

        python benchmarks/evaluations/evaluate_agent.py \
            --problem $PROBLEM \
            --model $MODEL \
            --samples $SAMPLES \
            --seed $seed \
            --scorers generic

        echo "Completed seed $seed"
    ) &
done

# Wait for all background jobs to complete
wait
echo "All runs completed!"
```

Make executable and run:
```bash
chmod +x run_parallel.sh
./run_parallel.sh
```

## Tracking Run Status

### Create a run tracking script:

```python
# check_runs.py
from pathlib import Path
import pandas as pd

def check_run_status(model, problem, prompt_style="full", rag_status="no_rag"):
    """Check which seeds have been completed."""
    model_safe = model.replace("/", "_").replace(":", "_")
    results_dir = Path(f"benchmarks/evaluations/results/models/{model_safe}/{problem}/{prompt_style}/{rag_status}")

    if not results_dir.exists():
        print(f"No results directory found: {results_dir}")
        return

    csv_file = results_dir / "output_quality_design_metrics.csv"
    if not csv_file.exists():
        print(f"No results CSV found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # Check which seeds are present
    if 'seed' in df.columns:
        seeds_completed = sorted(df['seed'].unique())
        num_samples_per_seed = df.groupby('seed').size()

        print(f"\n{'='*60}")
        print(f"Results for {model} on {problem}")
        print(f"{'='*60}")
        print(f"Location: {results_dir}")
        print(f"\nCompleted seeds: {list(seeds_completed)}")
        print(f"Total runs: {len(seeds_completed)}")
        print(f"\nSamples per seed:")
        for seed, count in num_samples_per_seed.items():
            print(f"  Seed {seed}: {count} samples")
        print(f"\nTotal samples: {len(df)}")
    else:
        print(f"Total samples: {len(df)} (no seed column)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python check_runs.py MODEL PROBLEM [PROMPT_STYLE] [RAG_STATUS]")
        print("Example: python check_runs.py 'openai:gpt-4o' beams2d full no_rag")
        sys.exit(1)

    model = sys.argv[1]
    problem = sys.argv[2]
    prompt_style = sys.argv[3] if len(sys.argv) > 3 else "full"
    rag_status = sys.argv[4] if len(sys.argv) > 4 else "no_rag"

    check_run_status(model, problem, prompt_style, rag_status)
```

Usage:
```bash
python check_runs.py "openai:gpt-4o" beams2d
```

## Analyzing Aggregated Results

### View Per-Design Metrics (All Seeds Combined)

```python
import pandas as pd
from pathlib import Path

model_safe = "openai_gpt-4o"
problem = "beams2d"
prompt_style = "full"
rag_status = "no_rag"

# Load design-level metrics
csv_path = Path(f"benchmarks/evaluations/results/models/{model_safe}/{problem}/{prompt_style}/{rag_status}/output_quality_design_metrics.csv")
df = pd.read_csv(csv_path)

# Aggregate by seed
print("Metrics by seed:")
print(df.groupby('seed')[['iou', 'pixel_accuracy', 'constraint_score', 'is_watertight']].mean())

# Overall statistics
print("\nOverall statistics:")
print(df[['iou', 'pixel_accuracy', 'is_watertight', 'mesh_repaired']].describe())

# Watertightness analysis
if 'is_watertight' in df.columns:
    watertight_rate = df.groupby('seed')['is_watertight'].mean()
    print("\nWatertightness rate by seed:")
    print(watertight_rate)
```

### View Global Metrics (Per-Seed Aggregates)

```python
# Load global metrics
global_csv = csv_path.parent / "output_quality_global_metrics.csv"
if global_csv.exists():
    global_df = pd.read_csv(global_csv)
    print("Global metrics across seeds:")
    print(global_df[['seed', 'mean_iou', 'mean_pixel_accuracy', 'mmd', 'dpp_diversity']].round(4))
```

## Comparing Multiple Models

```bash
# Run the same seeds for different models
for model in "openai:gpt-4o" "anthropic:claude-3-5-sonnet"; do
    python benchmarks/evaluations/run_full_benchmark.py \
        --problem beams2d \
        --seeds 1 2 3 4 5 \
        --n_samples 10 \
        --model "$model" \
        --agent-only
done

# Compare results
python benchmarks/evaluations/compare_results.py \
    --problem beams2d \
    --prompt-style full
```

## Resuming Failed Runs

If some seeds fail, resume with only the failed ones:

```bash
# If seeds 2 and 4 failed, run only those
python benchmarks/evaluations/run_full_benchmark.py \
    --problem beams2d \
    --seeds 2 4 \
    --n_samples 10 \
    --model openai:gpt-4o \
    --agent-only
```

## Best Practices

1. **Start Small**: Test with 1-2 seeds and few samples first
   ```bash
   python benchmarks/evaluations/run_full_benchmark.py --seeds 1 --n_samples 5 --agent-only
   ```

2. **Use Consistent Seeds**: Use the same seeds (e.g., 1-5) across all model comparisons

3. **Track Costs**: Monitor API costs, especially with many samples
   - 10 samples/seed × 5 seeds = 50 total evaluations
   - Each evaluation may make multiple LLM calls

4. **Save Intermediate Results**: Results are saved incrementally, so you can stop/resume

5. **Organize by Experiment**: Use different prompt styles or RAG settings to organize experiments

6. **Check Watertightness Metrics**: New metrics are automatically included:
   ```python
   df['is_watertight'].value_counts()
   df['mesh_repaired'].value_counts()
   df.groupby('seed')[['is_watertight', 'mesh_repaired']].mean()
   ```

## Example: Full Evaluation Workflow

```bash
# 1. Run 5 seeds with 20 samples each
python benchmarks/evaluations/run_full_benchmark.py \
    --problem beams2d \
    --seeds 1 2 3 4 5 \
    --n_samples 20 \
    --model openai:gpt-4o \
    --scorers generic \
    --agent-only

# 2. Check status
python check_runs.py "openai:gpt-4o" beams2d

# 3. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('benchmarks/evaluations/results/models/openai_gpt-4o/beams2d/full/no_rag/output_quality_design_metrics.csv')
print('Seeds:', sorted(df['seed'].unique()))
print('Samples per seed:', df.groupby('seed').size())
print('Mean IoU by seed:', df.groupby('seed')['iou'].mean())
print('Watertight rate:', df['is_watertight'].mean())
"

# 4. Compare with baseline
python benchmarks/evaluations/compare_results.py \
    --problem beams2d \
    --prompt-style full
```

## Troubleshooting

**Issue**: Runs fail with API errors
- **Solution**: Add retry logic or reduce parallelization

**Issue**: Disk space
- **Solution**: Archive old results, or use `--scorers engibench` for minimal metrics

**Issue**: Memory issues
- **Solution**: Run fewer seeds in parallel, or reduce `--samples`

**Issue**: Results not aggregating
- **Solution**: Ensure all runs use the same model name format and prompt style

**Issue**: Watertightness metrics missing
- **Solution**: Ensure trimesh is installed: `pip install trimesh>=4.0.0`
