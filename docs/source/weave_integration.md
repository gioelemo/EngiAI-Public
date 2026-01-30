# Benchmarking with Weave

This guide explains how to use the benchmarking system powered by Weights & Biases Weave for evaluating agent performance on engineering optimization problems.

## Overview

The Engineer Assistant includes a comprehensive benchmarking system that evaluates agent performance on structural optimization problems using real datasets. Weave integration is already set up throughout the codebase for automatic tracking and evaluation.

**Key Features:**
- **Automated Evaluations**: Run agents on standard problem sets with automatic scoring
- **Multiple Problem Types**: Beams, photonics, and thermoelastic optimization
- **Performance Metrics**: Both per-design scores and global distribution metrics
- **Trace Visualization**: View detailed execution traces in the Weave UI

## Quick Start

### Configuration

Add the following to your `.env` file:

```bash
# Weave configuration (for LLM tracing and benchmarking)
USE_WEAVE=true
USE_WEAVE_CHATBOT=false
WEAVE_PROJECT="gioelemo-ethz/engineer-assistant-benchmarks"
```

- `USE_WEAVE`: Enable Weave tracing for evaluations and benchmarks
- `USE_WEAVE_CHATBOT`: Enable/disable tracing for chatbot interactions (default: `false`)
- `WEAVE_PROJECT`: W&B project name in format `entity/project-name`

### Running Benchmarks

The main evaluation script is located in `benchmarks/evaluations/`:

```bash
cd benchmarks/evaluations
python evaluate_agent.py \
    --problem beams2d \
    --num-samples 10 \
    --agent-config config.yaml
```

**📚 For complete documentation see:**
- [**Benchmarks Overview**](../../benchmarks/README.md) - Available problems, datasets, and metrics
- [**Evaluation Guide**](../../benchmarks/evaluations/README.md) - How to run evaluations and configure scorers

### Available Problem Types

Three engineering optimization problems are available:

| Problem | Dataset | Design Space | Metrics |
|---------|---------|--------------|---------|
| **beams2d** | 50×100 grids | Structural beams | Compliance, volume fraction, binary |
| **photonics2d** | 120×120 grids | Photonic devices | Transmission efficiency, volume |
| **thermoelastic2d** | Variable size | Thermal structures | Compliance, thermal loss, volume |

See [benchmarks/README.md](../../benchmarks/README.md) for detailed problem descriptions.

## Evaluation Workflow

### 1. Choose Your Problem

Select from the available optimization problems:

```bash
# Structural beam optimization
python evaluate_agent.py --problem beams2d

# Photonic device design
python evaluate_agent.py --problem photonics2d

# Thermoelastic optimization
python evaluate_agent.py --problem thermoelastic2d
```

### 2. Configure Evaluation

Specify the number of test samples:

```bash
# Run on 10 samples for quick testing
python evaluate_agent.py --problem beams2d --num-samples 10

# Run on full dataset (50 samples for beams2d)
python evaluate_agent.py --problem beams2d --num-samples 50
```

### 3. Select Scorer

Choose between two scoring methods:

- **`generic`** (default): Per-design metrics (compliance, volume, binary)
- **`engibench`**: Global distribution metrics (MMD, DPP, RVC, optimality gap)

```bash
# Per-design evaluation
python evaluate_agent.py --problem beams2d --scorer generic

# Global distribution evaluation
python evaluate_agent.py --problem beams2d --scorer engibench
```

See the [Evaluation Guide](../../benchmarks/evaluations/README.md) for scorer details.

### 4. View Results

All evaluation runs are automatically tracked in Weave. Access your results at:

```
https://wandb.ai/your-username/engineer-assistant-benchmarks/weave
```

The Weave UI shows:
- **Traces**: Full execution traces with timing
- **Calls**: Individual LLM calls with inputs/outputs
- **Metrics**: Evaluation scores and performance data
- **Costs**: Token usage and estimated API costs

## Understanding Metrics

### Per-Design Metrics (generic scorer)

Evaluated for each generated design:

- **Compliance**: Structural flexibility (lower is better)
- **Volume Fraction**: Material usage (target-dependent)
- **Binary Score**: How close to binary (0/1) values

### Global Metrics (engibench scorer)

Evaluated across the full set of generated designs:

- **MMD (Maximum Mean Discrepancy)**: Distribution similarity to reference set
- **DPP (Determinantal Point Process)**: Design diversity
- **RVC (Relative Volume Coverage)**: Design space coverage
- **Optimality Gap**: Distance from optimal solutions

See [benchmarks/README.md](../../benchmarks/README.md#metrics) for detailed metric definitions.

## Advanced Usage

### Custom Agent Configuration

Create a configuration file for your agent:

```yaml
# agent_config.yaml
model: "gpt-4"
temperature: 0.7
max_iterations: 10
```

Run with custom config:

```bash
python evaluate_agent.py \
    --problem beams2d \
    --num-samples 10 \
    --agent-config agent_config.yaml
```

### Batch Evaluations

Evaluate multiple configurations:

```bash
# Test different problems
for problem in beams2d photonics2d thermoelastic2d; do
    python evaluate_agent.py --problem $problem --num-samples 10
done

# Test different scorers
for scorer in generic engibench; do
    python evaluate_agent.py --problem beams2d --scorer $scorer
done
```

### Programmatic Evaluation

Run evaluations from Python code:

```python
import weave
from benchmarks.evaluations.evaluate_agent import run_evaluation

# Initialize Weave
weave.init("engineer-assistant-benchmarks")

# Run evaluation
results = run_evaluation(
    problem="beams2d",
    num_samples=10,
    scorer="generic"
)

print(f"Average compliance: {results['avg_compliance']}")
```

## Adding New Problem Types

To add a new optimization problem to the benchmark suite:

1. **Create problem directory**: `benchmarks/problems/your_problem/`
2. **Add to registry**: Update `benchmarks/shared/problem_registry.py`
3. **Implement generator**: Create `generate_prompts.py` for your problem
4. **Add dataset**: Create or reference HuggingFace dataset

See the [Evaluation Guide](../../benchmarks/evaluations/README.md#adding-new-problem-types) for complete instructions.

## Best Practices

1. **Start Small**: Test with `--num-samples 10` before running full evaluations
2. **Use Generic Scorer**: For per-design analysis and debugging
3. **Use EngiBench Scorer**: For comparing overall performance across agents
4. **Track Costs**: Monitor token usage in the Weave UI to manage API costs
5. **Version Control**: Keep track of agent configurations and model versions
6. **Regular Testing**: Run benchmarks regularly to detect performance regressions

## Troubleshooting

### Weave Not Initialized

**Problem**: No traces appearing in Weave UI

**Solution**: Ensure `USE_WEAVE=true` in your `.env` file and `WANDB_API_KEY` is set:

```bash
export WANDB_API_KEY="your-api-key"
echo "USE_WEAVE=true" >> .env
```

### Import Errors

**Problem**: Cannot import benchmark modules

**Solution**: Run from the correct directory:

```bash
cd benchmarks/evaluations
python evaluate_agent.py --problem beams2d
```

### Dataset Loading Fails

**Problem**: Cannot load HuggingFace dataset

**Solution**: Check internet connection and dataset name in `problem_registry.py`:

```python
# Verify dataset names match HuggingFace
PROBLEMS = {
    "beams2d": ProblemConfig(
        dataset="IDEALLab/beams_2d_50_100_v0",  # Must match exactly
        ...
    )
}
```

### Scorer Not Found

**Problem**: `ValueError: Unknown scorer: ...`

**Solution**: Use valid scorer names: `generic` or `engibench`

```bash
# Correct
python evaluate_agent.py --problem beams2d --scorer generic

# Incorrect
python evaluate_agent.py --problem beams2d --scorer custom  # Not supported
```

## Further Reading

- [Benchmarks Overview](../../benchmarks/README.md) - Complete benchmark system documentation
- [Evaluation Guide](../../benchmarks/evaluations/README.md) - Detailed evaluation instructions
- [Problem Registry](../../benchmarks/shared/problem_registry.py) - Problem configuration reference
- [Weave Documentation](https://docs.wandb.ai/weave) - Official Weave documentation

## Technical Integration Details

For developers who need to understand how Weave is integrated into the codebase:

### Automatic Tracing

Weave is initialized in `config.py` and automatically traces all LangChain operations:

```python
from config import config

# Initialize Weave based on .env configuration
config.setup_weave_tracing()  # Called at startup
```

After initialization, all LangChain components are automatically traced:
- Agent executions
- LLM calls (OpenAI, Google, etc.)
- Tool usage
- Chain invocations

### Custom Operations

Use the `@weave.op()` decorator for custom traced operations:

```python
import weave

@weave.op()
def custom_optimization(design: dict, constraints: dict) -> dict:
    """Custom optimization function with Weave tracing."""
    optimized = run_optimization(design, constraints)
    return optimized
```

### Conditional Tracing

Check if Weave is enabled before Weave-specific operations:

```python
from src.utils.weave_integration import is_weave_enabled
import weave

if is_weave_enabled():
    dataset = weave.Dataset(name="my_dataset", rows=data)
    weave.publish(dataset)
else:
    # Alternative logging
    print("Weave disabled, using local logging")
```

See [Weave Integration Utils](../../src/utils/weave_integration.py) for implementation details.
