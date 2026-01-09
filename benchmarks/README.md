# Engineer Assistant Benchmarks

This directory contains benchmarking and evaluation infrastructure for the engineer assistant multi-agent system.

## Overview

The benchmarks evaluate the agent's performance on engineering design tasks across:
- **Multiple problem types** (beams2d, thermoelastic2d, etc.)
- **Multiple LLM models** (GPT-4o, Claude Sonnet, etc.)
- **Different configurations** (temperature, parameters, etc.)

Results are tracked using [Weave](https://wandb.ai/site/weave) for easy comparison and visualization.

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── problems/                    # Problem-specific prompt generation
│   ├── beams2d/                # 2D beam topology optimization
│   │   ├── README.md
│   │   ├── generate_prompts.py
│   │   ├── validate_prompts.py
│   │   ├── explore_dataset.py
│   │   └── data/
│   │       ├── generated/      # Generated prompts
│   │       ├── validated/      # Validation reports
│   │       └── raw/           # Raw data samples
│   └── thermoelastic2d/       # (Future) Thermoelastic problems
├── evaluations/                # Unified evaluation framework
│   ├── README.md
│   ├── evaluate_agent.py      # Main evaluation script
│   └── results/               # Results organized by model and problem
│       ├── gpt-4o/
│       │   ├── beams2d/
│       │   └── thermoelastic2d/
│       └── claude-3-5-sonnet-20241022/
│           ├── beams2d/
│           └── thermoelastic2d/
└── shared/                     # Shared utilities and scorers
    ├── __init__.py
    ├── scorers.py             # Evaluation scorer functions
    └── utils.py               # Helper functions
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

Ensure the generated prompts are well-formed:

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

- **Local:** Check `evaluations/results/{model}/{problem}/comparisons/` for comparison images
- **Weave Dashboard:** View full metrics, traces, and comparisons in the Weave UI

## Available Problems

### Beams 2D

**Dataset:** [IDEALLab/beams_2d_50_100_v0](https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0)

**Description:** 2D beam topology optimization with varying volume fractions, minimum feature sizes, and load distributions.

**Documentation:** [problems/beams2d/README.md](problems/beams2d/README.md)

**Evaluation:**
```bash
python evaluations/evaluate_agent.py --problem beams2d --samples 10
```

### Future Problems

- **Thermoelastic 2D** - Coupled thermal-structural optimization
- **3D Structures** - Three-dimensional topology optimization
- **Multi-Physics** - Combined physics problems

## Comparing Models

Evaluate multiple models on the same problem to compare performance:

```bash
# Evaluate GPT-4o
python evaluations/evaluate_agent.py --problem beams2d --model gpt-4o --samples 20

# Evaluate Claude Sonnet
python evaluations/evaluate_agent.py --problem beams2d --model claude-3-5-sonnet-20241022 --samples 20

# View comparison in Weave dashboard
```

Results are automatically organized by model for easy comparison.

## Evaluation Metrics

### Qualitative Scorers

These scorers assess the quality of the agent's reasoning and communication:

- **Constraint Accuracy** - Does agent mention all constraint values?
- **Target Awareness** - Does agent reference target compliance?
- **Understands Tradeoffs** - Does agent understand material/performance tradeoffs?
- **Actionable Guidance** - Does agent provide concrete steps?
- **No Contradictions** - Does agent avoid incorrect statements?

### Quantitative Scorers

These scorers measure the similarity between agent design and ground truth:

- **Design Match** - Overall design similarity score
  - **IoU** (Intersection over Union) - Topology overlap
  - **Pixel Accuracy** - Element-wise accuracy
  - **MSE** (Mean Squared Error) - Density field error
  - **Volume Fraction Error** - Material usage difference

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

2. **Create problem scripts:**
   - Copy and adapt scripts from `problems/beams2d/`
   - `generate_prompts.py` - Generate prompts from dataset
   - `validate_prompts.py` - Validate prompt quality
   - `explore_dataset.py` - (Optional) Explore dataset

3. **Add to evaluation config:**
   Edit `evaluations/evaluate_agent.py` and add to `PROBLEM_CONFIGS`:
   ```python
   "your_problem": {
       "dataset_name": "huggingface/dataset-name",
       "prompt_file": "your_prompts.json",
       "scorers": [...],
   }
   ```

4. **Document the problem:**
   Create `problems/your_problem/README.md` with dataset info and usage

5. **Test the workflow:**
   ```bash
   cd problems/your_problem
   python generate_prompts.py
   python validate_prompts.py
   cd ../../evaluations
   python evaluate_agent.py --problem your_problem --samples 5
   ```

## Adding Custom Scorers

To create problem-specific scorers:

1. Add scorer function to `shared/scorers.py`:
   ```python
   @weave.op()
   def score_your_metric(
       prompt: str,
       conditions: dict[str, Any],
       output: dict[str, Any],
       target: dict[str, Any],
       metadata: dict[str, Any],
   ) -> dict[str, Any]:
       # Your scoring logic
       return {"score": 0.85, "details": "..."}
   ```

2. Add to problem configuration in `evaluations/evaluate_agent.py`

3. Import in `shared/__init__.py`

## Requirements

All benchmarks use the main project dependencies from `environment.yml`:
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

The pixel-wise design comparison (`score_design_match`) measures how similar the agent's design is to the ground truth. Large differences are expected because:

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
