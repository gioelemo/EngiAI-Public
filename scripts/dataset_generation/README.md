# Beam Prompt Dataset Generation

This directory contains scripts for generating benchmark datasets for Weave evaluation using beam design data from HuggingFace.

## Dataset Source

**HuggingFace Dataset**: [IDEALLab/beams_2d_50_100_v0](https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0)
- **Rows**: ~3.88k
- **Features**: optimal_design, volfrac, rmin, forcedist, overhang_constraint, c, optimization_history

## Workflow

### 1. Exploration ✅
```bash
python scripts/dataset_generation/explore_beams_dataset.py
```

Explores the dataset structure and saves sample data to `data/datasets/beam_prompts/raw/`.

### 2. Prompt Generation ✅
```bash
python scripts/dataset_generation/generate_beam_prompts.py
```

Generates natural language prompts from beam conditions:
- Transforms numerical parameters into readable descriptions
- Creates structured prompts for benchmarking
- Saves locally to `data/datasets/beam_prompts/generated/`
- Publishes to Weave for tracking and evaluation

**Output Example:**
```
Design a 2D beam structure with the following constraints:
- Volume fraction: 23.8% (use only 23.8% of available material)
- Minimum feature size (rmin): 3.5
- Load condition: a uniformly distributed force
- Target: Minimize compliance (maximize stiffness)

The beam should be designed on a 50x100 grid.
```

### 3. Validation ✅
```bash
python scripts/dataset_generation/validate_prompts.py
```

Validates that generated prompts are correct and consistent:
- **Numerical accuracy**: Prompt text matches numerical conditions
- **Parameter ranges**: Values are within expected bounds
- **Completeness**: All required fields are present
- **Consistency**: Descriptions align with parameter values

Saves validation report to `data/datasets/beam_prompts/validated/validation_report.json`

**Current Results:** 100% validation success (50/50 prompts passed all checks)

### 4. Evaluation ✅
```bash
python scripts/dataset_generation/evaluate_agent.py
```

Evaluates the **complete multi-agent system** (SupervisorAgent + specialized agents) on beam design prompts:
- **Multi-agent routing**: Tests the supervisor's ability to route beam design tasks to appropriate agents (likely EngineeringAgent)
- **Task completion rate**: Agent successfully responds to all prompts
- **Response quality**: Structured, comprehensive responses
- **Numerical accuracy**: Correct acknowledgment of constraints
- **Domain knowledge**: Understanding of topology optimization and beam design

Uses the **production model configuration** (`config.llm_model` and `config.llm_temperature`) to test the actual chatbot multi-agent system. Tracks metrics using Weave evaluation framework:
- Acknowledges volume fraction and minimum radius
- Mentions compliance and optimization
- Shows structural understanding
- Provides substantial guidance

Results are visible in the Weave dashboard with full traceability of agent routing and tool usage.

**Current Results:** 100% quality score (5/5 samples, all metrics passed)

## Directory Structure

```
data/datasets/beam_prompts/
├── raw/          # Downloaded HF data samples
├── generated/    # Generated prompts
├── validated/    # Validation reports
└── evaluated/    # Agent evaluation results
```

## Requirements

```bash
pip install datasets  # HuggingFace datasets library
```

Already available in the project environment.
