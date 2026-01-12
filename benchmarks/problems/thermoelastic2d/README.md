# ThermoElastic2D Benchmark

Multi-objective topology optimization problem combining structural and thermal objectives.

## Problem Description

ThermoElastic2D is a coupled physics topology optimization problem that simultaneously considers:
1. **Structural performance** - Minimizing structural compliance (maximizing stiffness)
2. **Thermal performance** - Minimizing thermal compliance (maximizing heat dissipation)
3. **Material usage** - Minimizing volume fraction

This represents real-world engineering scenarios where structures must bear mechanical loads while efficiently dissipating heat (e.g., heat sinks, electronic enclosures, engine components).

## Dataset

- **HuggingFace**: [IDEALLab/thermoelastic_2d_v0](https://huggingface.co/datasets/IDEALLab/thermoelastic_2d_v0)
- **EngiBench**: [thermoelastic2d](https://engibench.ethz.ch/problems/thermoelastic2d/)
- **Design Space**: 64×64 grid
- **Material**: Binary distribution (0=void, 1=material)

## Key Features

### Multi-Objective Optimization
Unlike single-objective problems (beams2d, photonics2d), this problem optimizes three competing objectives simultaneously:

| Objective | Direction | Weight | Description |
|-----------|-----------|--------|-------------|
| Structural Compliance | Minimize | 0.4 | Structural stiffness under mechanical loads |
| Thermal Compliance | Minimize | 0.4 | Thermal conductivity for heat dissipation |
| Volume Fraction | Minimize | 0.2 | Material usage efficiency |

### Problem Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `volfrac` | 0.3 | Initial volume fraction constraint |
| `rmin` | 1.1 | Filter radius for manufacturing constraints |
| `weight` | 0.5 | Trade-off weight (0=thermal only, 1=structural only) |

### Boundary Conditions

The problem includes element arrays defining:
- **Fixed elements**: Boundary conditions (supports/fixtures)
- **Force elements (x, y)**: Applied mechanical loads
- **Heatsink elements**: Thermal boundary conditions (heat removal locations)

## Usage

### 1. Generate Prompts

Generate prompts from the HuggingFace dataset:

```bash
python benchmarks/problems/thermoelastic2d/generate_prompts.py --split test --samples 50
```

Options:
- `--split`: Dataset split (train/val/test)
- `--samples`: Number of samples to generate
- `--no-targets`: Exclude target values

### 2. Run Evaluation

Evaluate an agent on thermoelastic2d:

```bash
python benchmarks/evaluations/evaluate_agent.py \
  --problem thermoelastic2d \
  --model gpt-4o \
  --samples 5 \
  --split test
```

### 3. Explore Dataset

Explore the dataset structure and statistics:

```bash
python -m benchmarks.shared.explore_dataset --problem thermoelastic2d
```

## Evaluation Metrics

The generic scorer evaluates designs across multiple dimensions:

### 1. Design Similarity (60%)
- **IoU** (Intersection over Union): 40% weight
- **Pixel Accuracy**: 25% weight
- **MSE** (Mean Squared Error): Computed but not weighted

### 2. Objective Matching (35%)
Multi-objective scoring with weighted combination:
- **Structural compliance score**: 40% of objective score
- **Thermal compliance score**: 40% of objective score
- **Volume fraction score**: 20% of objective score

Each objective is scored based on relative error vs. threshold (20% for compliances, 10% for volume).

### 3. No Constraint Checking (0%)
Volume fraction is an objective, not a hard constraint in this problem.

## Multi-Objective Challenges

This problem is more challenging than single-objective problems because:

1. **Conflicting Objectives**: Structural and thermal optima often conflict
   - Structural: Prefers continuous load paths
   - Thermal: Prefers distributed material for heat spreading

2. **Weight Parameter**: The agent must understand how to balance objectives based on the `weight` parameter

3. **Pareto Optimality**: No single "best" solution - trade-offs between objectives

4. **Complex Physics**: Coupled structural-thermal simulation required

## Example Prompt

```
Design a 2D thermoelastic structure with coupled structural and thermal optimization:

Parameters:
- Initial volume fraction: 0.30 (target material usage)
- Filter radius (rmin): 1.10
- Objective weight: 0.50 (balance between structural and thermal compliance)

Objectives (all to minimize):
1. Structural compliance: Minimize structural deformation under mechanical loads
2. Thermal compliance: Minimize thermal resistance for heat dissipation
3. Volume fraction: Minimize material usage

The design should be on a 64x64 grid with binary material distribution (0=void, 1=material).
The weight parameter (0.50) controls the trade-off between structural (weight)
and thermal (1-weight) objectives.
```

## Scoring Configuration

The problem is configured in `benchmarks/shared/problem_registry.py`:

```python
"thermoelastic2d": ProblemConfig(
    name="thermoelastic2d",
    dataset_name="IDEALLab/thermoelastic_2d_v0",
    objectives=[
        ObjectiveConfig(
            name="structural_compliance",
            direction="minimize",
            weight=0.4,
            relative_error_threshold=0.2,
        ),
        ObjectiveConfig(
            name="thermal_compliance",
            direction="minimize",
            weight=0.4,
            relative_error_threshold=0.2,
        ),
        ObjectiveConfig(
            name="volume_fraction",
            direction="minimize",
            weight=0.2,
            relative_error_threshold=0.1,
        ),
    ],
    design_metrics_weights={
        "iou": 0.4,
        "pixel_accuracy": 0.25,
        "constraint_match": 0.0,
        "objective_match": 0.35,
    },
)
```

## Notes

### Generic Scorer
This problem uses the generic scorer (`benchmarks/shared/generic_scorer.py`) which automatically:
- Extracts all three objectives from tool messages
- Computes weighted multi-objective scores
- Handles the complex scoring logic

### Weight Parameter Interpretation
- `weight = 0.0`: Pure thermal optimization
- `weight = 0.5`: Balanced structural-thermal optimization
- `weight = 1.0`: Pure structural optimization

### Comparison to Other Problems
| Problem | Objectives | Design Size | Complexity |
|---------|------------|-------------|------------|
| beams2d | 1 (compliance) | 50×100 | Single-objective |
| photonics2d | 1 (overlap) | 120×120 | Single-objective |
| **thermoelastic2d** | **3 (struct, thermal, volume)** | **64×64** | **Multi-objective** |

## References

- [EngiBench Documentation](https://engibench.ethz.ch/problems/thermoelastic2d/)
- [Dataset on HuggingFace](https://huggingface.co/datasets/IDEALLab/thermoelastic_2d_v0)
- Generic Scorer: `benchmarks/shared/generic_scorer.py`
- Problem Registry: `benchmarks/shared/problem_registry.py`
