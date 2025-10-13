# Problem Information Features

## Overview

The engineering assistant now has enhanced capabilities to retrieve detailed information directly from EngiBench problem objects:

1. **Problem Details** (`get_problem_details`) - Design space, objectives, and conditions
2. **Dataset Information** (`get_dataset_info`) - Dataset splits, features, and sample counts

## 1. Problem Details (`get_problem_details`)

### What Information is Available?

This tool extracts information directly from `problem.design_space`, `problem.objectives`, and `problem.conditions`:

- **Design Space**: The geometric space where designs exist
  - Shape: (50, 100) for 2D beam problems
  - Bounds: [0.0, 1.0] representing material density
- **Objectives**: What the optimization minimizes/maximizes
  - Example: `('c', 'MINIMIZE')` - minimize compliance (maximize stiffness)
- **Conditions**: Parameters that can be varied
  - `volfrac`: Volume fraction (default 0.35)
  - `rmin`: Minimum radius (default 2.0)
  - `forcedist`: Force distribution (default 0.0)
  - `overhang_constraint`: Overhang constraint flag (default False)
- **Dataset ID**: HuggingFace dataset identifier

### How to Use from Chatbot

Ask natural language questions like:
- "What is the design space of the beams2d problem?"
- "What are the objectives for beam optimization?"
- "What conditions can I set for a beam problem?"
- "Tell me about the design space shape and bounds"

### Example Output

```
Design Space: Box(0.0, 1.0, (50, 100), float64)
- Shape: (50, 100)
- Bounds: 0.0 to 1.0

Objectives:
- Minimize compliance (c)

Conditions:
- volfrac: 0.35 (volume fraction)
- rmin: 2.0 (minimum radius)
- forcedist: 0.0 (force distribution)
- overhang_constraint: False
```

## 2. Dataset Information (`get_dataset_info`)

### What Information is Available?

The `get_dataset_info` tool provides:

1. **Dataset ID**: The HuggingFace dataset identifier (e.g., `IDEALLab/beams_2d_50_100_v0`)
2. **Dataset Splits**: Information about train/validation/test splits
   - Number of rows in each split
   - Number of columns (features) in each split
3. **Total Samples**: Total number of samples across all splits
4. **Features**: List of available features in the dataset:
   - `optimal_design`: The optimized design array
   - `volfrac`: Volume fraction parameter
   - `rmin`: Minimum radius parameter
   - `forcedist`: Force distribution parameter
   - `overhang_constraint`: Overhang constraint flag
   - `c`: Compliance value (objective)
   - `optimization_history`: History of the optimization process

## How to Use

### From the Chatbot

Simply ask the assistant about the dataset in natural language:

```
User: "What information is available about the dataset?"
User: "Tell me about the EngiBench dataset"
User: "How many samples are in the beam dataset?"
```

The assistant will automatically call the `get_dataset_info` tool and provide you with detailed information.

### Example Output

When you ask about the dataset, you'll get information like:

```
Dataset ID: IDEALLab/beams_2d_50_100_v0

Splits:
  train: 3880 rows, 7 columns
  val: 728 rows, 7 columns
  test: 243 rows, 7 columns

Total samples: 4851

Features: optimal_design, volfrac, rmin, forcedist, overhang_constraint, c, optimization_history
```

## Technical Details

### Tool Implementation

The tool is implemented in `src/tools/engibench.py`:

```python
@tool
def get_dataset_info(problem_type: str = "beams2d") -> dict[str, Any]:
    """Get information about the EngiBench dataset for a problem."""
    # ... implementation
```

### Agent Integration

The tool is available through the **EngineeringAgent**, which means:
- You can access it directly when using the engineering agent
- The supervisor agent can delegate dataset queries to the engineering agent

### Dataset Source

The dataset is accessed via the EngiBench library:
```python
from engibench.problems.beams2d.v0 import Beams2D

problem = Beams2D()
dataset = problem.dataset  # HuggingFace DatasetDict
```

## Use Cases

1. **Research**: Understanding the scale and structure of benchmark datasets
2. **Learning**: Exploring what data is available for training/testing
3. **Development**: Checking dataset features before implementing new algorithms
4. **Documentation**: Getting quick reference information about dataset composition

## Testing

Run the test script to see the feature in action:

```bash
python test_dataset_info.py
```

This will demonstrate how the agent responds to various queries about dataset information.

## Available Problem Types

Currently supported:
- `beams2d`: 2D beam topology optimization problems

More problem types can be added in the future as EngiBench expands.
