# Weave Integration Guide

This guide explains how to use Weights & Biases Weave for LLM tracing and benchmarking in the Engineer Assistant project.

## Overview

Weave is a lightweight toolkit for tracking and evaluating LLM applications. It automatically captures:
- Input and output data from LLM calls
- Latency and token usage
- Model parameters and configurations
- Full execution traces

This integration is configured separately from the existing `engiopt` W&B project to keep benchmark traces isolated.

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# Weave configuration (for LLM tracing and benchmarking)
USE_WEAVE=true
WEAVE_PROJECT="gioelemo-ethz/engineer-assistant-benchmarks"
```

**Note**: The `WEAVE_PROJECT` uses a separate project from the existing `engiopt` project to maintain separation between model training/optimization and LLM benchmarking.

### Configuration Options

- `USE_WEAVE`: Set to `true` to enable Weave tracing, `false` to disable
- `WEAVE_PROJECT`: The W&B project name in the format `entity/project-name`

## Basic Usage

### Initializing Weave

Weave can be initialized automatically or manually:

#### Automatic Initialization

```python
from config import config

# Initialize Weave based on .env configuration
config.setup_weave_tracing()
```

#### Manual Initialization

```python
from src.utils.weave_integration import init_weave

# Initialize Weave
if init_weave():
    print("Weave tracing is active")
else:
    print("Weave tracing is disabled or unavailable")
```

### Tracing Functions

Use the `@traced` decorator to automatically track function inputs and outputs:

```python
from src.utils.weave_integration import traced

@traced
def generate_response(prompt: str, temperature: float = 0.7) -> str:
    """Generate a response using an LLM."""
    # Your LLM call here
    response = llm.invoke(prompt, temperature=temperature)
    return response

# Use the function normally - it will be automatically traced
result = generate_response("What is the capital of France?")
```

### Tracing Async Functions

For async functions, use `@traced_async`:

```python
from src.utils.weave_integration import traced_async

@traced_async
async def async_generate_response(prompt: str) -> str:
    """Async LLM call with tracing."""
    response = await async_llm.ainvoke(prompt)
    return response

# Use with await
result = await async_generate_response("What is 2+2?")
```

### Using Context Managers

For more granular control, use the `WeaveContext` context manager:

```python
from src.utils.weave_integration import WeaveContext

with WeaveContext("my_custom_operation"):
    # Code in this block will be traced
    result = complex_operation()
    processed = post_process(result)
```

## Creating Evaluation Datasets

Create datasets for benchmarking your LLM applications:

```python
from src.utils.weave_integration import create_dataset

# Define your benchmark data
benchmark_data = [
    {
        "input": "What is the Young's modulus of steel?",
        "expected_output": "Approximately 200 GPa",
        "category": "material_properties"
    },
    {
        "input": "Calculate the stress for a force of 1000N on a 10mm² area",
        "expected_output": "100 MPa",
        "category": "stress_calculation"
    },
    {
        "input": "What is the yield strength of aluminum 6061-T6?",
        "expected_output": "Approximately 276 MPa",
        "category": "material_properties"
    }
]

# Create the dataset
dataset = create_dataset(
    name="engineering_qa_benchmark_v1",
    rows=benchmark_data
)
```

## Running Evaluations

Log evaluation results to track performance over time:

```python
from src.utils.weave_integration import log_evaluation, traced

@traced
def evaluate_model(dataset):
    """Evaluate the model on a dataset."""
    predictions = []

    for item in dataset:
        response = generate_response(item["input"])
        predictions.append(response)

    # Calculate scores (example)
    scores = {
        "accuracy": calculate_accuracy(predictions, dataset),
        "avg_latency_ms": calculate_avg_latency(predictions)
    }

    # Log to Weave
    log_evaluation(
        dataset_name="engineering_qa_benchmark_v1",
        predictions=predictions,
        scores=scores
    )

    return predictions, scores
```

## Automatic LLM Provider Instrumentation

Weave automatically instruments calls to major LLM providers without additional configuration:

- **OpenAI**: GPT-3.5, GPT-4, etc.
- **Anthropic**: Claude models
- **Other providers**: Cohere, Google, etc.

Just use your LLM client normally with Weave initialized:

```python
from openai import OpenAI
from src.utils.weave_integration import init_weave

# Initialize Weave
init_weave()

# Use OpenAI normally - calls will be automatically traced
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Viewing Traces

After running your code, Weave will output URLs in the terminal:

```
🍩 https://wandb.ai/gioelemo-ethz/engineer-assistant-benchmarks/weave
```

Click the link to view:
- **Traces**: Full execution traces with timing
- **Calls**: Individual LLM calls with inputs/outputs
- **Objects**: Datasets and evaluation results
- **Costs**: Token usage and estimated costs

## Integration with Existing Code

### Adding to Agent Workflows

Integrate Weave tracing into your existing agent workflows:

```python
from src.utils.weave_integration import traced
from src.agents.engineering_agent import EngineeringAgent

class TracedEngineeringAgent(EngineeringAgent):
    @traced
    def generate_design(self, requirements: str) -> dict:
        """Generate a design with Weave tracing."""
        return super().generate_design(requirements)

    @traced
    def analyze_cad_model(self, model_path: str) -> dict:
        """Analyze CAD model with Weave tracing."""
        return super().analyze_cad_model(model_path)
```

### Conditional Tracing

Check if Weave is enabled before performing Weave-specific operations:

```python
from src.utils.weave_integration import is_weave_enabled

if is_weave_enabled():
    # Perform Weave-specific operations
    log_evaluation(...)
else:
    # Fall back to alternative logging
    print("Weave is disabled, using console logging")
```

## Best Practices

1. **Separate Projects**: Use a separate Weave project for benchmarking vs. model training
2. **Descriptive Names**: Use clear, descriptive names for traced functions and datasets
3. **Consistent Datasets**: Keep benchmark datasets in version control for reproducibility
4. **Regular Evaluation**: Run evaluations regularly to track model performance over time
5. **Cost Monitoring**: Use Weave's cost tracking to monitor API usage and expenses

## Troubleshooting

### Weave Not Installed

If you see "Weave is not installed", install it:

```bash
pip install weave
```

Or use the project dependencies:

```bash
pip install -e .
```

### Tracing Not Working

1. Check that `USE_WEAVE=true` in your `.env` file
2. Verify your `WANDB_API_KEY` is set
3. Ensure you've called `init_weave()` or `config.setup_weave_tracing()`
4. Check the console for error messages

### Project Not Found

Make sure your `WEAVE_PROJECT` follows the format `entity/project-name`:

```bash
WEAVE_PROJECT="your-entity/your-project-name"
```

## Example: Complete Benchmark Pipeline

Here's a complete example of setting up and running a benchmark:

```python
from src.utils.weave_integration import (
    init_weave,
    traced,
    create_dataset,
    log_evaluation
)

# 1. Initialize Weave
init_weave()

# 2. Create a benchmark dataset
dataset = create_dataset(
    name="engineering_qa_v1",
    rows=[
        {"input": "What is Young's modulus?", "expected": "A measure of stiffness"},
        {"input": "Define stress", "expected": "Force per unit area"},
    ]
)

# 3. Define traced evaluation function
@traced
def evaluate_engineering_agent(question: str) -> str:
    """Evaluate the engineering agent's response."""
    from src.agents.engineering_agent import agent
    response = agent.invoke(question)
    return response

# 4. Run evaluation
predictions = []
for item in dataset.rows:
    pred = evaluate_engineering_agent(item["input"])
    predictions.append(pred)

# 5. Calculate metrics
from src.evaluation.metrics import calculate_similarity
scores = {
    "avg_similarity": sum(
        calculate_similarity(pred, item["expected"])
        for pred, item in zip(predictions, dataset.rows)
    ) / len(dataset.rows)
}

# 6. Log results
log_evaluation(
    dataset_name="engineering_qa_v1",
    predictions=predictions,
    scores=scores
)

print(f"Evaluation complete! Scores: {scores}")
```

## Further Reading

- [Weave Documentation](https://docs.wandb.ai/weave)
- [Weave Quickstart](https://docs.wandb.ai/weave/quickstart)
- [W&B Integration Guide](https://docs.wandb.ai/guides/integrations)
