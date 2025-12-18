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

### Automatic Tracing

**Important:** After calling `init_weave()`, all LangChain components are automatically traced. No decorators or manual instrumentation needed!

Weave automatically captures:
- All LangChain agent executions
- LLM calls (OpenAI, Anthropic, etc.)
- Chain invocations
- Tool usage
- Retriever queries

```python
from src.utils.weave_integration import init_weave
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

# Initialize Weave once at startup
init_weave()

# All LangChain operations are now automatically traced
llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("What is the capital of France?")  # Automatically traced!

# Agent executions are also traced
agent = create_agent(llm, tools)
result = agent.invoke({"input": "Design a beam"})  # Fully traced!

## Creating Evaluation Datasets

Create datasets for benchmarking using Weave's native API:

```python
import weave
from src.utils.weave_integration import init_weave

# Initialize Weave
init_weave()

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

# Publish the dataset to Weave
dataset = weave.Dataset(name="engineering_qa_benchmark_v1", rows=benchmark_data)
weave.publish(dataset)
```

## Running Evaluations

Evaluate your agents using Weave's evaluation framework:

```python
import weave
from src.utils.weave_integration import init_weave
from src.agents.engineering_agent import engineering_agent

# Initialize Weave
init_weave()

# Define evaluation function
@weave.op()
def evaluate_engineering_qa(input: str, expected_output: str) -> dict:
    """Evaluate engineering agent response."""
    response = engineering_agent.invoke({"input": input})

    # Calculate metrics (example)
    return {
        "response": response["output"],
        "matches_expected": expected_output.lower() in response["output"].lower()
    }

# Load dataset and run evaluation
dataset = weave.ref("engineering_qa_benchmark_v1").get()
evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[evaluate_engineering_qa]
)

results = evaluation.evaluate(engineering_agent)
print(f"Evaluation results: {results}")
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

Weave automatically traces all agent workflows after initialization - no code changes needed:

```python
from src.utils.weave_integration import init_weave
from src.agents.engineering_agent import engineering_agent

# Initialize Weave once at application startup
init_weave()

# All agent operations are now automatically traced
result = engineering_agent.invoke({"input": "Design a beam with 30% volume fraction"})
# ✅ Automatically traced - no decorators needed!

# Multi-agent workflows are also traced
from src.agents.supervisor import supervisor_agent
result = supervisor_agent.invoke({"messages": [HumanMessage(content="Optimize this structure")]})
# ✅ Full execution trace captured automatically!
```

### Custom Operations with Weave

For custom operations not automatically traced, use Weave's `@weave.op()` decorator:

```python
import weave

@weave.op()
def custom_optimization(design: dict, constraints: dict) -> dict:
    """Custom optimization function with Weave tracing."""
    # Your custom logic here
    optimized = run_optimization(design, constraints)
    return optimized
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
import weave
from src.utils.weave_integration import init_weave
from src.agents.engineering_agent import engineering_agent

# 1. Initialize Weave
init_weave()

# 2. Create a benchmark dataset
dataset = weave.Dataset(
    name="engineering_qa_v1",
    rows=[
        {"input": "What is Young's modulus?", "expected": "A measure of stiffness"},
        {"input": "Define stress", "expected": "Force per unit area"},
    ]
)
weave.publish(dataset)

# 3. Define evaluation scorer
@weave.op()
def evaluate_answer(input: str, expected: str, output: str) -> dict:
    """Score the agent's response."""
    # Simple keyword matching (replace with your scoring logic)
    keywords = expected.lower().split()
    matches = sum(1 for word in keywords if word in output.lower())
    score = matches / len(keywords)

    return {
        "keyword_match_score": score,
        "response": output
    }

# 4. Run evaluation
evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[evaluate_answer]
)

results = evaluation.evaluate(engineering_agent)

# 5. View results
print(f"Evaluation complete!")
print(f"Results: {results}")
print(f"View full traces at: https://wandb.ai/your-entity/engineer-assistant-benchmarks/weave")
```

## Further Reading

- [Weave Documentation](https://docs.wandb.ai/weave)
- [Weave Quickstart](https://docs.wandb.ai/weave/quickstart)
- [W&B Integration Guide](https://docs.wandb.ai/guides/integrations)
