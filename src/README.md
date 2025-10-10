# Engineer Assistant - Code Structure

This document explains the modular structure of the engineer assistant codebase.

## Directory Structure

```
src/
├── agents/              # Agent implementations
│   ├── __init__.py
│   └── math_agent.py    # Math and search agent
│
├── cli/                 # Command-line interfaces
│   ├── __init__.py
│   └── chat.py          # Interactive chat interface
│
├── models/              # Data models and state definitions
│   ├── __init__.py
│   └── state.py         # State schemas for agents
│
├── tools/               # Custom tools for agents
│   ├── __init__.py
│   ├── arithmetic.py    # Math operations (add, multiply, divide)
│   └── search.py        # Web search capabilities
│
├── utils/               # Utility functions
│   ├── __init__.py
│   └── prompts.py       # System prompts for agents
│
└── main.py              # Main entry point
```

## Module Descriptions

### `agents/`
Contains agent implementations. Each agent is a class that:
- Initializes an LLM with specific tools
- Defines the agent workflow (nodes, edges, logic)
- Provides an `invoke()` method to interact with the agent

**Current agents:**
- `MathAgent`: Handles arithmetic operations and web searches

**Adding a new agent:**
```python
# src/agents/engineering_agent.py
from src.agents.math_agent import MathAgent

class EngineeringAgent(MathAgent):
    def __init__(self):
        super().__init__()
        # Add engineering-specific tools
        self.tools.extend([cad_tool, simulation_tool])
```

### `tools/`
Contains LangChain tools that agents can use. Each tool is decorated with `@tool`.

**Current tools:**
- `arithmetic.py`: add, multiply, divide
- `search.py`: web search via Tavily

**Adding a new tool:**
```python
# src/tools/cad.py
from langchain_core.tools import tool

@tool
def convert_to_stl(data: list[list[float]]) -> str:
    \"\"\"Convert 2D array to STL file.

    Args:
        data: 2D array of height values

    Returns:
        Path to generated STL file
    \"\"\"
    # Implementation here
    return "output.stl"
```

### `models/`
Defines state schemas and custom types used throughout the application.

**Current models:**
- `MessagesState`: State for conversation-based agents

**Adding a new state:**
```python
# src/models/state.py
class EngineeringState(MessagesState):
    cad_files: NotRequired[list[str]]
    simulation_results: NotRequired[dict]
```

### `cli/`
Command-line interfaces for interacting with agents.

**Current CLIs:**
- `chat.py`: Interactive conversation interface

**Adding a new CLI:**
```python
# src/cli/batch.py
from src.agents.math_agent import MathAgent

def batch_process(input_file: str, output_file: str):
    \"\"\"Process a batch of queries from a file.\"\"\"
    agent = MathAgent()
    # Implementation here
```

### `utils/`
Utility functions and constants.

**Current utilities:**
- `prompts.py`: System prompts for different agents

**Adding utilities:**
```python
# src/utils/formatters.py
def format_calculation_result(result: float) -> str:
    \"\"\"Format calculation results nicely.\"\"\"
    return f"Result: {result:.2f}"
```

## Usage

### Running the Application

```bash
# Interactive chat
python -m src.main

# Or directly run the chat CLI
python -m src.cli.chat
```

### Using Components Programmatically

```python
# Use the math agent directly
from src.agents import MathAgent
from src.models import MessagesState
from langchain_core.messages import HumanMessage

agent = MathAgent()
state = MessagesState(messages=[HumanMessage(content="What is 5 + 3?")])
result = agent.invoke(state)
print(result["messages"][-1].content)
```

```python
# Use individual tools
from src.tools.arithmetic import add, multiply

result = add.invoke({"a": 5, "b": 3})
print(result)  # 8
```

## Adding New Features

### 1. Add a New Tool

1. Create file in `src/tools/` (e.g., `visualization.py`)
2. Define tools with `@tool` decorator
3. Export from `src/tools/__init__.py`
4. Add to an agent's tool list

### 2. Add a New Agent

1. Create file in `src/agents/` (e.g., `engineering_agent.py`)
2. Inherit from existing agent or create new class
3. Define custom tools and workflow
4. Export from `src/agents/__init__.py`

### 3. Add a New CLI

1. Create file in `src/cli/` (e.g., `api.py`)
2. Implement interface (REST API, batch processor, etc.)
3. Export from `src/cli/__init__.py`
4. Optionally add entry point in `src/main.py`

## Best Practices

1. **One tool per file** (unless closely related)
2. **Keep agents focused** (single responsibility)
3. **Use type hints** everywhere
4. **Document with docstrings** (Args, Returns, Raises)
5. **Add tests** in `tests/` mirroring the `src/` structure
6. **Update this README** when adding new modules

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_tools/test_arithmetic.py

# Type checking
mypy src/

# Linting
ruff check src/
```

## Migration from Old Code

The original `chatbot_new.py` has been refactored into:
- Tools → `src/tools/`
- State → `src/models/state.py`
- Agent logic → `src/agents/math_agent.py`
- Interactive CLI → `src/cli/chat.py`
- Entry point → `src/main.py`

The old file can be kept for reference or removed once the new structure is validated.
