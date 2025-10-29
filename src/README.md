# Engineer Assistant - Code Structure

This document explains the modular structure of the engineer assistant codebase.

## Directory Structure

```
src/
├── agents/                      # Multi-agent system implementations
│   ├── __init__.py
│   ├── supervisor_agent.py      # Main supervisor coordinating all agents
│   ├── engineering_agent.py     # Engineering design & optimization
│   ├── search_agent.py          # Web search capabilities
│   ├── hpc_agent.py            # HPC/SLURM job management
│
├── cli/                        # Command-line interfaces
│   ├── __init__.py
│   └── chat.py                 # Interactive chat interface
│
├── models/                     # Data models and state definitions
│   ├── __init__.py
│   └── state.py                # State schemas for agents
│
├── tools/                      # Custom tools for agents
│   ├── __init__.py
│   ├── search.py               # Web search via Tavily
│   ├── engibench.py            # Engineering benchmarks (beams, etc.)
│   ├── engiopt.py              # ML model training & optimization
│   ├── stl_export.py           # CAD file conversion to STL
│   ├── hpc.py                  # HPC cluster operations
│   ├── connection.py           # SSH/HPC connection management
│   └── job_monitor.py          # SLURM job monitoring & notifications
│
├── ui/                         # Web-based user interfaces
│   ├── __init__.py
│   ├── README.md               # UI documentation
│   └── streamlit_app.py        # Streamlit chat interface
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── prompts.py              # System prompts for all agents
│
├── __init__.py
├── example.py                  # Template example code
└── main.py                     # CLI entry point
```

## Module Descriptions

### `agents/`
Contains agent implementations using LangGraph for workflow orchestration.

**Current agents:**
- **`SupervisorAgent`**: Main coordinator that routes conversations to specialized agents based on context
- **`EngineeringAgent`**: Handles engineering design, optimization, and benchmarking workflows
- **`SearchAgent`**: Performs web searches and retrieves technical information
- **`HPCAgent`**: Manages HPC cluster operations, SLURM job submission, and monitoring
- **`CodeExecutionAgent`**: Executes Python code snippets (optional)

**Architecture Pattern:**
The system uses a **supervisor-based multi-agent architecture**:
1. User sends message to `SupervisorAgent`
2. Supervisor analyzes the request and routes to appropriate specialized agent
3. Specialized agent uses its tools to complete the task
4. Results flow back through supervisor to user

**Adding a new agent:**
```python
# src/agents/new_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.tools.new_tool import new_tool

class NewAgent:
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4")
        self.agent = create_react_agent(llm, tools=[new_tool])

    def invoke(self, state):
        return self.agent.invoke(state)
```

### `tools/`
Contains LangChain tools that agents can use. Each tool is decorated with `@tool`.

**Current tools:**
- **`search.py`**: Web search using Tavily API for retrieving technical information
- **`engibench.py`**: Engineering benchmark datasets (beams2d, cantilever, etc.)
- **`engiopt.py`**: ML model training and optimization (GANs, CNNs) with W&B tracking
- **`stl_export.py`**: Convert 2D heatmaps to 3D STL files for CAD/printing
- **`hpc.py`**: HPC cluster operations (submit jobs, transfer files, run commands)
- **`connection.py`**: SSH connection management for remote HPC systems
- **`job_monitor.py`**: SLURM job monitoring with status updates and notifications

**Adding a new tool:**
```python
# src/tools/new_tool.py
from langchain_core.tools import tool

@tool
def new_tool_function(param: str) -> str:
    """Brief description of what this tool does.

    Args:
        param: Description of parameter

    Returns:
        Description of return value
    """
    # Implementation here
    return "result"
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

### `ui/`
Web-based user interfaces for interacting with the agent system.

**Current UIs:**
- `streamlit_app.py`: Full-featured Streamlit chat interface with:
  - Multi-page navigation (Chat / W&B Report viewing)
  - Interactive model selection (OpenAI GPT-4, Claude 3, etc.)
  - Session info display (current model, message count)
  - 3D STL file viewer for generated CAD models
  - Consolidated settings (media saving, viewer options)
  - W&B training report embedding

**Running the UI:**
```bash
# Option 1: Direct streamlit command
streamlit run src/ui/streamlit_app.py

# Option 2: Using the provided script
./run_ui.sh
```

### `cli/`
Command-line interfaces for interacting with agents (alternative to web UI).

**Current CLIs:**
- `chat.py`: Interactive terminal-based conversation interface

**Running CLI:**
```bash
python -m src.cli.chat
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

**Primary Interface (Recommended):**
```bash
# Run Streamlit web UI
streamlit run src/ui/streamlit_app.py

# Or use the convenience script
./run_ui.sh
```

**Alternative CLI Interface:**
```bash
# Run terminal-based chat
python -m src.cli.chat
```

### Web UI Features
The Streamlit interface (`src/ui/streamlit_app.py`) provides:
- **Chat Interface**: Conversational interaction with the multi-agent system
- **Model Selection**: Choose from 7 AI models (GPT-4, Claude 3, etc.) via dropdown
- **W&B Reports**: View training reports and experiment tracking
- **3D Viewer**: Visualize generated STL files directly in browser
- **Session Management**: Track current model and conversation history

### Using Components Programmatically

```python
# Use the supervisor agent (recommended)
from src.agents.supervisor_agent import SupervisorAgent
from langchain_core.messages import HumanMessage

agent = SupervisorAgent()
result = agent.graph.invoke({
    "messages": [HumanMessage(content="Train a beam optimization model")]
})
print(result["messages"][-1].content)
```

```python
# Use a specific specialized agent directly
from src.agents.engineering_agent import EngineeringAgent

agent = EngineeringAgent()
result = agent.invoke({
    "messages": [HumanMessage(content="Load the beams2d dataset")]
})
```

```python
# Use HPC agent for cluster operations
from src.agents.hpc_agent import HPCAgent

agent = HPCAgent()
result = agent.invoke({
    "messages": [HumanMessage(content="Submit training job to HPC cluster")]
})
```

```python
# Use individual tools directly
from src.tools.search import tavily_search
from src.tools.engibench import load_dataset

# Search for information
search_result = tavily_search.invoke({"query": "SLURM job optimization"})

# Load engineering dataset
dataset = load_dataset.invoke({"dataset_name": "beams2d"})
```

## Adding New Features

### 1. Add a New Tool

1. Create file in `src/tools/` (e.g., `fem_analysis.py`)
2. Define tools with `@tool` decorator:
   ```python
   from langchain_core.tools import tool

   @tool
   def run_fem_analysis(mesh_file: str) -> dict:
       """Run FEM analysis on a mesh file.

       Args:
           mesh_file: Path to mesh file

       Returns:
           Analysis results with stress/strain data
       """
       # Implementation
       return {"max_stress": 1234.5}
   ```
3. Export from `src/tools/__init__.py`
4. Add to appropriate agent's tool list in `src/agents/`

### 2. Add a New Agent

1. Create file in `src/agents/` (e.g., `mechanical_agent.py`)
2. Use LangGraph's `create_react_agent`:
   ```python
   from langchain_openai import ChatOpenAI
   from langgraph.prebuilt import create_react_agent
   from src.tools.fem_analysis import run_fem_analysis

   class MechanicalAgent:
       def __init__(self):
           llm = ChatOpenAI(model="gpt-4")
           self.agent = create_react_agent(
               llm,
               tools=[run_fem_analysis],
               name="mechanical_agent"
           )

       def invoke(self, state):
           return self.agent.invoke(state)
   ```
3. Add to supervisor routing in `src/agents/supervisor_agent.py`
4. Export from `src/agents/__init__.py`

### 3. Add a New UI Feature

1. Edit `src/ui/streamlit_app.py`
2. Add new sidebar section or page:
   ```python
   with st.expander("🔧 New Feature"):
       setting = st.checkbox("Enable feature")
       if setting:
           # Feature implementation
   ```
3. Update session state management if needed
4. Document in `src/ui/README.md`

## Best Practices

1. **Modular tools** - One focused tool per file
2. **Specialized agents** - Each agent handles a specific domain
3. **Type hints** - Use everywhere for better IDE support
4. **Comprehensive docstrings** - Include Args, Returns, Examples
5. **Configuration via .env** - Never hardcode credentials
6. **Test your changes** - Add tests in `tests/` mirroring structure
7. **Update documentation** - Keep this README current

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_example.py
pytest tests/test_optimization_workflow.py
pytest tests/test_notebook_comparison.py

# Type checking
mypy src/

# Linting
ruff check src/
```

## Configuration

The system uses environment variables for configuration. Copy `.env.example` to `.env` and configure:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
WANDB_API_KEY=...

# Model Selection
LLM_MODEL=gpt-4o  # or gpt-4-turbo, claude-3-opus-20240229, etc.

# HPC Configuration
HPC_HOST=cluster.university.edu
HPC_USERNAME=your_username
HPC_PRIVATE_KEY_PATH=/path/to/ssh/key

# W&B Integration
WANDB_REPORT_URL=https://wandb.ai/your-project/...
WANDB_PROJECT=engineer-assistant
WANDB_ENTITY=your-username
```

## Architecture Overview

```
User Request
    ↓
SupervisorAgent (Coordinator)
    ├→ EngineeringAgent (Design/Optimization)
    │   └→ Tools: engibench, engiopt, stl_export
    ├→ SearchAgent (Information Retrieval)
    │   └→ Tools: tavily_search
    ├→ HPCAgent (Cluster Management)
    │   └→ Tools: hpc, connection, job_monitor
    └→ CodeExecutionAgent (Code Execution)
        └→ Tools: python_repl
    ↓
Response to User
```

The supervisor analyzes each request and routes it to the appropriate specialized agent, which uses its tools to complete the task.
