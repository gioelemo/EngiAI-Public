# Engineer Assistant - Code Structure

This document explains the modular structure of the engineer assistant codebase.

## Directory Structure

```
src/
├── agents/                     # Multi-agent system implementations
│   ├── __init__.py
│   ├── cli_agent.py            # CLI agent
│   ├── engineering_agent.py    # Engineering design & optimization
│   ├── hpc_agent.py            # HPC/SLURM job management
│   ├── search_agent.py         # Web search capabilities
│   └── supervisor_agent.py     # Main supervisor coordinating all agents
│
├── cli/                        # Command-line interfaces (DEPRECATED)
│   └── chat.py                 # Interactive chat interface (use Streamlit UI instead)
│
├── models/                     # Data models and state definitions
│   ├── __init__.py
│   └── state.py                # State schemas for agents
│
├── tools/                      # Custom tools for agents
│   ├── __init__.py
│   ├── cli.py                  # CLI tools
│   ├── connection.py           # SSH/HPC connection management
│   ├── engibench.py            # Engineering benchmarks (beams, etc.)
│   ├── engiopt.py              # ML model training & optimization
│   ├── hpc.py                  # HPC cluster operations
│   ├── job_monitor.py          # SLURM job monitoring & notifications
│   ├── search.py               # Web search via Tavily
│   └── stl_export.py           # CAD file conversion to STL
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
└── main.py                     # DEPRECATED: Shows migration message
```

## Module Descriptions

### `agents/`
Contains agent implementations using LangGraph for workflow orchestration.

**Current agents:**
- **`SupervisorAgent`**: Main coordinator that routes conversations to specialized agents based on context
- **`EngineeringAgent`**: Handles engineering design, optimization, and benchmarking workflows
- **`SearchAgent`**: Performs web searches and retrieves technical information
- **`HPCAgent`**: Manages HPC cluster operations, SLURM job submission, and monitoring
- **`CLIAgent`**: Manage Command Line commands

**Architecture Pattern:**
The system uses a **supervisor-based multi-agent architecture**:
1. User sends message to `SupervisorAgent`
2. Supervisor analyzes the request and routes to appropriate specialized agent
3. Specialized agent uses its tools to complete the task
4. Results flow back through supervisor to user

### `tools/`
Contains LangChain tools that agents can use. Each tool is decorated with `@tool`.

**Current tools:**
- **`cli.py`**: CLI commands
- **`connection.py`**: SSH connection management for remote HPC systems
- **`engibench.py`**: Engineering benchmark (beams2d)
- **`engiopt.py`**: ML model training and optimization (GANs, CNNs) with W&B tracking
- **`hpc.py`**: HPC cluster operations (submit jobs, transfer files, run commands)
- **`job_monitor.py`**: SLURM job monitoring with status updates and notifications
- **`search.py`**: Web search using Tavily API for retrieving technical information
- **`stl_export.py`**: Tools for generating the STL model from the results of the optimization


### `models/`
Defines state schemas and custom types used throughout the application.

### `ui/`
Web-based user interfaces for interacting with the agent system.

**Current UIs:**
- `streamlit_app.py`: Full-featured Streamlit chat interface with:
  - Multi-page navigation (Chat / W&B Report viewing)
  - 3D STL file viewer for generated CAD models
  - Consolidated settings (media saving, viewer options)
  - W&B training report embedding

**Running the UI:**
```bash
# Option 1: Using Makefile (recommended)
make run-ui

# Option 2: Direct streamlit command
streamlit run src/ui/streamlit_app.py
```

### `cli/`
⚠️ **DEPRECATED** - Use the Streamlit web UI instead (`src/ui/streamlit_app.py`).

The CLI interface has been deprecated in favor of the more feature-rich and user-friendly Streamlit web interface. The code is kept for reference but should not be used in production.

### `utils/`
Utility functions and constants.

**Current utilities:**
- `prompts.py`: System prompts for different agents

## Usage

### Running the Application

**Primary Interface:**
```bash
# Run Streamlit web UI
make run-ui

# Or directly with streamlit
streamlit run src/ui/streamlit_app.py
```

### Web UI Features
The Streamlit interface (`src/ui/streamlit_app.py`) provides:
- **Chat Interface**: Conversational interaction with the multi-agent system
- **W&B Reports**: View training reports and experiment tracking
- **3D Viewer**: Visualize generated STL files directly in browser
- **Session Management**: Track current model and conversation history

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
