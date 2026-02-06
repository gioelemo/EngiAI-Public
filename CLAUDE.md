# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Engineer Assistant is an AI-powered multi-agent system for mechanical engineering design, featuring topology optimization, ML-based design generation, HPC job management, and 3D printing integration. Built with LangChain/LangGraph.

## Development Commands

```bash
# Install dependencies
make install              # pip install -e .[dev]
pip install -e ".[all]"   # Install all optional deps (test, docs)

# Run tests
pytest                              # All tests
pytest -m "not slow"               # Fast tests only
pytest tests/test_agents/          # Specific directory
pytest --cov=src --cov-report=html # With coverage

# Linting and formatting
make lint                 # ruff check + mypy
make format               # ruff format + ruff check --fix
ruff check --fix .        # Auto-fix lint issues

# Run the application
make run-ui               # Start Streamlit UI (localhost:8501)
make docker-up            # Start all Docker services
make docker-rebuild       # Rebuild and restart Docker services

# Documentation
make docs                 # Build docs (auto-generates API docs)
make docs-watch           # Live preview with auto-rebuild
```

## Architecture

### Multi-Agent System (Supervisor Pattern)

The system uses a hierarchical supervisor pattern where `SupervisorAgent` routes requests to specialized agents:

- **EngineeringAgent** - Topology optimization via EngiBench/EngiOpt, STL export
- **RAGAgent** - Document Q&A using MMORE multimodal RAG service
- **ArXivAgent** - Research paper search and analysis
- **SearchAgent** - Web search via Tavily API
- **HPCAgent** - SLURM job submission on Euler cluster
- **CLIAgent** - Local command execution (requires confirmation)
- **PrusaAgent** - 3D printer control via Prusa MCP server

### Core Components

```
src/
├── agents/          # Agent implementations (BaseAgent pattern)
├── tools/           # Tool modules (engibench, engiopt, hpc, rag_tools, etc.)
├── models/          # LangGraph state definitions
├── utils/           # Prompts, API tracking utilities
├── ui/              # Streamlit web interface
└── checkpoint.py    # PostgreSQL/SQLite persistence
```

### Key Patterns

- **LangGraph State Machines**: Each agent is a DAG with `llm_call` → `tool_node` → conditional routing
- **Tool Binding**: LLMs receive tool definitions via `llm.bind_tools(tools)`
- **Human-in-the-Loop**: SupervisorAgent interrupts before CLI execution
- **Factory Pattern**: `create_rag_tools()`, `create_search_tool()`, `create_arxiv_tools()`

### External Services

- **MMORE**: Multimodal RAG service at `MMORE_RAG_URL` (default: localhost:8000)
- **Prusa MCP**: 3D printer control server (optional, set `SKIP_MCP=true` to disable)
- **PostgreSQL**: Conversation persistence (fallback: SQLite)

## Testing

Uses `GenericFakeChatModel` from LangChain for mocking LLM calls:

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

@pytest.mark.unit
def test_something():
    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    with patch("src.agents.your_agent.init_chat_model", return_value=fake_llm):
        # test code
```

Markers: `@pytest.mark.unit`, `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.requires_api`

## Configuration

Required environment variables (see `.env.example`):
- `OPENAI_API_KEY`, `GOOGLE_API_KEY`,`TAVILY_API_KEY` - Required API keys
- `LLM_MODEL` - Model spec (default: `openai:gpt-4o`)
- `MMORE_RAG_URL` - RAG service URL
- `DATABASE_URL` - Persistence (PostgreSQL or SQLite)

Optional feature flags:
- `SKIP_MCP=true` - Disable Prusa integration
- `SKIP_MMORE=true` - Disable RAG (graceful degradation)
- `USE_WEAVE=true` - Enable Weave tracing for benchmarks

## Benchmarks

Evaluation scripts in `benchmarks/evaluations/`:
```bash
# Run agent evaluation
make mmore-eval-up                                    # Start eval service
make mmore-eval-run ARGS="--problem beams2d --samples 1"
```

### Prompt Styles

The beams2d benchmark supports multiple prompt styles for evaluating different aspects of agent behavior:

- **full**: Exact numerical parameters
- **approximate**: Rounded/approximate values
- **natural**: Natural language descriptions only
- **workflow**: Full workflow with STL export (hardcoded parameters)
- **workflow-random**: Full workflow with randomized STL parameters and validation

#### workflow-random Prompt Style

The `workflow-random` style extends the `workflow` style by randomizing STL export parameters and validating that the agent uses the correct values. This tests the agent's ability to follow precise numerical instructions for 3D printing parameters.

**Randomized Parameters:**
- `mirror_y`: Boolean (True/False) - whether to mirror along Y-axis
- `scale_xy`: Float (0.5-5.0) - X/Y dimension scaling factor
- `scale_z`: Float (5.0-20.0) - Z extrusion height
- `threshold`: Float (0.3-0.7) - Density threshold for solid/void conversion

**Validation:**
The scorer validates that the STL export tool was called with parameters matching the prompt (within tolerance of ±0.01 for floats, exact match for booleans). Task completion requires both successful STL export AND parameter validation.

**Usage:**
```bash
# Generate workflow-random prompts
cd benchmarks/problems/beams2d
python generate_prompts.py --samples 5 --style workflow-random --seed 42

# Run evaluation
python benchmarks/evaluations/evaluate_agent.py \
    --problem beams2d --samples 5 --prompt-style workflow-random --seed 42

# Full benchmark with multiple seeds
python benchmarks/evaluations/run_full_benchmark.py \
    --problem beams2d --samples 10 --seeds 1 2 3 \
    --prompt-style workflow-random --agent-only
```

**Validation Metrics:**
Results include per-parameter validation metrics:
- `stl_param_validation_score`: 1.0 if all params valid, 0.0 otherwise
- `stl_param_violations`: Count of parameter mismatches
- `stl_{param}_actual/expected/error/valid`: Per-parameter details

Example prompt excerpt:
```
2. Post-processing & Export
   - Thresholding: Apply a 0.58 density threshold to convert...
   - Mirror: Mirror the design across the y-axis for the final geometry
   - XY Scaling: Scale the X and Y dimensions by 2.47
   - Extrusion: Extrude the 2D result by 17.9 units in the Z-axis...
   - Export: Save the final geometry as an STL file with these exact parameters
```
