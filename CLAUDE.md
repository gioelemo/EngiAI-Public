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
- `OPENAI_API_KEY`, `TAVILY_API_KEY` - Required API keys
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
