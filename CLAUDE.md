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
- **natural**: Natural language descriptions only
- **workflow**: Full workflow with STL export (hardcoded parameters)
- **workflow-random**: Full workflow with randomized STL parameters and validation
- **workflow-derived-params**: Workflow with STL parameters derived from optimization inputs
- **workflow-distractor**: Workflow with distractor parameters mixed with real STL params
- **workflow-conditional**: Workflow with if/then branching based on simulation results

#### workflow-random Prompt Style

The `workflow-random` style extends the `workflow` style by randomizing STL export parameters and validating that the agent uses the correct values. This tests the agent's ability to follow precise numerical instructions for 3D printing parameters.

**Randomized Parameters:**
- `mirror_y`: Boolean (True/False) - whether to mirror along Y-axis
- `scale_xy`: Float (0.5-5.0) - X/Y dimension scaling factor
- `scale_z`: Float (5.0-20.0) - Z extrusion height
- `threshold`: Float (0.3-0.7) - Density threshold for solid/void conversion

**Validation:**
The scorer validates that the STL export tool was called with parameters matching the prompt (within tolerance of ±0.05 for floats, exact match for booleans). Task completion requires both successful STL export AND parameter validation.

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

#### workflow-derived-params Prompt Style

The `workflow-derived-params` style tests **arithmetic reasoning** — the agent must compute STL export parameters from the optimization inputs rather than following explicit values. The prompt gives derivation rules, not final numbers.

**Derivation Rules:**
- `threshold` = volume fraction value (e.g., volfrac=0.35 → threshold=0.35)
- `scale_xy` = 2 × filter radius (e.g., rmin=1.5 → scale_xy=3.0)
- `scale_z` = threshold × 40 (e.g., 0.35 × 40 = 14.0)
- `mirror_y` = True only if volume fraction > 0.4

**Key Design:** No randomness — parameters are deterministically derived from optimization inputs. No seed needed.

**Validation:** Same ±0.05 float tolerance and exact boolean match as workflow-random.

**Usage:**
```bash
# Generate workflow-derived-params prompts
cd benchmarks/problems/beams2d
python generate_prompts.py --samples 5 --style workflow-derived-params

# Run evaluation
python benchmarks/evaluations/evaluate_agent.py \
    --problem beams2d --samples 5 --prompt-style workflow-derived-params
```

Example prompt excerpt:
```
2. Post-processing & Export
   The STL export parameters must be derived from the optimization inputs:
   - Thresholding: Use the volume fraction value as the density threshold
   - XY Scaling: Scale the X and Y dimensions by twice the filter radius
   - Extrusion: Extrude the 2D result in the Z-axis by the threshold value multiplied by 40
   - Mirror: Mirror the design across the y-axis only if the volume fraction is greater than 0.4
```

#### workflow-distractor Prompt Style

The `workflow-distractor` style tests **parameter filtering** — the agent must distinguish valid STL tool parameters from plausible but irrelevant distractors mixed into the prompt. No hints are provided about which parameters are distractors; the agent must consult the `convert_design_to_stl` tool schema to determine which parameters it actually accepts.

**Distractor Parameters** (randomized, presented without hints):
- `smoothing_sigma`: Float (0.5-3.0) — Gaussian smoothing
- `infill_density`: Int (10-50%) — infill density percentage
- `layer_height`: Float (0.1-0.3mm) — layer height

**Why it breaks LLMs:** If the agent passes extra parameters to `convert_design_to_stl`, LangChain's schema validation rejects the tool call entirely — the tool never executes and the task fails. Without hints, the agent must reason about the tool schema to filter correctly.

**Real Parameters:** Same 4 randomized STL params as workflow-random (threshold, scale_xy, scale_z, mirror_y).

**Validation:** Same ±0.05 float tolerance and exact boolean match as workflow-random. Only the 4 real params are validated.

**Usage:**
```bash
# Generate workflow-distractor prompts
cd benchmarks/problems/beams2d
python generate_prompts.py --samples 5 --style workflow-distractor --seed 42

# Run evaluation
python benchmarks/evaluations/evaluate_agent.py \
    --problem beams2d --samples 5 --prompt-style workflow-distractor --seed 42
```

Example prompt excerpt:
```
2. Post-processing & Export
   - Thresholding: Apply a 0.53 density threshold to convert the continuous density map into binary geometry
   - Smoothing: Apply Gaussian smoothing with sigma=2.0
   - Mirror: Mirror the design across the y-axis for the final geometry
   - XY Scaling: Scale the X and Y dimensions by 1.88
   - Infill: Use 20% infill density
   - Extrusion: Extrude the 2D result by 12.4 units in the Z-axis to create a 3D volume
   - Layer Height: Use 0.15mm layer height
   - Export: Save the final geometry as an STL file with all the applicable parameters listed above
```

#### workflow-conditional Prompt Style

The `workflow-conditional` style extends `workflow-random` by adding if/then branching logic that depends on the simulation result. The agent must read the compliance value from `simulate_design`, compare it against a threshold, and select the correct parameter set. This tests tool-output → reasoning → tool-input chaining.

**Flow:**
1. Optimize the design with given parameters
2. Simulate the design to get compliance
3. Compare compliance against a randomized threshold
4. Apply the correct branch-specific parameters (threshold, mirror_y)
5. Apply common parameters (scale_xy, scale_z) regardless of branch
6. Export as STL

**Randomized Parameters:**
- `compliance_threshold`: Float (100-300) - the branching condition
- `branch_high.threshold` / `branch_low.threshold`: Float (0.3-0.7) - density thresholds (guaranteed ≥0.1 apart)
- `branch_high.mirror_y` / `branch_low.mirror_y`: Boolean - mirror settings (guaranteed opposite)
- `common.scale_xy`: Float (0.5-5.0) - X/Y scaling (both branches)
- `common.scale_z`: Float (5.0-20.0) - Z extrusion height (both branches)

**Validation:**
The scorer uses ground truth compliance from the dataset to determine the correct branch, then validates STL parameters as in workflow-random (±0.05 for floats, exact match for booleans).

**Usage:**
```bash
# Generate workflow-conditional prompts
cd benchmarks/problems/beams2d
python generate_prompts.py --samples 5 --style workflow-conditional --seed 42

# Run evaluation
python benchmarks/evaluations/evaluate_agent.py \
    --problem beams2d --samples 5 --prompt-style workflow-conditional --seed 42
```

**Additional Validation Metrics:**
- `conditional_compliance_threshold`: The randomized branching threshold
- `conditional_gt_compliance`: Ground truth compliance from the dataset
- `conditional_correct_branch`: Which branch was correct ("high" or "low")

Example prompt excerpt:
```
3. Post-processing & Export (conditional on compliance)
   - If compliance > 213.4:
     - Thresholding: Apply a 0.42 density threshold
     - Mirror: Mirror the design across the y-axis
   - If compliance <= 213.4:
     - Thresholding: Apply a 0.61 density threshold
     - Mirror: Do NOT mirror the design
   - In both cases:
     - XY Scaling: Scale the X and Y dimensions by 3.14
     - Extrusion: Extrude the 2D result by 11.7 units in the Z-axis
   - Export: Save the final geometry as an STL file with these exact parameters
```

#### workflow-multi-export Prompt Style

The `workflow-multi-export` style requires the agent to call `convert_design_to_stl` **twice** with completely different parameter sets from the same optimization. This tests working memory and instruction tracking — LLMs commonly merge the two exports into one, swap parameters between them, or only do one export.

**Flow:**
1. Optimize the design with given parameters
2. Simulate the design
3. Export STL with Export A parameters
4. Export STL with Export B parameters (different from A)

**Randomized Parameters (per export):**
- `mirror_y`: Boolean - guaranteed opposite between A and B
- `threshold`: Float (0.3-0.7) - gap ≥ 0.1 between A and B
- `scale_xy`: Float (0.5-5.0) - gap ≥ 0.2 between A and B
- `scale_z`: Float (5.0-20.0) - gap ≥ 0.2 between A and B

**Validation:**
The scorer validates both STL calls **in order** (first call → Export A, second call → Export B). All-or-nothing: both exports must have correct parameters for task completion. Uses the same ±0.05 float tolerance and exact boolean match as workflow-random.

**Usage:**
```bash
# Generate workflow-multi-export prompts
cd benchmarks/problems/beams2d
python generate_prompts.py --samples 5 --style workflow-multi-export --seed 42

# Run evaluation
python benchmarks/evaluations/evaluate_agent.py \
    --problem beams2d --samples 5 --prompt-style workflow-multi-export --seed 42
```

**Validation Metrics:**
- `multi_export_count`: Number of successful STL calls detected
- `multi_export_both_valid`: Whether both exports passed validation
- `export_a_stl_{param}_actual/expected/error/valid`: Per-parameter details for Export A
- `export_b_stl_{param}_actual/expected/error/valid`: Per-parameter details for Export B
