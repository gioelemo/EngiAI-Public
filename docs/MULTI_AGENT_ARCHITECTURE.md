# Multi-Agent Architecture

This document describes the multi-agent system architecture for the Engineer Assistant.

## Overview

The system uses a **supervisor-worker** pattern where a coordinator agent routes tasks to specialized agents based on their expertise.

## Architecture Diagram

```
┌─────────────────────────┐
│   Supervisor Agent      │
│   (Coordinator/Router)  │
└───────────┬─────────────┘
            │
    ┌───────┼───────┬─────────┐
    │       │       │         │
    ▼       ▼       ▼         ▼
┌──────┐ ┌─────┐ ┌─────┐ ┌────────┐
│Engin-│ │ CAD │ │Sear-│ │ ...    │
│eering│ │Agent│ │ ch  │ │Future  │
│Agent │ │     │ │Agent│ │Agents  │
└──────┘ └─────┘ └─────┘ └────────┘
```

## Agents

### 1. Supervisor Agent
**Role:** Coordinator and router

**Responsibilities:**
- Analyze user requests
- Route tasks to appropriate specialized agents
- Coordinate multi-step workflows
- Synthesize results from multiple agents

**Example:** User asks "Design an optimized beam and make it 3D printable"
- Routes to Engineering Agent → optimize design → saves .npy file
- Routes to CAD Agent → convert .npy to STL → ready for printing

### 2. Engineering Agent
**Role:** Structural optimization and design

**Tools:**
- `create_beam_problem` - Set up optimization problems
- `simulate_beam_design` - Evaluate designs
- `optimize_beam_design` - Run topology optimization
- `render_beam_design` - Visualize designs (saves PNG + NPY)
- `get_problem_info` - Learn about available problems

**Expertise:**
- Topology optimization
- Structural analysis
- Compliance calculations
- Volume fraction constraints

### 3. CAD Agent
**Role:** 3D model conversion and file handling

**Tools:**
- `convert_design_to_stl` - Convert .npy to STL format

**Expertise:**
- STL file generation
- 3D mesh creation
- Scale and parameter adjustments
- Preparing files for 3D printing/CAD software

### 4. Search Agent
**Role:** Web research and information gathering

**Tools:**
- Web search tool

**Expertise:**
- Finding current information
- Research on engineering topics
- Best practices and standards

## Workflow Examples

### Example 1: Simple Single-Agent Task
```
User: "Optimize a 2D beam design"
→ Supervisor routes to Engineering Agent
→ Engineering Agent optimizes and renders
→ Result: PNG image + .npy file
```

### Example 2: Multi-Agent Workflow
```
User: "Create an optimized beam and convert it to STL"
→ Supervisor routes to Engineering Agent
  → Engineering Agent optimizes design
  → Saves beam_design.png and beam_design.npy
→ Supervisor routes to CAD Agent
  → CAD Agent converts beam_design.npy to beam_design.stl
→ Result: PNG, NPY, and STL files ready for 3D printing
```

### Example 3: Research + Implementation
```
User: "What are best practices for topology optimization?"
→ Supervisor routes to Search Agent
  → Search Agent finds information
→ Supervisor provides summary
→ User: "Now implement it"
→ Supervisor routes to Engineering Agent
  → Engineering Agent creates optimized design
```

## Usage

### Start the Multi-Agent System
```bash
python -m src.main supervisor
# or
python -m src.main team
```

### Start Individual Agents
```bash
# Engineering only
python -m src.main engineering

# Math only
python -m src.main math

# Search only
python -m src.main search

# General (math + search)
python -m src.main general
```

## Benefits of Multi-Agent Architecture

1. **Separation of Concerns**: Each agent has a clear, focused responsibility
2. **Scalability**: Easy to add new specialized agents
3. **Maintainability**: Changes to one agent don't affect others
4. **Flexibility**: Supervisor can handle complex multi-step workflows
5. **Reusability**: Individual agents can be used standalone or coordinated

## Adding New Agents

To add a new specialized agent:

1. **Create agent class** in `src/agents/new_agent.py`
   ```python
   class NewAgent:
       def __init__(self, model_name=None):
           self.tools = [...]  # Add tools
           self.graph = self._build_graph()
   ```

2. **Add agent prompt** in `src/utils/prompts.py`
   ```python
   NEW_AGENT_SYSTEM_PROMPT = """..."""
   ```

3. **Register with supervisor** in `src/agents/supervisor_agent.py`
   ```python
   self.new_agent = NewAgent(model_name=self.model_name)
   # Add to workflow
   ```

4. **Update routing logic** in supervisor to include new agent

## Future Extensions

Potential new agents to add:

- **Analysis Agent**: Data visualization, post-processing, statistics
- **Documentation Agent**: Generate reports, export results
- **Materials Agent**: Material properties, selection guidance
- **FEA Agent**: Finite element analysis integration
- **Manufacturing Agent**: Manufacturability checks, cost estimation

## Technical Details

### State Management
- Each agent has its own checkpoint/memory (InMemorySaver)
- Supervisor coordinates state between agents
- Messages pass between agents through supervisor

### LLM Models
- All agents use the same LLM (configurable in `config.py`)
- Each agent has specialized system prompts
- Function calling enables tool use

### Error Handling
- Individual agents handle their own errors
- Supervisor can retry with different agents
- Graceful degradation if agent unavailable

## Best Practices

1. **Clear Task Delegation**: Supervisor should clearly explain which agent is handling what
2. **Context Preservation**: Pass relevant context when switching between agents
3. **Result Synthesis**: Supervisor should summarize multi-agent results
4. **User Feedback**: Inform user about workflow progress
5. **Error Recovery**: Supervisor should handle agent failures gracefully
