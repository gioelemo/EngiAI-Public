# Multi-Agent System - Quick Start Guide

## What is it?

A **multi-agent system** where specialized AI agents work together, coordinated by a supervisor agent. Each agent has specific expertise and tools.

## Quick Start

```bash
# Start the multi-agent system
python -m src.main supervisor

# Example conversation:
You: Optimize a beam design and convert it to STL for 3D printing

# The system will:
# 1. Engineering Agent: Creates and optimizes the design → saves .npy file
# 2. CAD Agent: Converts .npy to STL → ready for 3D printer
```

## Agents Overview

| Agent | Role | Key Tools |
|-------|------|-----------|
| **Supervisor** | Coordinator | Routes tasks, manages workflow |
| **Engineering** | Structural Optimization | `optimize_beam_design`, `simulate_beam_design`, `render_beam_design` |
| **CAD** | 3D Model Conversion | `convert_design_to_stl` |
| **Search** | Web Research | Web search |

## Example Workflows

### 1. Complete Design to Manufacturing
```
Input: "Design an optimized beam and prepare it for 3D printing"

Workflow:
┌─────────────────┐
│ Engineering     │  Optimize design
│ Agent           │  → beam_design.npy
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CAD Agent       │  Convert to STL
│                 │  → beam_design.stl ✅
└─────────────────┘
```

### 2. Research Then Implement
```
Input: "What are best practices for topology optimization? Then create one."

Workflow:
┌─────────────────┐
│ Search Agent    │  Research best practices
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Engineering     │  Apply practices
│ Agent           │  → optimized design
└─────────────────┘
```

### 3. Design Exploration
```
Input: "Show me different beam designs with varying volume fractions"

Workflow:
┌─────────────────┐
│ Engineering     │  Create multiple designs
│ Agent           │  → design_30%.png/npy
│                 │  → design_40%.png/npy
│                 │  → design_50%.png/npy
└─────────────────┘
```

## When to Use Which Mode

### Use **Supervisor** (Multi-Agent) for:
- ✅ Complete workflows (design → STL conversion)
- ✅ Tasks requiring multiple types of expertise
- ✅ Complex projects with multiple steps
- ✅ When you're not sure which agent to use

### Use **Engineering Agent** for:
- ✅ Pure optimization tasks
- ✅ Design simulations
- ✅ Structural analysis
- ✅ Creating visualizations

### Use Individual Agents for:
- ✅ Focused tasks in one domain
- ✅ Faster response (no routing overhead)
- ✅ When you know exactly what you need

## File Outputs

The system generates multiple file formats:

```
beam_design.png  ← Visualization (heatmap)
beam_design.npy  ← Raw design data (NumPy array)
beam_design.stl  ← 3D model (for printing/CAD)
```

## Command Reference

```bash
# Multi-agent system (recommended for complex tasks)
python -m src.main supervisor

# Individual agents
python -m src.main engineering  # Structural optimization
python -m src.main search       # Web research
python -m src.main math         # Arithmetic
python -m src.main general      # Math + Search
```

## Common Use Cases

### 1. Create 3D Printable Part
```
You: Create an optimized beam with 40% material and convert to STL
```

### 2. Design Comparison
```
You: Compare designs with 30%, 40%, and 50% volume fractions
```

### 3. Research and Apply
```
You: What force distribution should I use for a cantilever beam?
     Then optimize one with those settings.
```

### 4. Parameter Exploration
```
You: Show me how changing the volume fraction affects compliance
```

## Tips for Best Results

1. **Be specific**: "Optimize a beam with 35% material, point load at center"
2. **Request complete workflows**: "Design and convert to STL"
3. **Ask for comparisons**: "Show designs at different volume fractions"
4. **Combine research and action**: "Find best practices, then implement"

## Architecture Benefits

- 🎯 **Specialized Expertise**: Each agent is expert in its domain
- 🔄 **Flexible Workflows**: Supervisor handles complex multi-step tasks
- 📦 **Modular Design**: Easy to add new agents
- 🚀 **Scalable**: Add more specialized agents as needed

## Next Steps

1. Try the supervisor mode: `python -m src.main supervisor`
2. Ask for a complete workflow: "Optimize a beam and make it 3D printable"
3. See the generated files: `beam_design.png`, `beam_design.npy`, `beam_design.stl`
4. Explore different parameters and designs

For more details, see [MULTI_AGENT_ARCHITECTURE.md](./MULTI_AGENT_ARCHITECTURE.md)
