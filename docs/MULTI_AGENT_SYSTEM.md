# Multi-Agent Engineering System

## Overview

The engineer-assistant now features a **unified multi-agent system** that combines all specialized capabilities into one powerful agent.

## Architecture

### Supervisor Agent (Recommended)
The supervisor agent has access to **all tools** from all specialized domains:

```
┌─────────────────────────────────────────────┐
│         SUPERVISOR AGENT                    │
│  (Unified Multi-Domain Assistant)           │
├─────────────────────────────────────────────┤
│                                             │
│  🔧 Engineering Tools:                      │
│    • create_beam_problem                    │
│    • simulate_beam_design                   │
│    • optimize_beam_design                   │
│    • render_beam_design (PNG + .npy)        │
│    • get_problem_info                       │
│                                             │
│  🎨 CAD Tools:                              │
│    • convert_design_to_stl                  │
│                                             │
│  🔍 Research Tools:                         │
│    • TavilySearch (web search)              │
│                                             │
└─────────────────────────────────────────────┘
```

### Individual Specialized Agents
For focused tasks, you can still use individual agents:

- **Engineering Agent**: Structural optimization only
- **CAD Agent**: STL conversion only
- **Search Agent**: Web research only
- **Math Agent**: Arithmetic operations
- **General Agent**: Math + Search

## Complete Workflows

### Example 1: Optimize & 3D Print
User: *"optimize a beam and generate an STL file"*

Workflow:
1. **create_beam_problem** - Set up optimization
2. **optimize_beam_design** - Find optimal topology
3. **render_beam_design** - Save PNG image + .npy array
4. **convert_design_to_stl** - Convert .npy → STL for 3D printing

Output: Design image, raw data, 3D printable file

### Example 2: Research-Informed Design
User: *"what's the best volume fraction for stiff beams, then optimize one"*

Workflow:
1. **TavilySearch** - Research best practices
2. **create_beam_problem** - Set up with recommended parameters
3. **optimize_beam_design** - Generate optimal design
4. **render_beam_design** - Visualize results

### Example 3: Quick STL Conversion
User: *"convert beam_design.npy to STL"*

Workflow:
1. **convert_design_to_stl** - Direct conversion

## Usage

### Start the Supervisor Agent
```bash
python -m src.main supervisor
# or shorthand:
python -m src.main team
python -m src.main multi
```

### Start Individual Agents
```bash
python -m src.main engineering  # Engineering only
python -m src.main search       # Search only
python -m src.main general      # Math + Search (default)
```

## Key Features

### 1. Automatic File Management
- `render_beam_design` saves **both** PNG and .npy files
- .npy files are automatically used for STL conversion
- All file paths are tracked and reported

### 2. Complete Pipelines
The supervisor can handle multi-step workflows without manual intervention:
- Design → Visualization → 3D Printing
- Research → Design → Analysis
- Optimization → Parameter Studies → Comparison

### 3. Tool Coordination
The agent intelligently chains tools:
- Uses render output (.npy path) as convert input
- Applies search results to design parameters
- Iterates on designs based on simulation results

## File Outputs

### From render_beam_design:
- `beam_design.png` - Visualization (heatmap)
- `beam_design.npy` - Raw design array (NumPy format)

### From convert_design_to_stl:
- `beam_design.stl` - 3D mesh (ready for 3D printing/CAD)

## Technical Details

### render_beam_design
```python
{
    "success": True,
    "save_path": "beam_design.png",
    "npy_path": "beam_design.npy",     # ← Use this for STL conversion
    "design_shape": (50, 100),
    "seed": 42,
    "volume_fraction": 0.35,
    "compliance": 123.45
}
```

### convert_design_to_stl
```python
{
    "success": True,
    "stl_path": "beam_design.stl",
    "input_shape": (50, 100),
    "num_triangles": 19602,
    "message": "Successfully converted..."
}
```

## Benefits Over Individual Agents

### Before (Individual Agents)
1. User starts Engineering Agent
2. Optimize design, get .npy file
3. Exit Engineering Agent
4. Start CAD Agent
5. Convert .npy to STL
6. Exit CAD Agent

### Now (Supervisor Agent)
1. User starts Supervisor
2. Request: "optimize beam and make it 3D printable"
3. Agent handles complete workflow automatically
4. All files ready

## System Prompt

The supervisor has a comprehensive system prompt that:
- Lists all available tools by category
- Explains complete workflow patterns
- Guides tool chaining and coordination
- Provides engineering concept definitions
- Suggests proactive next steps

## Dependencies

All tools work out-of-the-box with existing dependencies:
- `engibench[beams2d]` - Engineering optimization
- `numpy-stl` - STL file generation
- `langchain` - Agent framework
- `tavily-python` - Web search

## Future Enhancements

Potential additions:
- **Analysis Agent**: Data analysis, comparison studies, parameter sweeps
- **Visualization Agent**: Advanced plotting, animations, mesh inspection
- **Documentation Agent**: Auto-generate design reports, technical documentation
- More EngiBench problems (airfoils, heat conduction, thermoelastic)

## Troubleshooting

### Issue: Agent not responding
**Solution**: Check that all tools are properly imported and initialized

### Issue: STL conversion fails
**Solution**: Ensure .npy file exists and contains 2D array

### Issue: "numpy-stl not found"
**Solution**: `pip install numpy-stl` or reinstall environment

## Examples

### Simple Task
```
User: show me a random beam design
Agent: [calls render_beam_design with random seed]
Agent: "Design saved to beam_design.png and beam_design.npy"
```

### Complex Task
```
User: optimize a beam with 40% material and convert to STL
Agent: [calls create_beam_problem]
Agent: [calls optimize_beam_design with volume_fraction=0.4]
Agent: [calls render_beam_design] → saves PNG + .npy
Agent: [calls convert_design_to_stl with the .npy path] → saves STL
Agent: "Complete! Files: beam_design.png, beam_design.npy, beam_design.stl
       Optimized design achieves compliance of 145.2 with 40% material usage.
       STL file has 19,602 triangles ready for 3D printing."
```

## Conclusion

The multi-agent system provides a seamless experience for complex engineering workflows. The supervisor agent acts as an expert assistant that automatically coordinates tools to solve complete problems from concept to manufacturing-ready files.
