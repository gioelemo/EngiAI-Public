# Using Agents

Engineer Assistant provides several specialized agents for different tasks. All agents are **LangGraph state machines** that are invoked through the Supervisor Agent, which automatically routes requests to the appropriate agent.

## How Agents Work

Each agent is a LangGraph graph with an LLM and a set of tools. You interact with agents through the **Streamlit UI** or by invoking the **Supervisor Agent** programmatically:

```python
from langchain_core.messages import HumanMessage
from src.agents.supervisor_agent import SupervisorAgent

supervisor = SupervisorAgent()
state = {
    "messages": [HumanMessage(content="Optimize a beam with 30% material")],
    "next": "",
}
result = supervisor.invoke(state, config={"configurable": {"thread_id": "demo"}})

# The response is in result["messages"]
for msg in result["messages"]:
    print(msg.content)
```

The supervisor analyzes the user's message and routes it to the appropriate specialized agent.

## Available Agents

### Engineering Agent

Handles structural design and topology optimization using EngiBench problems (beams2d, photonics2d).

**Tools available:**
- `create_problem` — Initialize a problem instance
- `optimize_design` — Run topology optimization
- `simulate_design` — Evaluate a design
- `render_design` — Visualize the design
- `convert_design_to_stl` — Export to STL for 3D printing
- `get_problem_details` / `get_dataset_info` — Problem metadata
- `download_wandb_model` / `load_wandb_model` / `sample_designs_from_model` — Generative model tools

**Example queries:**
```
"Create a 2D beam design with 35% volume fraction"
"Optimize and export to STL with 10mm extrusion height"
```

### ArXiv Agent

Searches, downloads, and analyzes research papers from arXiv.

**Tools available:**
- `search_arxiv` — Search for papers by query
- `get_arxiv_paper` — Get paper details by arXiv ID
- `download_and_analyze_paper` — Download PDF and index in MMORE
- `ask_about_papers` — Query indexed papers via RAG
- `list_analyzed_papers` — List all indexed papers

**Example queries:**
```
"Find recent papers on generative design"
"Download and analyze paper 2301.12345"
```

### RAG Agent

Queries the knowledge base using Retrieval-Augmented Generation via the MMORE service.

**Tools available:**
- `search_documents` — Query MMORE for relevant documents
- `add_document` — Upload a PDF to the knowledge base
- `add_url_to_knowledge_base` — Crawl and index a web page
- `list_documents` / `delete_document` — Manage indexed documents

**Example queries:**
```
"What papers discuss topology optimization?"
"What does the EngiBench paper say about volume fraction?"
```

### HPC Agent

Manages high-performance computing workflows on SLURM clusters.

**Tools available:**
- `test_hpc_connection` — Verify SSH connectivity
- `submit_slurm_job` — Submit a SLURM job script
- `get_slurm_job_status` — Check job status
- `monitor_job_until_complete` — Wait for job completion
- `cancel_slurm_job` — Cancel a running job
- `download_job_outputs` — Retrieve results
- `generate_training_command` / `evaluate_model` — ML training pipeline tools

**Example queries:**
```
"Submit training job to Euler cluster"
"Check status of my running jobs"
```

See the [HPC Integration Guide](hpc.md) for setup details.

### Search Agent

Performs web searches for technical information using Tavily API.

**Example queries:**
```
"Find the latest research on additive manufacturing"
"What are the best practices for topology optimization?"
```

### Prusa Agent

Controls 3D printers via Prusa Connect through an MCP server.

**Tools:** Dynamically loaded from the Prusa MCP server (printer listing, file upload, print job management, status monitoring).

See the [Prusa Integration Guide](prusa.md) for setup and usage.

### CLI Agent

Executes command-line tools on the local system. Supports human-in-the-loop confirmation for safety.

**Tools available:**
- `execute_cli_command` — Run shell commands
- `open_gui_application` — Launch GUI apps (PrusaSlicer, VS Code, etc.)

**Example queries:**
```
"List files in the outputs directory"
"Open PrusaSlicer with the latest design"
```

**Note:** In Docker, GUI apps are opened on the host machine via the host service (`services/host_service.py`).

## Supervisor Agent

The Supervisor Agent is the entry point that routes requests to specialized agents. It:

1. Analyzes the user's message intent
2. Routes to the appropriate agent with scoped instructions
3. Collects the agent's response
4. Decides whether to route to another agent or return to the user

The supervisor also handles:
- **Human-in-the-loop**: Interrupts before CLI agent execution for user confirmation
- **Loop detection**: Prevents the same agent from being called repeatedly without progress

## Next Steps

- [Learn about the available tools](tools.md)
- [Configure your agents](../configuration.md)
- [API Reference](../api/agents.rst)
