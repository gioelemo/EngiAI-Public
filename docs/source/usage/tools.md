# Tools

EngiAI provides a comprehensive set of tools that agents use to complete tasks. All tools are LangChain `@tool` decorated functions that agents call automatically based on user requests.

## ArXiv Tools

Search and analyze academic papers.

### `search_arxiv`

Search for papers on arXiv.

```python
from src.tools import search_arxiv

results = search_arxiv.invoke({"query": "topology optimization", "max_results": 5})
```

### `get_arxiv_paper`

Retrieve paper details by arXiv ID.

```python
from src.tools import get_arxiv_paper

paper = get_arxiv_paper.invoke({"arxiv_id": "2301.12345"})
```

Additional tools (created via factory with MMORE client):
- `download_and_analyze_paper` — Download PDF and index in MMORE
- `ask_about_papers` — Query indexed papers via RAG
- `list_analyzed_papers` — List all indexed papers

## RAG Tools

Retrieve information from your knowledge base using MMORE.

### `MMOREClient`

The MMORE client provides direct access to the document retrieval service:

```python
from src.tools import MMOREClient

mmore = MMOREClient()
results = mmore.retrieve(
    query="What is topology optimization?",
    max_matches=5
)
```

### RAG tool functions (created via factory)

```python
from src.tools import create_rag_tools, MMOREClient

mmore = MMOREClient()
tools = create_rag_tools(mmore)
# Returns: [search_documents, add_document, add_url_to_knowledge_base,
#           list_documents, delete_document]
```

## HPC Tools

Interact with high-performance computing clusters via SSH/SLURM.

### `submit_slurm_job`

Submit a SLURM job script to the cluster.

```python
from src.tools.hpc import submit_slurm_job

result = submit_slurm_job.invoke({
    "slurm_file": "#!/bin/bash\n#SBATCH --job-name=test\npython train.py",
    "host_alias": "euler",
})
```

### `get_slurm_job_status`

Check the status of a submitted job.

```python
from src.tools.hpc import get_slurm_job_status

status = get_slurm_job_status.invoke({"job_id": "12345"})
```

### Other HPC tools

- `test_hpc_connection` — Verify SSH connectivity
- `cancel_slurm_job` — Cancel a running job
- `download_job_outputs` — Retrieve job output files

### Job monitoring tools (`src/tools/job_monitor.py`)

Polling-based monitors that sit on top of `get_slurm_job_status`:

- `monitor_job_until_complete` — Poll a job until it finishes (or `max_checks` is reached) and optionally auto-download outputs
- `check_job_status_change` — Compare a job's current state against a cached status and return whether it changed
- `get_active_jobs_summary` — Summarize all jobs currently tracked in the in-memory status cache

## Engineering Tools

Optimization and design tools using EngiBench.

Supported problem types:
- **beams2d**: 2D structural topology optimization
- **photonics2d**: 2D photonic device topology optimization
- **thermoelastic2d**: 2D thermoelastic (coupled thermal/structural) topology optimization

### `optimize_design`

Run topology optimization on an EngiBench problem.

```python
from src.tools import optimize_design

result = optimize_design.invoke({
    "problem_type": "beams2d",
    "volfrac": 0.3,
    "forcedist": 0.5,
})
```

### All EngiBench tools

- `create_problem` — Initialize a problem instance
- `simulate_design` — Evaluate a design's objectives
- `optimize_design` — Run topology optimization
- `render_design` — Visualize the design as an image
- `get_problem_details` — Get detailed problem description
- `get_dataset_info` — Get dataset information

### EngiOpt tools

Tools for working with trained generative models:

- `download_wandb_model` — Download a model from W&B
- `load_wandb_model` — Load a downloaded model
- `sample_designs_from_model` — Generate designs from a trained model
- `generate_training_command` — Generate SLURM training scripts
- `evaluate_model` — Evaluate a trained model against baselines

#### Algorithm registry (`src/tools/algorithms.py`)

Single source of truth for supported generative algorithm IDs. Add an entry to `SUPPORTED_ALGORITHMS` (and to the `AlgorithmId` Literal) to expose a new algorithm to every EngiOpt tool automatically.

- `AlgorithmId` — Literal type of valid algorithm IDs (currently `cgan_cnn_2d`, `diffusion_2d_cond`)
- `SUPPORTED_ALGORITHMS` — Dict of algorithm metadata (class, dimensions, conditional, model type, model file names)
- `ALGORITHM_IDS` — List derived from `AlgorithmId`, used by the EngiOpt tools for validation

## Export Tools

### `convert_design_to_stl`

Convert a 2D design to STL format for 3D printing.

```python
from src.tools import convert_design_to_stl

result = convert_design_to_stl.invoke({
    "npy_file_path": "outputs/design.npy",
    "scale_xy": 1.0,
    "scale_z": 10.0,
    "mirror_y": False,
    "threshold": 0.5,
})
```

## CLI Tools

- `execute_cli_command` — Run shell commands locally
- `open_gui_application` — Launch GUI applications (PrusaSlicer, VS Code, etc.)

## Search Tools

- `create_search_tool()` — Factory that returns a Tavily web search tool

## Human Input Tools

Tools that let an agent pause and request clarification from the user.

### `ask_human_for_clarification`

Used by the engineering agent when a user's request does not specify required numerical design parameters (e.g., volume fraction, force distance, filter radius). The agent calls this tool instead of guessing defaults; the UI surfaces the question and waits for a reply before any design tool runs.

```python
from src.tools.human_input import ask_human_for_clarification

ask_human_for_clarification.invoke({
    "clarification_request": (
        "What volume fraction should I use for this beam design? "
        "Please specify a value between 0.1 and 0.9."
    ),
})
```

## Next Steps

- [Learn about agents](agents.md)
- [HPC Setup Guide](hpc.md)
- [API Reference](../api/tools.rst)
