# Tools

Engineer Assistant provides a comprehensive set of tools for various tasks.

## ArXiv Tools

Search and analyze academic papers.

### `search_arxiv`

Search for papers on arXiv.

```python
from src.tools import arxiv_tools

results = arxiv_tools.search_papers("machine learning", max_results=10)
```

### `get_arxiv_paper`

Download and retrieve paper details.

```python
paper = arxiv_tools.get_paper("2301.12345")
```

## RAG Tools

Retrieve information from your knowledge base using MMORE.

### `search_documents`

Query MMORE for relevant documents.

```python
from src.tools import MMOREClient

mmore = MMOREClient()
results = mmore.retrieve(
    query="What is topology optimization?",
    max_matches=5
)
```

### `upload_file`

Add new documents to MMORE.

```python
from src.tools import MMOREClient

mmore = MMOREClient()
response = mmore.upload_file(
    file_path="document.pdf",
    file_id="my_document"
)
```

## HPC Tools

Interact with high-performance computing clusters.

### `submit_slurm_job`

Submit a job to a SLURM cluster.

```python
from src.tools import hpc

job_id = hpc.submit_job(
    script="simulation.sh",
    nodes=4,
    time="24:00:00"
)
```

### `monitor_job`

Check job status and retrieve results.

```python
status = hpc.check_job_status(job_id)
output = hpc.get_job_output(job_id)
```

## Engineering Tools

Optimization and design tools.

### `engibench_optimize`

Run optimization using EngiBench problems.

```python
from src.tools import engibench

result = engibench.optimize(
    problem="beams2d",
    config={"volfrac": 0.3, "forcedist": 0.5}
)
```

### `engiopt_optimize`

Use EngiOpt algorithms for optimization.

```python
from src.tools import engiopt

result = engiopt.optimize(
    design=initial_design,
    objective=objective_fn
)
```

## Export Tools

Export designs to various formats.

### `export_to_stl`

Convert designs to STL format for 3D printing.

```python
from src.tools import stl_export

stl_export.design_to_stl(
    design=optimized_design,
    output="design.stl",
    extrude_height=10.0
)
```

## Next Steps

- [Learn about agents](agents.md)
- [HPC Setup Guide](hpc.md)
- [API Reference](../api/tools.rst)
