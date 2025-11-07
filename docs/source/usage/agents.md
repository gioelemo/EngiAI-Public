# Using Agents

Engineer Assistant provides several specialized agents for different tasks.

## Available Agents

### Engineering Agent

The Engineering Agent handles optimization tasks using state-of-the-art algorithms.

```python
from src.agents import EngineeringAgent

agent = EngineeringAgent()
result = agent.optimize(problem="beams2d", config={"volfrac": 0.3})
```

### ArXiv Agent

Search, download, and analyze research papers from arXiv.

```python
from src.agents import ArXivAgent

agent = ArXivAgent()
papers = agent.search("topology optimization", max_results=5)
```

### RAG Agent

Query your knowledge base using Retrieval-Augmented Generation.

```python
from src.agents import RAGAgent

agent = RAGAgent()
answer = agent.ask("How does topology optimization work?")
```

### HPC Agent

Submit and monitor jobs on high-performance computing clusters.

```python
from src.agents import HPCAgent

agent = HPCAgent()
job_id = agent.submit_job(script="simulation.sh")
status = agent.check_status(job_id)
```

### Prusa Agent

Control 3D printers for rapid prototyping.

```python
from src.agents import PrusaAgent

agent = PrusaAgent()
agent.print_design(stl_file="design.stl")
```

## Supervisor Agent

The Supervisor Agent orchestrates multiple agents to handle complex workflows.

```python
from src.agents import SupervisorAgent

supervisor = SupervisorAgent()
result = supervisor.process("Optimize a beam design and export to STL")
```

## Next Steps

- [Learn about the available tools](tools.md)
- [Configure your agents](../configuration.md)
- [API Reference](../api/agents.rst)
