Agents API Reference
====================

This page documents all available agents in Engineer Assistant.

.. note::
   Auto-generated API documentation will be added in a future release.
   For now, see the usage guides and code examples.

Agent Classes
-------------

All agents are located in the ``src.agents`` module.

Base Agent
~~~~~~~~~~

The base class for all agents.

**Module**: ``src.agents.base_agent``

**Class**: ``BaseAgent``

Provides common functionality for agent initialization, tool binding, and invocation.

Supervisor Agent
~~~~~~~~~~~~~~~~

Orchestrates multiple agents to handle complex workflows.

**Module**: ``src.agents.supervisor_agent``

**Class**: ``SupervisorAgent``

The supervisor routes requests to appropriate specialized agents.

Engineering Agent
~~~~~~~~~~~~~~~~~

Handles optimization and engineering design tasks.

**Module**: ``src.agents.engineering_agent``

**Class**: ``EngineeringAgent``

Integrates with EngiBench and EngiOpt for optimization problems.

ArXiv Agent
~~~~~~~~~~~

Search, download, and analyze research papers from arXiv.

**Module**: ``src.agents.arxiv_agent``

**Class**: ``ArXivAgent``

Provides tools for paper search, retrieval, and RAG-based analysis.

RAG Agent
~~~~~~~~~

Query knowledge base using Retrieval-Augmented Generation.

**Module**: ``src.agents.rag_agent``

**Class**: ``RAGAgent``

Retrieves relevant documents and generates answers from the vector store.

HPC Agent
~~~~~~~~~

Submit and monitor jobs on HPC clusters.

**Module**: ``src.agents.hpc_agent``

**Class**: ``HPCAgent``

Supports SLURM job submission and monitoring.

Prusa Agent
~~~~~~~~~~~

Control 3D printers for rapid prototyping.

**Module**: ``src.agents.prusa_agent``

**Class**: ``PrusaAgent``

Interfaces with Prusa printers for design fabrication.

Search Agent
~~~~~~~~~~~~

Perform web searches for information retrieval.

**Module**: ``src.agents.search_agent``

**Class**: ``SearchAgent``

Uses various search APIs to find relevant information.

CLI Agent
~~~~~~~~~

Command-line interface for agent interaction.

**Module**: ``src.agents.cli_agent``

**Class**: ``CLIAgent``

Provides terminal-based interaction with the assistant.

See Also
--------

* :doc:`../usage/agents` - User guide with examples
* :doc:`tools` - Available tools documentation
