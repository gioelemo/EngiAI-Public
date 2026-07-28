EngiAI Documentation
================================

A Jarvis-style multimodal AI assistant for closed-loop design-for-manufacturing correction.

EngiAI is a sophisticated multi-agent system that combines:

* **🐳 Docker Deployment**: Production-ready containerized deployment
* **🤖 Multi-Agent System**: Supervisor coordinates specialized agents
* **🔧 Engineering Tools**: EngiBench integration for structural optimization
* **📚 RAG System**: Document Q&A with MMORE multimodal RAG service
* **🖨️ 3D Printer Integration**: Prusa Connect integration via MCP
* **🖥️ HPC Integration**: SLURM job management for remote compute clusters
* **🔍 Research Tools**: ArXiv search and web research capabilities
* **💬 Interactive UI**: Streamlit web interface with chat and visualization

Quick Links
-----------

* **New to EngiAI?** Start with the :doc:`quickstart`
* **Want to deploy?** See :doc:`installation` (Docker recommended)
* **Need help?** Check the :doc:`troubleshooting` guide
* **Understand the system?** Read the :doc:`architecture` overview

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   configuration
   architecture

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Deployment

   docker_deployment
   deployment
   database_setup
   troubleshooting

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   usage/agents
   usage/tools
   usage/prusa
   usage/hpc
   usage/ui

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Advanced Topics

   paper_import_guide
   weave_integration

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   contributing

Features at a Glance
====================

Multi-Agent Architecture
------------------------

The system uses a supervisor-based architecture where specialized agents handle specific tasks:

* **Supervisor Agent**: Routes requests to appropriate specialized agents
* **Engineering Agent**: Structural optimization and topology design
* **RAG Agent**: Knowledge base queries with retrieval-augmented generation
* **ArXiv Agent**: Research paper search and analysis
* **Search Agent**: Web research and information gathering
* **HPC Agent**: High-performance computing job management
* **Prusa Agent**: 3D printer control and monitoring
* **CLI Agent**: Command-line tool execution

See :doc:`architecture` for detailed system design.

Key Capabilities
----------------

**Engineering & Optimization**
   * Topology optimization with EngiBench (beams2d, photonics2d, thermoelastic2d)
   * STL export for 3D printing
   * Design simulation and constraints checking
   * Multi-physics optimization

**Research & Knowledge**
   * ArXiv paper search and download
   * RAG-based document Q&A with MMORE
   * Multimodal document processing
   * PDF processing and chunking

**3D Printing**
   * Prusa Connect integration
   * Print job submission and monitoring
   * File management
   * Printer status tracking

**HPC & Remote Computing**
   * SSH-based cluster access
   * SLURM job submission
   * Job monitoring and output retrieval
   * File transfer (SFTP)

**Web Interface**
   * Interactive chat with agents
   * Multi-chat session management
   * File uploads (PDFs, designs, data)
   * Real-time visualization
   * W&B report embedding

Technology Stack
----------------

* **Frameworks**: LangGraph, LangChain, Streamlit, Fabric
* **LLMs**: OpenAI GPT-4o/GPT-4.1, Google Gemini-3-Flash, Ollama (Qwen3, Qwen3.5)
* **Storage**: MMORE (RAG), PostgreSQL/SQLite (relational)
* **Integrations**: EngiBench/EngiOpt, Prusa Connect, Weights & Biases, Tavily

Deployment Options
------------------

**🐳 Docker** (Recommended for production)
   * Isolated environment with all dependencies
   * Easy deployment with docker-compose
   * Optional Prusa MCP server integration
   * PostgreSQL database included

**💻 Local Development** (For active development)
   * Conda environment with Python 3.13+
   * Hot reload for code changes
   * SQLite database
   * Full IDE integration

See :doc:`installation` for setup instructions.

Getting Help
============

* **Troubleshooting**: See :doc:`troubleshooting` for common issues
* **GitHub Issues**: `Report bugs or request features <https://github.com/gioelemo/EngiAI-Public/issues>`_
* **Documentation**: Browse the guides in the sidebar
* **Examples**: Check the ``scripts/`` directory in the repository

Project Links
=============

* **GitHub**: https://github.com/gioelemo/EngiAI-Public
* **License**: MIT

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
