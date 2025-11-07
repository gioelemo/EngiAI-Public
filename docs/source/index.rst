Engineer Assistant Documentation
================================

A Jarvis-style multimodal AI assistant for closed-loop design-for-manufacturing correction.

Engineer Assistant is a sophisticated multi-agent system that combines:

* **ArXiv Integration**: Search, download, and analyze research papers
* **RAG Agent**: Retrieve and query information from a knowledge base
* **HPC Agent**: Submit and monitor jobs on high-performance computing clusters
* **Engineering Agent**: Optimize designs using state-of-the-art algorithms
* **Prusa Agent**: Control 3D printers for rapid prototyping

The system uses LangGraph for agent orchestration and provides both CLI and web UI interfaces.

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   quickstart
   configuration
   database_setup
   docker_deployment
   paper_import_guide

.. toctree::
   :hidden:
   :caption: User Guide

   usage/agents
   usage/tools
   usage/hpc
   usage/ui

.. toctree::
   :hidden:
   :caption: Reference

   api/agents
   api/tools
   api/utils

.. toctree::
   :hidden:
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
