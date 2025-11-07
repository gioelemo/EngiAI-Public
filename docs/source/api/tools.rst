Tools API Reference
===================

This page documents all available tools in Engineer Assistant.

.. note::
   Auto-generated API documentation will be added in a future release.
   For now, see the usage guides and code examples.

Tool Modules
------------

All tools are located in the ``src.tools`` module.

ArXiv Tools
~~~~~~~~~~~

**Module**: ``src.tools.arxiv_tools``

Functions for searching and retrieving papers from arXiv.

Key Functions:

* ``search_papers(query, max_results)`` - Search arXiv
* ``get_paper(arxiv_id)`` - Get paper details
* ``download_paper(arxiv_id, path)`` - Download PDF

RAG Chain
~~~~~~~~~

**Module**: ``src.tools.rag_chain``

Retrieval-Augmented Generation chain for querying documents.

Key Functions:

* ``query(question)`` - Query the knowledge base
* ``get_relevant_docs(query, k)`` - Retrieve similar documents

Vector Store
~~~~~~~~~~~~

**Module**: ``src.tools.vector_store``

Vector database for document embeddings and retrieval.

Key Functions:

* ``add_documents(docs)`` - Add documents to store
* ``similarity_search(query, k)`` - Search for similar documents
* ``delete_collection()`` - Clear the store

Document Processor
~~~~~~~~~~~~~~~~~~

**Module**: ``src.tools.document_processor``

Process and chunk documents for RAG.

Key Functions:

* ``process_pdf(path)`` - Extract text from PDF
* ``chunk_text(text, chunk_size)`` - Split into chunks

HPC Tools
~~~~~~~~~

**Module**: ``src.tools.hpc``

High-performance computing cluster integration.

Key Functions:

* ``submit_job(script, **kwargs)`` - Submit SLURM job
* ``check_status(job_id)`` - Check job status
* ``cancel_job(job_id)`` - Cancel running job

Job Monitor
~~~~~~~~~~~

**Module**: ``src.tools.job_monitor``

Monitor and manage HPC job execution.

Key Functions:

* ``monitor_job(job_id)`` - Track job progress
* ``get_output(job_id)`` - Retrieve job output

EngiBench Integration
~~~~~~~~~~~~~~~~~~~~~

**Module**: ``src.tools.engibench``

Interface with EngiBench optimization problems.

Key Functions:

* ``optimize(problem, config)`` - Run optimization
* ``list_problems()`` - Get available problems

EngiOpt Integration
~~~~~~~~~~~~~~~~~~~

**Module**: ``src.tools.engiopt``

Use EngiOpt algorithms for optimization.

Key Functions:

* ``optimize(design, objective)`` - Optimize design
* ``get_algorithms()`` - List available algorithms

STL Export
~~~~~~~~~~

**Module**: ``src.tools.stl_export``

Export designs to STL format for 3D printing.

Key Functions:

* ``design_to_stl(design, output)`` - Convert to STL
* ``extrude_2d_to_3d(design, height)`` - Create 3D model

Search Tools
~~~~~~~~~~~~

**Module**: ``src.tools.search``

Web search functionality.

Key Functions:

* ``search(query)`` - Perform web search
* ``get_page_content(url)`` - Fetch page content

CLI Tools
~~~~~~~~~

**Module**: ``src.tools.cli``

Command-line interface utilities.

Key Functions:

* ``run_command(cmd)`` - Execute shell command
* ``get_user_input(prompt)`` - Interactive input

Connection Tools
~~~~~~~~~~~~~~~~

**Module**: ``src.tools.connection``

Network and SSH connection utilities.

Key Functions:

* ``ssh_connect(host, user)`` - Establish SSH connection
* ``upload_file(local, remote)`` - Transfer file to remote

See Also
--------

* :doc:`../usage/tools` - User guide with examples
* :doc:`agents` - Available agents documentation
