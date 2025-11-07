Utils API Reference
===================

This page documents utility modules in Engineer Assistant.

.. note::
   Auto-generated API documentation will be added in a future release.
   For now, see the usage guides and code examples.

Utility Modules
---------------

Prompts
~~~~~~~

**Module**: ``src.utils.prompts``

System prompts and templates for agents.

Contains pre-defined prompts for:

* Engineering optimization tasks
* Paper analysis and summarization
* HPC job management
* General assistant behavior

State Management
~~~~~~~~~~~~~~~~

**Module**: ``src.models.state``

State management for multi-agent workflows.

Key Classes:

* ``AgentState`` - Shared state between agents
* ``MessageState`` - Message history tracking

Defines the data structures used by LangGraph for agent coordination.

Checkpoint
~~~~~~~~~~

**Module**: ``src.checkpoint``

Checkpoint management for long-running processes.

Key Functions:

* ``save_checkpoint(state, path)`` - Save current state
* ``load_checkpoint(path)`` - Restore saved state
* ``list_checkpoints()`` - List available checkpoints

Useful for resuming interrupted optimization runs or conversations.

See Also
--------

* :doc:`../usage/agents` - User guide with examples
* :doc:`agents` - Available agents documentation
* :doc:`tools` - Available tools documentation
