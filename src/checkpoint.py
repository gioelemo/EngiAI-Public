"""
Global checkpointer instance for persistent conversation memory.

This module provides a singleton checkpointer instance that's shared across all agents.
- Uses PostgresSaver for PostgreSQL connections
- Uses MemorySaver for SQLite (persistent within single session)
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from config import config

# Global checkpointer instance (initialized lazily)
_checkpointer: PostgresSaver | MemorySaver | None = None
_context_manager = None  # Keep context manager alive
_initialized: bool = False


def get_checkpointer() -> PostgresSaver | MemorySaver:
    """
    Get or create the global checkpointer instance.

    The checkpointer is created once and reused across all agents.
    - PostgreSQL URLs: Uses PostgresSaver with persistent database storage
    - SQLite URLs: Uses MemorySaver (persistent within session only)

    Note: langgraph 1.0.1 doesn't include SqliteSaver, so SQLite configurations
    will use MemorySaver which persists for the application lifetime but not
    across restarts.

    Returns:
        Checkpointer instance (PostgresSaver or MemorySaver)
    """
    global _checkpointer, _context_manager, _initialized  # noqa: PLW0603

    if _checkpointer is None:
        # Check if using PostgreSQL or SQLite
        if config.database_url.startswith(
            "postgresql://"
        ) or config.database_url.startswith("postgres://"):
            # Use PostgreSQL persistent checkpointer
            _context_manager = PostgresSaver.from_conn_string(config.database_url)
            _checkpointer = (
                _context_manager.__enter__()
            )  # Get the actual saver from context manager

            # Initialize database tables on first use
            if not _initialized:
                _checkpointer.setup()
                _initialized = True
                print(
                    "✅ PostgreSQL checkpointer initialized (persistent across restarts)"
                )
        else:
            # SQLite or other: fall back to MemorySaver
            _checkpointer = MemorySaver()
            _initialized = True
            print("⚠️  Using MemorySaver (SQLite not supported in langgraph 1.0.1)")
            print("   Conversations persist within session but not across restarts")
            print("   For persistent storage, configure PostgreSQL in DATABASE_URL")

    return _checkpointer


def close_checkpointer() -> None:
    """
    Close the global checkpointer connection.

    This should be called when the application is shutting down.
    """
    global _checkpointer, _context_manager, _initialized  # noqa: PLW0603

    if _checkpointer is not None:
        if isinstance(_checkpointer, PostgresSaver) and _context_manager is not None:
            _context_manager.__exit__(None, None, None)  # Exit the context manager
        _checkpointer = None
        _context_manager = None
        _initialized = False
