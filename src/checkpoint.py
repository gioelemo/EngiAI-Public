"""
Global checkpointer instance for persistent conversation memory.

This module provides a singleton checkpointer instance that's shared across all agents.
- Uses PostgresSaver for PostgreSQL connections (if available)
- Uses MemorySaver for SQLite or as fallback (persistent within single session)
"""

from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.postgres import PostgresSaver

    POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None  # type: ignore[assignment, misc]
    POSTGRES_AVAILABLE = False

from config import config

# Global checkpointer instance (initialized lazily)
_checkpointer: MemorySaver | None = None
_context_manager = None  # Keep context manager alive
_initialized: bool = False


def get_checkpointer() -> MemorySaver:
    """
    Get or create the global checkpointer instance.

    The checkpointer is created once and reused across all agents.
    - PostgreSQL URLs: Uses PostgresSaver with persistent database storage (if available)
    - SQLite URLs or fallback: Uses MemorySaver (persistent within session only)

    Note: PostgresSaver requires langgraph-checkpoint-postgres package.
    If not available, falls back to MemorySaver.

    Returns:
        Checkpointer instance (PostgresSaver or MemorySaver)
    """
    global _checkpointer, _context_manager, _initialized  # noqa: PLW0603

    if _checkpointer is None:
        # Check if using PostgreSQL and if PostgresSaver is available
        if (
            POSTGRES_AVAILABLE
            and PostgresSaver is not None
            and (
                config.database_url.startswith("postgresql://")
                or config.database_url.startswith("postgres://")
            )
        ):
            # Use PostgreSQL persistent checkpointer
            try:
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
            except Exception as e:
                print(f"⚠️  Failed to initialize PostgreSQL checkpointer: {e}")
                print("   Falling back to MemorySaver")
                _checkpointer = MemorySaver()
                _initialized = True
        else:
            # SQLite or other: fall back to MemorySaver
            _checkpointer = MemorySaver()
            _initialized = True
            if not POSTGRES_AVAILABLE:
                print(
                    "⚠️  PostgresSaver not available (install langgraph-checkpoint-postgres)"
                )
            print("   Using MemorySaver - conversations persist within session only")
            print(
                "   For persistent storage across restarts, install langgraph-checkpoint-postgres"
            )

    return _checkpointer  # type: ignore[return-value]


def close_checkpointer() -> None:
    """
    Close the global checkpointer connection.

    This should be called when the application is shutting down.
    """
    global _checkpointer, _context_manager, _initialized  # noqa: PLW0603

    if _checkpointer is not None:
        if (
            POSTGRES_AVAILABLE
            and PostgresSaver is not None
            and isinstance(_checkpointer, PostgresSaver)
            and _context_manager is not None
        ):
            _context_manager.__exit__(None, None, None)  # Exit the context manager
        _checkpointer = None
        _context_manager = None
        _initialized = False
