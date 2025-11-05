"""Tests for the checkpoint system with PostgreSQL fallback."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.checkpoint import POSTGRES_AVAILABLE, close_checkpointer, get_checkpointer


class TestCheckpointerInitialization:
    """Test checkpointer initialization with different configurations."""

    def test_memorysaver_when_postgres_not_available(self, monkeypatch):
        """Test that MemorySaver is used when PostgreSQL is not available."""
        # Mock POSTGRES_AVAILABLE to False
        monkeypatch.setattr("src.checkpoint.POSTGRES_AVAILABLE", False)
        monkeypatch.setattr("src.checkpoint.PostgresSaver", None)

        # Reset global state
        import src.checkpoint as cp

        cp._checkpointer = None
        cp._initialized = False

        checkpointer = get_checkpointer()

        assert checkpointer is not None
        from langgraph.checkpoint.memory import MemorySaver

        assert isinstance(checkpointer, MemorySaver)

    def test_memorysaver_with_sqlite_url(self, monkeypatch):
        """Test that MemorySaver is used with SQLite database URL."""
        from config import config

        original_url = config.database_url
        config.database_url = "sqlite:///test.db"

        # Reset global state
        import src.checkpoint as cp

        cp._checkpointer = None
        cp._initialized = False

        try:
            checkpointer = get_checkpointer()

            assert checkpointer is not None
            from langgraph.checkpoint.memory import MemorySaver

            assert isinstance(checkpointer, MemorySaver)
        finally:
            config.database_url = original_url
            # Reset state
            cp._checkpointer = None
            cp._initialized = False

    @pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
    def test_postgres_saver_with_postgres_url(self, monkeypatch):
        """Test that PostgresSaver is attempted with PostgreSQL URL."""
        from config import config

        original_url = config.database_url
        config.database_url = "postgresql://user:pass@localhost/test"

        # Reset global state
        import src.checkpoint as cp

        cp._checkpointer = None
        cp._initialized = False

        # Mock PostgresSaver to avoid actual connection
        mock_saver = Mock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_saver)

        with patch("src.checkpoint.PostgresSaver") as mock_postgres:
            mock_postgres.from_conn_string.return_value = mock_context

            try:
                checkpointer = get_checkpointer()

                assert checkpointer is not None
                mock_postgres.from_conn_string.assert_called_once_with(
                    "postgresql://user:pass@localhost/test"
                )
                mock_saver.setup.assert_called_once()
            finally:
                config.database_url = original_url
                cp._checkpointer = None
                cp._initialized = False

    def test_singleton_behavior(self):
        """Test that get_checkpointer returns the same instance."""
        # Reset global state
        import src.checkpoint as cp

        cp._checkpointer = None
        cp._initialized = False

        checkpointer1 = get_checkpointer()
        checkpointer2 = get_checkpointer()

        assert checkpointer1 is checkpointer2

    def test_fallback_on_postgres_connection_error(self, monkeypatch):
        """Test fallback to MemorySaver when PostgreSQL connection fails."""
        from config import config

        if not POSTGRES_AVAILABLE:
            pytest.skip("PostgreSQL not available")

        original_url = config.database_url
        config.database_url = "postgresql://invalid:connection@localhost/test"

        # Reset global state
        import src.checkpoint as cp

        cp._checkpointer = None
        cp._initialized = False

        # Mock PostgresSaver to raise an exception
        with patch("src.checkpoint.PostgresSaver") as mock_postgres:
            mock_postgres.from_conn_string.side_effect = Exception("Connection failed")

            try:
                checkpointer = get_checkpointer()

                # Should fall back to MemorySaver
                from langgraph.checkpoint.memory import MemorySaver

                assert isinstance(checkpointer, MemorySaver)
            finally:
                config.database_url = original_url
                cp._checkpointer = None
                cp._initialized = False


class TestCheckpointerClose:
    """Test checkpointer cleanup."""

    def test_close_checkpointer_with_memorysaver(self):
        """Test closing MemorySaver checkpointer."""
        import src.checkpoint as cp

        # Initialize with MemorySaver
        cp._checkpointer = None
        cp._initialized = False
        get_checkpointer()

        # Close it
        close_checkpointer()

        assert cp._checkpointer is None
        assert not cp._initialized

    @pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
    def test_close_checkpointer_with_postgres(self, monkeypatch):
        """Test closing PostgreSQL checkpointer."""
        import src.checkpoint as cp
        from config import config

        if not POSTGRES_AVAILABLE:
            pytest.skip("PostgreSQL not available")

        original_url = config.database_url
        config.database_url = "postgresql://user:pass@localhost/test"

        # Reset global state
        cp._checkpointer = None
        cp._initialized = False

        # Mock PostgresSaver - need to actually create a mock class
        from langgraph.checkpoint.postgres import (
            PostgresSaver as RealPostgresSaver,
        )

        mock_saver = Mock(spec=RealPostgresSaver)
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_saver)
        mock_context.__exit__ = Mock()

        with (
            patch("src.checkpoint.PostgresSaver", RealPostgresSaver),
            patch.object(
                RealPostgresSaver, "from_conn_string", return_value=mock_context
            ),
        ):
            try:
                get_checkpointer()
                cp._context_manager = mock_context

                close_checkpointer()

                mock_context.__exit__.assert_called_once()
                assert cp._checkpointer is None
                assert not cp._initialized
            finally:
                config.database_url = original_url
                cp._checkpointer = None
                cp._initialized = False


class TestCheckpointerEdgeCases:
    """Test edge cases and error scenarios."""

    def test_multiple_initializations(self):
        """Test that multiple calls don't reinitialize."""
        import src.checkpoint as cp

        cp._checkpointer = None
        cp._initialized = False

        checkpointer1 = get_checkpointer()
        checkpointer2 = get_checkpointer()
        checkpointer3 = get_checkpointer()

        assert checkpointer1 is checkpointer2 is checkpointer3
        assert cp._initialized

    def test_close_without_initialization(self):
        """Test that closing without initialization doesn't error."""
        import src.checkpoint as cp

        cp._checkpointer = None
        cp._initialized = False

        # Should not raise error
        close_checkpointer()

        assert cp._checkpointer is None
        assert not cp._initialized
