"""Tests for the checkpoint system with PostgreSQL fallback."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.checkpoint import POSTGRES_AVAILABLE, get_checkpointer


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
