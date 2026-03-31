"""
Tests for prusa_mcp_server/server.py module.

These tests cover the Prusa MCP server startup and configuration.
"""

from unittest.mock import MagicMock, patch

import pytest

from prusa_mcp_server.server import main

# ============================================================================
# MAIN FUNCTION TESTS
# ============================================================================


@pytest.mark.unit
def test_main_starts_server(monkeypatch):
    """Test that main starts the SSE server with uvicorn."""
    monkeypatch.setenv("PRUSA_MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("PRUSA_MCP_PORT", "8765")

    fake_mcp = MagicMock()
    fake_mcp.sse_app = MagicMock()  # non-callable → used directly

    with (
        patch("prusa_mcp_server.server.uvicorn.run") as mock_run,
        patch.dict(
            "sys.modules",
            {"prusa_mcp": MagicMock(), "prusa_mcp.server": MagicMock(mcp=fake_mcp)},
        ),
    ):
        main()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["host"] == "127.0.0.1"
        assert call_kwargs[1]["port"] == 8765


@pytest.mark.unit
def test_main_calls_app_factory(monkeypatch):
    """Test that main calls app factory if sse_app is callable."""
    monkeypatch.setenv("PRUSA_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("PRUSA_MCP_PORT", "8765")

    fake_app = MagicMock()
    fake_mcp = MagicMock()
    fake_mcp.sse_app = MagicMock(return_value=fake_app)  # callable → factory

    with (
        patch("prusa_mcp_server.server.uvicorn.run") as mock_run,
        patch.dict(
            "sys.modules",
            {"prusa_mcp": MagicMock(), "prusa_mcp.server": MagicMock(mcp=fake_mcp)},
        ),
    ):
        main()

        mock_run.assert_called_once()


@pytest.mark.unit
def test_main_default_host_and_port(monkeypatch):
    """Test main uses default host and port when not set."""
    monkeypatch.delenv("PRUSA_MCP_HOST", raising=False)
    monkeypatch.delenv("PRUSA_MCP_PORT", raising=False)

    fake_mcp = MagicMock()
    fake_mcp.sse_app = object()  # non-callable

    with (
        patch("prusa_mcp_server.server.uvicorn.run") as mock_run,
        patch.dict(
            "sys.modules",
            {"prusa_mcp": MagicMock(), "prusa_mcp.server": MagicMock(mcp=fake_mcp)},
        ),
    ):
        main()

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 8000


@pytest.mark.unit
def test_main_logs_info(monkeypatch, caplog):
    """Test that main logs information messages."""
    import logging

    monkeypatch.setenv("PRUSA_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("PRUSA_MCP_PORT", "8765")

    fake_mcp = MagicMock()
    fake_mcp.sse_app = object()

    with (
        patch("prusa_mcp_server.server.uvicorn.run"),
        patch.dict(
            "sys.modules",
            {"prusa_mcp": MagicMock(), "prusa_mcp.server": MagicMock(mcp=fake_mcp)},
        ),
        caplog.at_level(logging.INFO),
    ):
        main()

    assert any(
        "Starting" in record.message or "Running" in record.message
        for record in caplog.records
    )
