"""
Tests for prusa_mcp_server/server.py module.

These tests cover the Prusa MCP server startup and configuration.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from prusa_mcp_server.server import get_prusa_mcp_path, main

# ============================================================================
# GET PRUSA MCP PATH TESTS
# ============================================================================


@pytest.mark.unit
def test_get_prusa_mcp_path_from_env(monkeypatch):
    """Test getting Prusa MCP path from environment variable."""
    monkeypatch.setenv("PRUSA_MCP_PATH", "/custom/prusa-mcp")

    result = get_prusa_mcp_path()

    assert result == Path("/custom/prusa-mcp")


@pytest.mark.unit
def test_get_prusa_mcp_path_default(monkeypatch):
    """Test default Prusa MCP path."""
    monkeypatch.delenv("PRUSA_MCP_PATH", raising=False)

    result = get_prusa_mcp_path()

    assert "prusa-mcp" in str(result)
    assert Path.home() in result.parents or result.parent == Path.home() / "Desktop"


# ============================================================================
# MAIN FUNCTION TESTS
# ============================================================================


def _create_fake_package(tmp_path, server_content):
    """Helper to create a fake prusa_mcp package at tmp_path/src/prusa_mcp/."""
    pkg_dir = tmp_path / "src" / "prusa_mcp"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "server.py").write_text(server_content)


@pytest.mark.unit
def test_main_file_not_found(monkeypatch, tmp_path):
    """Test main raises FileNotFoundError when Prusa MCP not found."""
    monkeypatch.setenv("PRUSA_MCP_PATH", str(tmp_path / "nonexistent"))

    with pytest.raises(FileNotFoundError) as excinfo:
        main()

    assert "not found" in str(excinfo.value)


@pytest.mark.unit
def test_main_loads_module(monkeypatch, tmp_path):
    """Test that main loads the Prusa MCP module."""
    _create_fake_package(
        tmp_path,
        """
class FakeMCP:
    @property
    def sse_app(self):
        return lambda: None

mcp = FakeMCP()
""",
    )

    monkeypatch.setenv("PRUSA_MCP_PATH", str(tmp_path))
    monkeypatch.setenv("PRUSA_MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("PRUSA_MCP_PORT", "8000")

    # Mock uvicorn.run to prevent actual server startup
    with patch("prusa_mcp_server.server.uvicorn.run") as mock_run:
        main()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["host"] == "127.0.0.1"
        assert call_kwargs[1]["port"] == 8000


@pytest.mark.unit
def test_main_calls_app_factory(monkeypatch, tmp_path):
    """Test that main calls app factory if sse_app is callable."""
    _create_fake_package(
        tmp_path,
        """
class FakeApp:
    pass

class FakeMCP:
    def __init__(self):
        self.factory_called = False

    @property
    def sse_app(self):
        def factory():
            self.factory_called = True
            return FakeApp()
        return factory

mcp = FakeMCP()
""",
    )

    monkeypatch.setenv("PRUSA_MCP_PATH", str(tmp_path))
    monkeypatch.setenv("PRUSA_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("PRUSA_MCP_PORT", "8765")

    with patch("prusa_mcp_server.server.uvicorn.run") as mock_run:
        main()

        # Should have called the factory
        mock_run.assert_called_once()


@pytest.mark.unit
def test_main_default_host_and_port(monkeypatch, tmp_path):
    """Test main uses default host and port when not set."""
    _create_fake_package(
        tmp_path,
        """
class FakeMCP:
    @property
    def sse_app(self):
        return object()

mcp = FakeMCP()
""",
    )

    monkeypatch.setenv("PRUSA_MCP_PATH", str(tmp_path))
    monkeypatch.delenv("PRUSA_MCP_HOST", raising=False)
    monkeypatch.delenv("PRUSA_MCP_PORT", raising=False)

    with patch("prusa_mcp_server.server.uvicorn.run") as mock_run:
        main()

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 8000


# ============================================================================
# INTEGRATION-STYLE TESTS
# ============================================================================


@pytest.mark.unit
def test_get_prusa_mcp_path_returns_path_object(monkeypatch):
    """Test that get_prusa_mcp_path returns a Path object."""
    monkeypatch.setenv("PRUSA_MCP_PATH", "/some/path")

    result = get_prusa_mcp_path()

    assert isinstance(result, Path)


@pytest.mark.unit
def test_main_logs_info(monkeypatch, tmp_path, caplog):
    """Test that main logs information messages."""
    _create_fake_package(
        tmp_path,
        """
class FakeMCP:
    @property
    def sse_app(self):
        return object()

mcp = FakeMCP()
""",
    )

    monkeypatch.setenv("PRUSA_MCP_PATH", str(tmp_path))

    with patch("prusa_mcp_server.server.uvicorn.run"):
        import logging

        with caplog.at_level(logging.INFO):
            main()

    # Should have logged loading message
    assert any(
        "Loading" in record.message or "Starting" in record.message
        for record in caplog.records
    )
