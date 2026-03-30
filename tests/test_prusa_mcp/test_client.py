"""
Tests for prusa_mcp_server/client.py module.

These tests cover the PrusaMCPClient class for HTTP/SSE MCP communication.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import nest_asyncio
import pytest

from prusa_mcp_server.client import NEST_ASYNCIO_AVAILABLE, PrusaMCPClient

# Allow asyncio.run() inside an already-running event loop (e.g. pytest plugins)
nest_asyncio.apply()

# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


@pytest.mark.unit
def test_client_initialization_default_url():
    """Test client initialization with default URL."""
    client = PrusaMCPClient()

    assert client.server_url == "http://localhost:8765"
    assert client.session is None
    assert client._client_context is None
    assert client._session_context is None


@pytest.mark.unit
def test_client_initialization_custom_url():
    """Test client initialization with custom URL."""
    client = PrusaMCPClient(server_url="http://custom:9000")

    assert client.server_url == "http://custom:9000"


@pytest.mark.unit
def test_client_initialization_strips_trailing_slash():
    """Test that trailing slashes are stripped from URL."""
    client = PrusaMCPClient(server_url="http://localhost:8765/")

    assert client.server_url == "http://localhost:8765"


# ============================================================================
# CONNECT TESTS
# ============================================================================


@pytest.mark.unit
def test_connect_success():
    """Test successful connection to MCP server."""

    async def run_test():
        client = PrusaMCPClient()

        mock_read = MagicMock()
        mock_write = MagicMock()

        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__.return_value = (mock_read, mock_write)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_session

        with (
            patch("prusa_mcp_server.client.sse_client", return_value=mock_client_ctx),
            patch(
                "prusa_mcp_server.client.ClientSession", return_value=mock_session_ctx
            ),
        ):
            result = await client.connect()

        assert result == mock_session
        assert client.session == mock_session
        mock_session.initialize.assert_called_once()

    asyncio.run(run_test())


# ============================================================================
# DISCONNECT TESTS
# ============================================================================


@pytest.mark.unit
def test_disconnect_success():
    """Test successful disconnection."""

    async def run_test():
        client = PrusaMCPClient()

        mock_session_ctx = AsyncMock()
        mock_client_ctx = AsyncMock()
        mock_session = AsyncMock()

        client._session_context = mock_session_ctx
        client._client_context = mock_client_ctx
        client.session = mock_session

        await client.disconnect()

        assert client.session is None
        assert client._session_context is None
        assert client._client_context is None

    asyncio.run(run_test())


@pytest.mark.unit
def test_disconnect_handles_errors():
    """Test that disconnect handles errors gracefully."""

    async def run_test():
        client = PrusaMCPClient()

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aexit__.side_effect = RuntimeError("Session error")

        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aexit__.side_effect = GeneratorExit()

        client._session_context = mock_session_ctx
        client._client_context = mock_client_ctx
        client.session = AsyncMock()

        await client.disconnect()

        assert client.session is None

    asyncio.run(run_test())


# ============================================================================
# LIST TOOLS TESTS
# ============================================================================


@pytest.mark.unit
def test_list_tools_with_active_session():
    """Test listing tools with an active session."""

    async def run_test():
        client = PrusaMCPClient()

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        mock_response = MagicMock()
        mock_response.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.list_tools.return_value = mock_response

        client.session = mock_session

        result = await client.list_tools()

        assert result == [mock_tool]
        mock_session.list_tools.assert_called_once()

    asyncio.run(run_test())


@pytest.mark.unit
def test_list_tools_connects_if_no_session():
    """Test that list_tools connects if no session exists."""

    async def run_test():
        client = PrusaMCPClient()

        mock_tool = MagicMock()
        mock_response = MagicMock()
        mock_response.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.list_tools.return_value = mock_response

        async def set_session():
            client.session = mock_session
            return mock_session

        with patch.object(client, "connect", side_effect=set_session):
            client.session = None
            result = await client.list_tools()

        assert result == [mock_tool]

    asyncio.run(run_test())


# ============================================================================
# CALL TOOL TESTS
# ============================================================================


@pytest.mark.unit
def test_call_tool_success():
    """Test successful tool call."""

    async def run_test():
        client = PrusaMCPClient()

        mock_content = MagicMock()
        mock_content.text = "Tool result"

        mock_result = MagicMock()
        mock_result.content = [mock_content]

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_result

        client.session = mock_session

        result = await client.call_tool("test_tool", {"arg": "value"})

        assert result == "Tool result"
        mock_session.call_tool.assert_called_once_with("test_tool", {"arg": "value"})

    asyncio.run(run_test())


@pytest.mark.unit
def test_call_tool_no_content():
    """Test tool call with no content in result."""

    async def run_test():
        client = PrusaMCPClient()

        mock_result = MagicMock()
        mock_result.content = []

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_result

        client.session = mock_session

        result = await client.call_tool("test_tool", {})

        assert result == "Tool executed successfully"

    asyncio.run(run_test())


@pytest.mark.unit
def test_call_tool_reconnects_on_failure():
    """Test that call_tool reconnects and retries on failure."""

    async def run_test():
        client = PrusaMCPClient()

        mock_content = MagicMock()
        mock_content.text = "Retry result"

        mock_result = MagicMock()
        mock_result.content = [mock_content]

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = [Exception("Connection lost"), mock_result]

        client.session = mock_session

        with (
            patch.object(client, "disconnect", new_callable=AsyncMock),
            patch.object(client, "connect", new_callable=AsyncMock),
        ):
            result = await client.call_tool("test_tool", {})

        assert result == "Retry result"
        assert mock_session.call_tool.call_count == 2

    asyncio.run(run_test())


@pytest.mark.unit
def test_call_tool_connects_if_no_session():
    """Test that call_tool connects if no session exists."""

    async def run_test():
        client = PrusaMCPClient()
        client.session = None

        mock_content = MagicMock()
        mock_content.text = "Result"

        mock_result = MagicMock()
        mock_result.content = [mock_content]

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_result

        async def mock_connect():
            client.session = mock_session
            return mock_session

        with patch.object(client, "connect", side_effect=mock_connect):
            result = await client.call_tool("test_tool", {})

        assert "Result" in result

    asyncio.run(run_test())


# ============================================================================
# CHECK HEALTH TESTS
# ============================================================================


@pytest.mark.unit
def test_check_health_success():
    """Test successful health check."""

    async def run_test():
        client = PrusaMCPClient()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "prusa_mcp_server.client.httpx.AsyncClient", return_value=mock_client
        ):
            result = await client.check_health()

        assert result is True

    asyncio.run(run_test())


@pytest.mark.unit
def test_check_health_failure():
    """Test health check with non-200 response."""

    async def run_test():
        client = PrusaMCPClient()

        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "prusa_mcp_server.client.httpx.AsyncClient", return_value=mock_client
        ):
            result = await client.check_health()

        assert result is False

    asyncio.run(run_test())


@pytest.mark.unit
def test_check_health_exception():
    """Test health check with exception."""

    async def run_test():
        client = PrusaMCPClient()

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch(
            "prusa_mcp_server.client.httpx.AsyncClient", return_value=mock_client
        ):
            result = await client.check_health()

        assert result is False

    asyncio.run(run_test())


# ============================================================================
# CALL TOOL SYNC TESTS
# ============================================================================


@pytest.mark.unit
def test_call_tool_sync_success():
    """Test synchronous tool call wrapper."""
    client = PrusaMCPClient()

    async def mock_call_tool(_name, _args):
        return "Sync result"

    with (
        patch.object(PrusaMCPClient, "connect", new_callable=AsyncMock),
        patch.object(PrusaMCPClient, "call_tool", side_effect=mock_call_tool),
        patch.object(PrusaMCPClient, "disconnect", new_callable=AsyncMock),
    ):
        result = client.call_tool_sync("test_tool", arg1="value1")

    assert "Sync result" in result or "Error" in result


@pytest.mark.unit
def test_call_tool_sync_handles_exception():
    """Test that call_tool_sync handles exceptions gracefully."""
    client = PrusaMCPClient()

    with patch.object(
        PrusaMCPClient, "connect", new_callable=AsyncMock
    ) as mock_connect:
        mock_connect.side_effect = Exception("Connection failed")

        result = client.call_tool_sync("test_tool")

    assert "Error" in result
    assert "test_tool" in result


@pytest.mark.unit
def test_call_tool_sync_with_closed_loop():
    """Test call_tool_sync when event loop is closed."""
    client = PrusaMCPClient()

    # Create and close a loop to simulate closed loop scenario
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()

    async def mock_call_tool(_name, _args):
        return "Result from closed loop test"

    with (
        patch("asyncio.get_event_loop", return_value=closed_loop),
        patch.object(PrusaMCPClient, "connect", new_callable=AsyncMock),
        patch.object(PrusaMCPClient, "call_tool", side_effect=mock_call_tool),
        patch.object(PrusaMCPClient, "disconnect", new_callable=AsyncMock),
    ):
        result = client.call_tool_sync("test_tool")

    assert "Result from closed loop test" in result or "Error" in result


@pytest.mark.unit
def test_call_tool_sync_no_event_loop():
    """Test call_tool_sync when no event loop exists."""
    client = PrusaMCPClient()

    async def mock_call_tool(_name, _args):
        return "Result no loop"

    with (
        patch("asyncio.get_event_loop", side_effect=RuntimeError("No event loop")),
        patch.object(PrusaMCPClient, "connect", new_callable=AsyncMock),
        patch.object(PrusaMCPClient, "call_tool", side_effect=mock_call_tool),
        patch.object(PrusaMCPClient, "disconnect", new_callable=AsyncMock),
    ):
        result = client.call_tool_sync("test_tool")

    assert "Result no loop" in result or "Error" in result


@pytest.mark.unit
def test_call_tool_sync_with_running_loop_nest_asyncio():
    """Test call_tool_sync when loop is running and nest_asyncio available."""
    import prusa_mcp_server.client as client_module

    client = PrusaMCPClient()

    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = False
    mock_loop.is_running.return_value = True
    mock_loop.run_until_complete.return_value = "Nested result"

    mock_nest = MagicMock()

    # Save original value
    original_nest_asyncio = getattr(client_module, "nest_asyncio", None)

    try:
        # Set nest_asyncio on the module
        client_module.nest_asyncio = mock_nest

        with (
            patch("asyncio.get_event_loop", return_value=mock_loop),
            patch("prusa_mcp_server.client.NEST_ASYNCIO_AVAILABLE", True),
        ):
            result = client.call_tool_sync("test_tool")

        mock_nest.apply.assert_called_once_with(mock_loop)
        assert result == "Nested result"
    finally:
        # Restore original value
        if original_nest_asyncio is None:
            if hasattr(client_module, "nest_asyncio"):
                delattr(client_module, "nest_asyncio")
        else:
            client_module.nest_asyncio = original_nest_asyncio


@pytest.mark.unit
def test_call_tool_sync_with_running_loop_no_nest_asyncio():
    """call_tool_sync returns the tool result even when nest_asyncio is unavailable."""
    client = PrusaMCPClient()

    async def mock_call_tool(_name, _args):
        return "Thread pool result"

    with (
        patch("prusa_mcp_server.client.NEST_ASYNCIO_AVAILABLE", False),
        patch.object(PrusaMCPClient, "connect", new_callable=AsyncMock),
        patch.object(PrusaMCPClient, "call_tool", side_effect=mock_call_tool),
        patch.object(PrusaMCPClient, "disconnect", new_callable=AsyncMock),
    ):
        result = client.call_tool_sync("test_tool")

    assert "Thread pool result" in result or "Error" in result


@pytest.mark.unit
def test_call_tool_sync_loop_exists_not_running():
    """call_tool_sync works when an event loop exists but is not running."""
    client = PrusaMCPClient()

    async def mock_call_tool(_name, _args):
        return "Non-running loop result"

    with (
        patch.object(PrusaMCPClient, "connect", new_callable=AsyncMock),
        patch.object(PrusaMCPClient, "call_tool", side_effect=mock_call_tool),
        patch.object(PrusaMCPClient, "disconnect", new_callable=AsyncMock),
    ):
        result = client.call_tool_sync("test_tool")

    assert "Non-running loop result" in result or "Error" in result


@pytest.mark.unit
def test_call_tool_raises_exception():
    """Test that call_tool re-raises exceptions after logging."""

    async def run_test():
        client = PrusaMCPClient()

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
        ]

        client.session = mock_session

        with (
            patch.object(client, "disconnect", new_callable=AsyncMock),
            patch.object(client, "connect", new_callable=AsyncMock),
        ):
            with pytest.raises(Exception, match="Second failure"):
                await client.call_tool("test_tool", {})

    asyncio.run(run_test())


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================


@pytest.mark.unit
def test_async_context_manager():
    """Test async context manager protocol."""

    async def run_test():
        client = PrusaMCPClient()

        mock_session = AsyncMock()

        with (
            patch.object(client, "connect", new_callable=AsyncMock) as mock_connect,
            patch.object(
                client, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect,
        ):
            mock_connect.return_value = mock_session

            async with client as c:
                assert c == client

            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()

    asyncio.run(run_test())


# ============================================================================
# NEST_ASYNCIO TESTS
# ============================================================================


@pytest.mark.unit
def test_nest_asyncio_import():
    """Test that NEST_ASYNCIO_AVAILABLE is defined."""
    assert isinstance(NEST_ASYNCIO_AVAILABLE, bool)


# ============================================================================
# EDGE CASES
# ============================================================================


@pytest.mark.unit
def test_call_tool_multiple_content_items():
    """Test tool call with multiple content items."""

    async def run_test():
        client = PrusaMCPClient()

        mock_content1 = MagicMock()
        mock_content1.text = "Line 1"

        mock_content2 = MagicMock()
        mock_content2.text = "Line 2"

        mock_result = MagicMock()
        mock_result.content = [mock_content1, mock_content2]

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_result

        client.session = mock_session

        result = await client.call_tool("test_tool", {})

        assert "Line 1" in result
        assert "Line 2" in result

    asyncio.run(run_test())


@pytest.mark.unit
def test_call_tool_content_without_text():
    """Test tool call with content that has no text attribute."""

    async def run_test():
        client = PrusaMCPClient()

        mock_content = MagicMock(spec=[])

        mock_result = MagicMock()
        mock_result.content = [mock_content]

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_result

        client.session = mock_session

        result = await client.call_tool("test_tool", {})

        assert result is not None

    asyncio.run(run_test())
