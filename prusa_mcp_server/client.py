"""
HTTP/SSE Client for connecting to external Prusa MCP Server.

This client replaces the stdio transport with HTTP/SSE transport,
allowing the agent to connect to an externally running MCP server.
"""

import asyncio
import concurrent.futures
import contextlib
import logging
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

# Try to import nest_asyncio for better async compatibility
try:
    import nest_asyncio  # type: ignore[import-not-found]

    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False


class PrusaMCPClient:
    """Client for connecting to external Prusa MCP server via HTTP/SSE."""

    def __init__(self, server_url: str = "http://localhost:8765"):
        """Initialize the MCP client.

        Args:
            server_url: URL of the external MCP server (default: http://localhost:8765)
        """
        self.server_url = server_url.rstrip("/")
        self.session: ClientSession | None = None
        self._client_context: Any = None
        self._session_context: Any = None

    async def connect(self) -> ClientSession:
        """Connect to the MCP server and return an initialized session."""
        logger.info(f"Connecting to Prusa MCP server at {self.server_url}")

        # Create SSE client connection
        self._client_context = sse_client(f"{self.server_url}/sse")
        read, write = await self._client_context.__aenter__()  # type: ignore[attr-defined]

        # Create client session
        self._session_context = ClientSession(read, write)
        self.session = await self._session_context.__aenter__()  # type: ignore[attr-defined]

        # Initialize the session
        await self.session.initialize()

        logger.info("Connected to Prusa MCP server successfully")
        return self.session

    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if self._session_context:
                with contextlib.suppress(RuntimeError, GeneratorExit):
                    await self._session_context.__aexit__(None, None, None)
            if self._client_context:
                with contextlib.suppress(RuntimeError, GeneratorExit):
                    await self._client_context.__aexit__(None, None, None)
        finally:
            self.session = None
            self._session_context = None
            self._client_context = None
            logger.info("Disconnected from Prusa MCP server")

    async def list_tools(self) -> list:
        """List available tools from the MCP server."""
        if not self.session:
            await self.connect()

        response = await self.session.list_tools()  # type: ignore[union-attr]
        return response.tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as a dictionary

        Returns:
            Tool result as a string
        """
        try:
            # Always ensure we have a fresh connection for tool calls
            # This handles the case where the session was created in a different async context
            if not self.session:
                logger.info(f"No active session for tool '{tool_name}', connecting...")
                await self.connect()

            # Try the tool call, reconnect once if it fails
            try:
                logger.debug(f"Calling tool '{tool_name}' with arguments: {arguments}")
                result = await self.session.call_tool(tool_name, arguments)  # type: ignore[union-attr]
                logger.debug(f"Tool '{tool_name}' result: {result}")
            except Exception as e:
                logger.warning(f"Tool call failed, attempting to reconnect: {e}")
                # Disconnect and reconnect
                await self.disconnect()
                await self.connect()
                # Retry the call
                logger.debug(f"Retrying tool '{tool_name}' with arguments: {arguments}")
                result = await self.session.call_tool(tool_name, arguments)  # type: ignore[union-attr]
                logger.debug(f"Tool '{tool_name}' result after retry: {result}")

            if result.content:
                return "\n".join(
                    [c.text if hasattr(c, "text") else str(c) for c in result.content]
                )
            else:
                return "Tool executed successfully"
        except Exception:
            logger.exception(f"Error in async call_tool for {tool_name}")
            raise

    async def check_health(self) -> bool:
        """Check if the MCP server is healthy and reachable."""
        http_ok = 200
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.server_url}/sse", timeout=5.0, follow_redirects=True
                )
                return response.status_code == http_ok
        except Exception:
            logger.exception("Health check failed")
            return False

    def call_tool_sync(self, tool_name: str, **kwargs) -> str:
        """Synchronous wrapper for calling tools.

        This is useful for integration with LangChain tools which expect sync functions.
        """
        try:
            logger.info(f"call_tool_sync: Starting call for '{tool_name}'")

            # IMPORTANT: Create a completely fresh connection for each tool call
            # This avoids issues with reusing sessions across different async contexts
            async def _call_with_fresh_connection():
                """Create a fresh connection and call the tool."""
                # Create a new client instance for this call
                temp_client = PrusaMCPClient(self.server_url)
                try:
                    await temp_client.connect()
                    result = await temp_client.call_tool(tool_name, kwargs)
                    return result
                finally:
                    # Always disconnect after the call
                    await temp_client.disconnect()

            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    logger.info(
                        "call_tool_sync: Event loop is closed, creating new one"
                    )
                    loop = None  # type: ignore[assignment]
                else:
                    logger.info(
                        f"call_tool_sync: Found event loop, running={loop.is_running()}"
                    )
            except RuntimeError as e:
                logger.info(f"call_tool_sync: No event loop: {e}")
                loop = None  # type: ignore[assignment]

            if loop is None:
                # No event loop, create a new one
                logger.info("call_tool_sync: Creating new event loop")
                return asyncio.run(_call_with_fresh_connection())

            # Check if loop is running
            if loop.is_running():
                logger.info(
                    "call_tool_sync: Loop is running, using nest_asyncio or thread pool"
                )
                if NEST_ASYNCIO_AVAILABLE:
                    logger.info("call_tool_sync: Applying nest_asyncio")
                    nest_asyncio.apply(loop)
                    result = loop.run_until_complete(_call_with_fresh_connection())
                    logger.info(
                        f"call_tool_sync: Got result from nest_asyncio: {result[:100] if result else '(empty)'}"
                    )
                    return result
                else:
                    # Fall back to thread pool
                    logger.info("call_tool_sync: Using thread pool executor")
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(_call_with_fresh_connection())
                        )
                        return future.result()

            # Loop exists but not running - run in new thread to avoid blocking
            logger.info(
                "call_tool_sync: Loop exists but not running, using thread pool to avoid blocking"
            )
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(_call_with_fresh_connection())
                )
                return future.result()

        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}")
            return f"Error calling tool {tool_name}: {e!s}"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
