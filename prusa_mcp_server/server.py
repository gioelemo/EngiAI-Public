"""
External Prusa MCP Server with SSE transport.

This server runs the Prusa MCP FastMCP server with SSE transport,
allowing it to run as a separate service that can be deployed independently.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_prusa_mcp_path() -> Path:
    """Get the path to the Prusa MCP server."""
    prusa_mcp_path = os.getenv(
        "PRUSA_MCP_PATH", str(Path.home() / "Desktop" / "prusa-mcp")
    )
    return Path(prusa_mcp_path)


def main():
    """Run the Prusa MCP server with SSE transport."""
    # Get prusa-mcp path
    prusa_mcp_path = get_prusa_mcp_path()
    prusa_mcp_file = prusa_mcp_path / "src" / "prusa-mcp.py"

    if not prusa_mcp_file.exists():
        msg = f"Prusa MCP server not found at {prusa_mcp_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading Prusa MCP server from {prusa_mcp_file}")

    # Load the prusa-mcp module
    spec = importlib.util.spec_from_file_location("prusa_mcp", prusa_mcp_file)
    if spec is None or spec.loader is None:
        msg = f"Could not load spec for {prusa_mcp_file}"
        logger.error(msg)
        raise ImportError(msg)

    prusa_mcp_module = importlib.util.module_from_spec(spec)
    sys.modules["prusa_mcp"] = prusa_mcp_module
    spec.loader.exec_module(prusa_mcp_module)

    # Get the FastMCP instance
    if not hasattr(prusa_mcp_module, "mcp"):
        msg = "Could not find 'mcp' (FastMCP instance) in prusa-mcp.py"
        logger.error(msg)
        raise AttributeError(msg)

    mcp_server = prusa_mcp_module.mcp
    logger.info(f"Found FastMCP server: {mcp_server}")

    # Get host and port from environment
    host = os.getenv("PRUSA_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("PRUSA_MCP_PORT", "8000"))

    logger.info(f"Starting Prusa MCP Server on {host}:{port} with SSE transport")

    # Get the SSE app from FastMCP
    app = mcp_server.sse_app

    # Run the app with uvicorn on the specified host/port
    logger.info(f"Running SSE server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
