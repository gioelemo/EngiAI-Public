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
    # Default: prusa-mcp submodule in the same directory as this file
    default_path = Path(__file__).parent / "prusa-mcp"
    prusa_mcp_path = os.getenv("PRUSA_MCP_PATH", str(default_path))
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

    # Configure transport security BEFORE loading the module
    from mcp.server.fastmcp.server import TransportSecuritySettings  # noqa: PLC0415

    # Set allowed hosts to include Docker service name and localhost
    allowed_hosts_env = os.getenv(
        "MCP_ALLOWED_HOSTS",
        "localhost,127.0.0.1,prusa-mcp-server,prusa-mcp-server:8765",
    )
    allowed_hosts = [h.strip() for h in allowed_hosts_env.split(",")]
    logger.info(f"Configuring transport security with allowed hosts: {allowed_hosts}")

    # Patch FastMCP to use our transport security settings
    original_fastmcp_init = None
    try:
        from mcp.server.fastmcp.server import FastMCP as FastMCPClass  # noqa: PLC0415

        original_fastmcp_init = FastMCPClass.__init__

        def patched_init(self, *args, **kwargs):
            # Inject transport_security if not provided
            if "transport_security" not in kwargs:
                kwargs["transport_security"] = TransportSecuritySettings(
                    allowed_hosts=allowed_hosts
                )
            original_fastmcp_init(self, *args, **kwargs)

        FastMCPClass.__init__ = patched_init
        logger.info("Patched FastMCP initialization with transport security")
    except Exception as e:
        logger.warning(f"Could not patch FastMCP: {e}")

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

    # If app is a callable (factory function), call it to get the actual app instance
    if callable(app):
        logger.info("Detected app factory, calling it to get app instance")
        app = app()

    # Run the app with uvicorn on the specified host/port
    logger.info(f"Running SSE server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
