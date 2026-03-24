"""
External Prusa MCP Server with SSE transport.

This server runs the Prusa MCP FastMCP server with SSE transport,
allowing it to run as a separate service that can be deployed independently.
"""

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
    # Get prusa-mcp path and add its src/ to sys.path so we can import prusa_mcp
    prusa_mcp_path = get_prusa_mcp_path()
    src_path = prusa_mcp_path / "src"

    if not (src_path / "prusa_mcp" / "server.py").exists():
        msg = f"Prusa MCP server not found at {src_path / 'prusa_mcp' / 'server.py'}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Add src/ to sys.path so `import prusa_mcp` works
    sys.path.insert(0, str(src_path))
    logger.info(f"Loading Prusa MCP server from {src_path / 'prusa_mcp'}")

    # Configure transport security BEFORE loading the module
    from mcp.server.fastmcp.server import (  # noqa: PLC0415
        TransportSecuritySettings,  # pyright: ignore[reportPrivateImportUsage]
    )

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

    # Import the prusa_mcp package (now on sys.path)
    from prusa_mcp.server import (  # type: ignore[import-not-found]  # noqa: PLC0415
        mcp as mcp_server,
    )

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
