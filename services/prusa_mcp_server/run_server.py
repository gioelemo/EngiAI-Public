#!/usr/bin/env python3
"""
Entry point script for Prusa MCP Server.

This script runs as a direct script (not as a module) to avoid
the RuntimeWarning about sys.modules.
"""

if __name__ == "__main__":
    from prusa_mcp_server.server import main

    main()
