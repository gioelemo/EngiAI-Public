#!/bin/bash
# Standalone script to run the Prusa MCP Server

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    # Source the .env in a POSIX-compatible way and export all variables.
    # Using `set -a` ensures variables defined in the file are exported to the environment.
    # This is more robust than parsing with xargs which can fail on comments or complex values.
    set -a
    . .env
    set +a
fi

# Default values
HOST="${PRUSA_MCP_HOST:-0.0.0.0}"
PORT="${PRUSA_MCP_PORT:-8765}"

echo "=========================================="
echo "Starting Prusa MCP Server"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "=========================================="

# Run the server
python -m prusa_mcp_server.server "$HOST" "$PORT"
