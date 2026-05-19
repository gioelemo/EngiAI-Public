# Dockerfile for Prusa MCP Server
# Python version — keep default in sync with .python-version.
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Minimal system dependencies.
# prusa-mcp now authenticates via OAuth2 Bearer tokens (no browser automation),
# so no Playwright / Chromium / X libraries are required — git is only needed
# so pip can clone prusa-mcp from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the MCP server wrapper
COPY services/prusa_mcp_server /app/prusa_mcp_server
COPY services/prusa_mcp_server/requirements-mcp.txt /app/

# Install Python dependencies (prusa-mcp installed via pip from GitHub)
RUN pip install --no-cache-dir -r requirements-mcp.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8765/sse', timeout=5.0)" || exit 1

# Run the server as a script (not as a module to avoid RuntimeWarning)
CMD ["python", "/app/prusa_mcp_server/run_server.py"]
