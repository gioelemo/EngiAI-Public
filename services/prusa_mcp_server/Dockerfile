# Dockerfile for Prusa MCP Server
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including Playwright requirements
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the MCP server wrapper
COPY services/prusa_mcp_server /app/prusa_mcp_server
COPY services/prusa_mcp_server/requirements-mcp.txt /app/

# Install Python dependencies (prusa-mcp will be mounted at runtime)
RUN pip install --no-cache-dir -r requirements-mcp.txt

# Install Playwright browsers
RUN pip install --no-cache-dir playwright
RUN playwright install chromium --with-deps

# Set environment variables
ENV PRUSA_MCP_PATH=/app/prusa-mcp
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8765/sse', timeout=5.0)" || exit 1

# Run the server as a script (not as a module to avoid RuntimeWarning)
CMD ["python", "/app/prusa_mcp_server/run_server.py"]
