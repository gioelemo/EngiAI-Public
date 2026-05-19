# Python version — keep default in sync with .python-version.
# Override at build time with: docker build --build-arg PYTHON_VERSION=$(cat .python-version) .
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

# Set working directory (Streamlit requires non-root directory)
WORKDIR /app

# Install system dependencies including Node.js for custom component
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better layer caching
COPY pyproject.toml ./

# Install Python dependencies
# Use pip with pyproject.toml (poetry not required for installation)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Copy and set permissions for entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose Streamlit port
EXPOSE 8501

# Configure health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set environment variable for Streamlit
ENV PYTHONPATH=/app:/app/services
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Use custom entrypoint to set up SSH, then run Streamlit
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
