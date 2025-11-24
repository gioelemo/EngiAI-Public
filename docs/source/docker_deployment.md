# Docker Deployment Guide

This guide explains how to deploy the Engineer Assistant chatbot using Docker on Windows Server or any other platform.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)
- API keys for OpenAI and Tavily

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Create a `.env` file** from the example:
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** and add your API keys:
   ```bash
   OPENAI_API_KEY=your_actual_openai_key
   TAVILY_API_KEY=your_actual_tavily_key
   ```

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

5. **View logs**:
   ```bash
   docker-compose logs -f chatbot
   ```

6. **Stop the application**:
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker CLI

1. **Build the Docker image**:
   ```bash
   docker build -t engineer-assistant-chatbot .
   ```

2. **Run the container** with environment variables:
   ```bash
   docker run -d \
     --name engineer-assistant \
     -p 8501:8501 \
     -e OPENAI_API_KEY=your_openai_key \
     -e TAVILY_API_KEY=your_tavily_key \
     -v $(pwd)/data:/app/data \
     engineer-assistant-chatbot
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

## Windows Server Deployment

### Method 1: Build on Windows Server

1. Install Docker Desktop for Windows or Docker Engine
2. Clone your repository or copy the project files to the server
3. Follow the "Quick Start" instructions above

### Method 2: Transfer Image from Another Machine

If you build the image on your local machine and want to deploy on Windows Server:

1. **On your build machine** (Mac/Linux):
   ```bash
   # Build the image
   docker build -t engineer-assistant-chatbot .

   # Save the image to a tar file
   docker save -o engineer-assistant-chatbot.tar engineer-assistant-chatbot
   ```

2. **Transfer the tar file** to Windows Server (via SCP, FTP, USB, etc.)

3. **On Windows Server**:
   ```powershell
   # Load the image
   docker load -i engineer-assistant-chatbot.tar

   # Create a .env file with your API keys
   # Then run with Docker Compose or Docker CLI
   docker run -d `
     --name engineer-assistant `
     -p 8501:8501 `
     --env-file .env `
     -v ${PWD}/data:/app/data `
     engineer-assistant-chatbot
   ```

### Method 3: Using a Container Registry

1. **Push to Docker Hub** (or another registry):
   ```bash
   # Tag the image
   docker tag engineer-assistant-chatbot your-username/engineer-assistant-chatbot:latest

   # Login to Docker Hub
   docker login

   # Push the image
   docker push your-username/engineer-assistant-chatbot:latest
   ```

2. **On Windows Server**, pull and run:
   ```powershell
   docker pull your-username/engineer-assistant-chatbot:latest

   docker run -d `
     --name engineer-assistant `
     -p 8501:8501 `
     --env-file .env `
     -v ${PWD}/data:/app/data `
     your-username/engineer-assistant-chatbot:latest
   ```

## Environment Variables

### Required Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key

### Important Variables
- `SKIP_MCP`: Skip Prusa MCP server initialization (default: `true` in Docker)
  - Set to `true` for basic deployment without 3D printer integration
  - Set to `false` only if you have the Prusa MCP server set up

### Optional Variables
- `MATHPIX_API_ID`: MathPix API ID (for PDF text extraction)
- `MATHPIX_API_KEY`: MathPix API key
- `LLM_MODEL`: Model to use (default: `openai:gpt-4o`)
- `EMBEDDINGS_MODEL`: Embeddings model (default: `text-embedding-3-small`)
- `DATABASE_URL`: Database connection string (default: `sqlite:///data/conversations.db`)
- `LANGCHAIN_TRACING`: Enable LangSmith tracing (default: `false`)

See `.env.example` in the project root for all available configuration options.

### About MCP Server (Prusa 3D Printer Integration)

The application includes optional integration with Prusa Connect via an MCP (Model Context Protocol) server for 3D printer management. For most deployments, this is not needed and should be disabled by setting `SKIP_MCP=true`.

**To enable Prusa integration** (advanced):
1. Set up the Prusa MCP server at the path specified in `PRUSA_MCP_PATH`
2. Set `SKIP_MCP=false` in your `.env` file
3. Configure `PRUSA_MCP_PATH` environment variable in your `.env` file

For basic chatbot functionality, keep `SKIP_MCP=true` (the default).

### About HPC Cluster Integration (SSH Access)

The Docker configuration **automatically mounts your SSH configuration** from `~/.ssh` into the container. This enables:
- Connection to HPC clusters (like ETH's Euler cluster)
- SLURM job submission and monitoring
- Remote file operations

**How it works:**
1. Your `~/.ssh` directory is mounted read-only to `/root/.ssh-host` in the container
2. The `docker-entrypoint.sh` script copies the SSH configuration and sets proper permissions (600 for keys, 644 for known_hosts)
3. SSH client is pre-installed in the container

**What you need:**
- Your `~/.ssh/config` file with host aliases (e.g., "euler")
- SSH private keys in `~/.ssh/`
- The remote host's entry in `~/.ssh/known_hosts` (add it by connecting once from your host machine)

**For passphrase-protected SSH keys:**
If your SSH key is protected with a passphrase, you need to use SSH agent forwarding:

1. **Start SSH agent and add your key** (on your host machine):
   ```bash
   # Start the SSH agent
   eval "$(ssh-agent -s)"

   # Add your SSH key (you'll be prompted for the passphrase once)
   ssh-add ~/.ssh/id_rsa  # or id_ed25519, or whatever your key is named

   # Verify the key is loaded
   ssh-add -l
   ```

2. **Start the container** (the SSH agent socket is automatically mounted):
   ```bash
   docker-compose up -d
   ```

3. **The container will use your host's SSH agent** - no passphrase required!

**Alternative: Use an unencrypted key** (less secure):
If you prefer not to use SSH agent, you can create a separate SSH key without a passphrase specifically for the container:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_docker_euler -N ""
# Then add this key to your Euler account's authorized_keys
```

**Testing SSH connection:**
Once the container is running, you can test the SSH connection through the chatbot UI by asking it to connect to the HPC cluster.

**Security notes:**
- SSH keys are mounted read-only from your host
- Keys are copied into the container (not shared directly)
- Container is isolated from your host system
- For production, consider using SSH agent forwarding or secrets management

## Data Persistence

The application stores conversation data in the `/app/data` directory. To persist this data:

1. **Using Docker Compose**: The `docker-compose.yml` already mounts `./data:/app/data`

2. **Using Docker CLI**: Add the volume mount:
   ```bash
   -v $(pwd)/data:/app/data
   ```

## Troubleshooting

### Container won't start - Configuration Error

**Error**: `ValueError` in config validation

**Solution**: Ensure `OPENAI_API_KEY` and `TAVILY_API_KEY` are set in your `.env` file or passed as environment variables.

### Port 8501 already in use

**Solution**: Either stop the service using port 8501 or map to a different port:
```bash
docker run -p 8080:8501 ...  # Access on http://localhost:8080
```

### Database connection issues

**Solution**: If using PostgreSQL, ensure the database is accessible from the container. For SQLite (default), ensure the data directory is mounted.

### Viewing container logs

```bash
# Using Docker Compose
docker-compose logs -f chatbot

# Using Docker CLI
docker logs -f engineer-assistant
```

### Restarting the container

```bash
# Using Docker Compose
docker-compose restart

# Using Docker CLI
docker restart engineer-assistant
```

## Health Check

The application includes a health check endpoint. You can verify the application is running:

```bash
curl http://localhost:8501/_stcore/health
```

## Updating the Application

1. **Pull latest code** from your repository
2. **Rebuild the image**:
   ```bash
   docker-compose build
   ```
3. **Restart the container**:
   ```bash
   docker-compose up -d
   ```

## Security Best Practices

1. **Never commit `.env` files** with real API keys to version control
2. **Use secrets management** in production (Docker secrets, Kubernetes secrets, etc.)
3. **Restrict network access** to the container if needed
4. **Regularly update** the base image and dependencies

## Production Considerations

### Using Docker Swarm or Kubernetes

For production deployments, consider:
- Using orchestration platforms (Docker Swarm, Kubernetes)
- Implementing load balancing
- Setting up automated backups for the data directory
- Using managed databases instead of SQLite
- Implementing monitoring and alerting

### Resource Limits

Add resource limits to prevent excessive resource usage:

```yaml
# In docker-compose.yml
services:
  chatbot:
    # ... other configuration ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Support

For issues and questions:
- Check the main README.md in the project root
- Review the [Database Setup Guide](database_setup.md) for database configuration
- Check Docker logs for error messages
