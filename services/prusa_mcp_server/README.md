# Prusa MCP Server - External Deployment

This directory contains the externalized Prusa MCP server that can run as a standalone service, separate from your main chatbot application.

## Architecture

```
┌─────────────────────┐      HTTP/SSE      ┌──────────────────────┐
│   Main Chatbot      │ ◄─────────────────► │  Prusa MCP Server    │
│   (Agent Code)      │    Port 8765        │  (Standalone)        │
└─────────────────────┘                     └──────────────────────┘
         │                                            │
         │                                            │
         ▼                                            ▼
   LangGraph Agents                          Prusa Connect API
   (Engineering, HPC, etc.)                  (3D Printer Control)
```

## Files

- **`server.py`**: HTTP/SSE wrapper for the Prusa MCP server
- **`client.py`**: HTTP client for connecting to the external server
- **`Dockerfile`**: Docker image for the standalone server
- **`run.sh`**: Bash script to run the server locally
- **`README.md`**: This file

## Deployment Options

### Option 1: Run Locally (Development)

**1. Install dependencies:**
```bash
cd /Users/gioelemolinari/Desktop/engiai
pip install -r requirements-mcp.txt
```

**2. Start the MCP server:**
```bash
./services/prusa_mcp_server/run.sh
```

**3. In another terminal, start your chatbot:**
```bash
export SKIP_MCP=false
export PRUSA_MCP_URL=http://localhost:8765
streamlit run src/ui/streamlit_app.py
```

### Option 2: Docker Compose (Recommended for Production)

**1. Ensure your prusa-mcp directory is available:**
```bash
# Copy or symlink prusa-mcp into the project directory
cp -r ~/Desktop/prusa-mcp ./prusa-mcp
# OR create a symlink
ln -s ~/Desktop/prusa-mcp ./prusa-mcp
```

**2. Start both services:**
```bash
docker-compose up -d
```

This will start:
- **Prusa MCP Server** on port 8765
- **Main Chatbot** on port 8501 (connected to MCP server)

**3. View logs:**
```bash
# MCP Server logs
docker-compose logs -f prusa-mcp-server

# Chatbot logs
docker-compose logs -f chatbot
```

**4. Stop services:**
```bash
docker-compose down
```

### Option 3: Separate Docker Containers (Advanced)

Deploy the MCP server and chatbot on different machines:

**On Server A (MCP Server):**
```bash
# Build the MCP server image
docker build -f services/prusa_mcp_server/Dockerfile -t prusa-mcp-server .

# Run the MCP server (tokens provisioned on the host via `make prusa-login`)
docker run -d \
  --name prusa-mcp-server \
  -p 8765:8765 \
  -e PRUSA_TOKEN_FILE=/app/data/prusa_tokens.json \
  -v $(pwd)/data:/app/data \
  prusa-mcp-server
```

**On Server B (Chatbot):**
```bash
# Set the MCP server URL to point to Server A
docker run -d \
  --name engiai \
  -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  -e TAVILY_API_KEY=your-key \
  -e SKIP_MCP=false \
  -e PRUSA_MCP_URL=http://server-a-ip:8765 \
  engiai-chatbot
```

### Option 4: Kubernetes Deployment

**1. Create MCP Server Deployment:**
```yaml
# mcp-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prusa-mcp-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prusa-mcp-server
  template:
    metadata:
      labels:
        app: prusa-mcp-server
    spec:
      containers:
      - name: prusa-mcp-server
        image: your-registry/prusa-mcp-server:latest
        ports:
        - containerPort: 8765
        env:
        - name: PRUSA_TOKEN_FILE
          value: /app/data/prusa_tokens.json
        volumeMounts:
        - name: prusa-tokens
          mountPath: /app/data
      volumes:
      - name: prusa-tokens
        secret:
          secretName: prusa-tokens  # contains prusa_tokens.json created by `make prusa-login`
---
apiVersion: v1
kind: Service
metadata:
  name: prusa-mcp-server
spec:
  selector:
    app: prusa-mcp-server
  ports:
  - port: 8765
    targetPort: 8765
```

**2. Deploy:**
```bash
kubectl apply -f mcp-server-deployment.yaml
```

**3. Configure chatbot to use the service:**
```yaml
env:
- name: PRUSA_MCP_URL
  value: "http://prusa-mcp-server:8765"
- name: SKIP_MCP
  value: "false"
```

## Environment Variables

### MCP Server Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRUSA_MCP_PATH` | Optional override pointing to an external `prusa-mcp` source checkout (rarely needed — the server uses the pip-installed `prusa_mcp` package by default) | unset |
| `PRUSA_MCP_HOST` | Host to bind the server to | `0.0.0.0` |
| `PRUSA_MCP_PORT` | Port to run the server on | `8765` |
| `PRUSA_TOKEN_FILE` | OAuth2 token file (run `make prusa-login` on the host to populate) | `~/.config/prusa-mcp/tokens.json` |

### Chatbot Client Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SKIP_MCP` | Skip MCP integration | `true` |
| `PRUSA_MCP_URL` | URL of external MCP server | `http://localhost:8765` |

## Testing the Connection

**1. Check if MCP server is running:**
```bash
curl http://localhost:8765/sse
```

**2. Test from Python:**
```python
from prusa_mcp_server.client import PrusaMCPClient
import asyncio

async def test():
    client = PrusaMCPClient("http://localhost:8765")

    # Check health
    healthy = await client.check_health()
    print(f"Server healthy: {healthy}")

    # List tools
    tools = await client.list_tools()
    print(f"Available tools: {[t.name for t in tools]}")

asyncio.run(test())
```

## Troubleshooting

### Server won't start

**Error**: `FileNotFoundError: Prusa MCP server not found`

**Solution**: Initialize the git submodule:
```bash
git submodule update --init --recursive
```
Or set `PRUSA_MCP_PATH` to a custom location if needed.

### Connection refused

**Error**: `Connection refused to http://localhost:8765`

**Solution**:
1. Check if server is running: `docker ps` or `ps aux | grep prusa`
2. Check firewall rules
3. Verify port is not in use: `lsof -i :8765`

### Docker: Can't find prusa-mcp

**Solution**: Copy or mount prusa-mcp into the build context:
```bash
# Copy
cp -r ~/Desktop/prusa-mcp ./

# Or update Dockerfile to mount from external location
```

### Chatbot can't connect to MCP server

**Solution**:
1. Ensure `SKIP_MCP=false` is set
2. Check `PRUSA_MCP_URL` points to correct host/port
3. Verify network connectivity between containers
4. Check MCP server logs for errors

## Security Considerations

1. **Authentication**: Consider adding API key authentication for production
2. **Network**: Use private networks in Docker/K8s
3. **TLS**: Enable HTTPS for production deployments
4. **Credentials**: Use secrets management for Prusa Connect credentials
5. **Firewall**: Restrict access to MCP server port (8765)

## Benefits of External Deployment

✅ **Scalability**: Run multiple chatbot instances with one MCP server
✅ **Isolation**: MCP server crashes don't affect chatbot
✅ **Updates**: Update MCP server without redeploying chatbot
✅ **Monitoring**: Separate logs and metrics for each service
✅ **Resource Management**: Allocate resources independently
✅ **Security**: Network isolation and firewall rules

## Migration from Embedded MCP

The codebase has been updated to support both embedded and external MCP:

**Old (Embedded):**
```python
# Uses stdio transport internally
prusa_agent = PrusaAgent(skip_mcp=False)
```

**New (External):**
```bash
# Set environment variables
export SKIP_MCP=false
export PRUSA_MCP_URL=http://localhost:8765

# Start external MCP server first
./services/prusa_mcp_server/run.sh

# Then start chatbot (automatically uses HTTP client)
streamlit run src/ui/streamlit_app.py
```

The agent code automatically detects `PRUSA_MCP_URL` and switches to HTTP transport!
