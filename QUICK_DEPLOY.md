# Quick Deployment Guide - Windows Server WSL

## Quick Start (5 Minutes)

### 1. Prerequisites Check

```powershell
# Check if WSL is installed
wsl --list --verbose

# Check if Docker is running
docker --version
```

### 2. Transfer and Extract

```bash
# In WSL Ubuntu terminal
mkdir -p ~/deployments && cd ~/deployments

# Copy from Windows (adjust path as needed)
cp /mnt/c/Downloads/engineer-assistant-v1.0.0.zip .
cp /mnt/c/Downloads/prusa-mcp-server-v1.0.0.zip .

# Extract
unzip engineer-assistant-v1.0.0.zip -d engineer-assistant
unzip prusa-mcp-server-v1.0.0.zip -d .

cd engineer-assistant
```

### 3. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

**Minimum required configuration:**
```bash
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here
PRUSA_MCP_LOCAL_PATH=/home/your-username/deployments/prusa-mcp
```

### 4. Deploy

```bash
# Build and start all services
docker-compose -f docker-compose.mcp.yml up -d

# Check status (wait ~30 seconds for health checks)
docker-compose -f docker-compose.mcp.yml ps
```

### 5. Access

Open browser to: **http://localhost:8501**

Or from Windows host: **http://<wsl-ip>:8501**

Find WSL IP:
```bash
hostname -I
```

## Common Commands

```bash
# View logs
docker-compose -f docker-compose.mcp.yml logs -f

# Restart
docker-compose -f docker-compose.mcp.yml restart

# Stop
docker-compose -f docker-compose.mcp.yml down

# Update
docker-compose -f docker-compose.mcp.yml pull
docker-compose -f docker-compose.mcp.yml up -d
```

## Port Forwarding (Windows Host Access)

```powershell
# Run in PowerShell as Administrator
# First, get WSL IP: wsl hostname -I
netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=<WSL-IP>

# Add firewall rule
New-NetFirewallRule -DisplayName "Engineer Assistant" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```

Access from anywhere: **http://<server-ip>:8501**

## Troubleshooting

### Services won't start
```bash
docker-compose -f docker-compose.mcp.yml logs
```

### Network issues
```bash
docker-compose -f docker-compose.mcp.yml down
docker-compose -f docker-compose.mcp.yml up -d
```

### WSL uses too much memory
Create `C:\Users\YourUsername\.wslconfig`:
```ini
[wsl2]
memory=8GB
processors=4
```

Then:
```powershell
wsl --shutdown
wsl
```

## Full Documentation

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete instructions and production setup.

## Support

- Issues: https://github.com/gioelemo/engineer-assistant/issues
- Docs: https://gioelemo.github.io/engineer-assistant/
