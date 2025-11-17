# Deployment Guide - Windows Server with WSL

This guide covers deploying Engineer Assistant on Windows Server using WSL (Windows Subsystem for Linux) with Docker.

## Prerequisites

### 1. Enable WSL on Windows Server

Open PowerShell as Administrator and run:

```powershell
# Enable WSL
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart the server
Restart-Computer
```

After restart, set WSL 2 as default:

```powershell
wsl --set-default-version 2
```

### 2. Install Ubuntu on WSL

```powershell
# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Launch Ubuntu (it will ask you to create a user)
wsl
```

### 3. Install Docker Desktop for Windows

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop
2. Install Docker Desktop
3. In Docker Desktop settings:
   - Enable "Use the WSL 2 based engine"
   - Under "Resources > WSL Integration", enable integration with your Ubuntu distribution

## Deployment Steps

### Step 1: Transfer Archives to Server

Transfer the release archives to your Windows Server:

```powershell
# Example using SCP (from your local machine)
scp engineer-assistant-v1.0.0.zip user@server-ip:C:\Temp\
scp prusa-mcp-server-v1.0.0.zip user@server-ip:C:\Temp\
```

Or use any file transfer method (RDP, file share, etc.)

### Step 2: Extract Archives in WSL

Open WSL Ubuntu terminal:

```bash
# Create deployment directory
mkdir -p ~/deployments
cd ~/deployments

# Copy archives from Windows to WSL
cp /mnt/c/Temp/engineer-assistant-v1.0.0.zip .
cp /mnt/c/Temp/prusa-mcp-server-v1.0.0.zip .

# Extract archives
unzip engineer-assistant-v1.0.0.zip -d engineer-assistant
unzip prusa-mcp-server-v1.0.0.zip -d .

# Navigate to the application directory
cd engineer-assistant
```

### Step 3: Configure Environment Variables

Create a `.env` file with your configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your settings
nano .env
```

Required environment variables:

```bash
# OpenAI API Key (required)
OPENAI_API_KEY=sk-...

# Tavily API Key (for web search)
TAVILY_API_KEY=tvly-...

# Prusa Connect Credentials (optional, can login via UI)
PRUSA_EMAIL=your-email@example.com
PRUSA_PASSWORD=your-password

# LLM Configuration
LLM_MODEL=openai:gpt-4o
EMBEDDINGS_MODEL=text-embedding-3-small

# Database Configuration
DATABASE_URL=postgresql://engiai_user:engineer_ai_2025@postgres:5432/engineer_assistant

# Papers Directory (optional, for bulk import)
PAPERS_SOURCE_DIR=/path/to/papers

# Prusa MCP Local Path
PRUSA_MCP_LOCAL_PATH=/home/your-user/deployments/prusa-mcp

# MCP Configuration
SKIP_MCP=false
HEADLESS=1
```

### Step 4: Update Prusa MCP Path

Update the `docker-compose.mcp.yml` to point to your extracted Prusa MCP directory:

```bash
# Edit docker-compose.mcp.yml
nano docker-compose.mcp.yml

# Update the PRUSA_MCP_LOCAL_PATH volume mount:
# From: ${PRUSA_MCP_LOCAL_PATH:-~/Desktop/prusa-mcp}:/app/prusa-mcp:ro
# To: ${PRUSA_MCP_LOCAL_PATH:-/home/your-user/deployments/prusa-mcp}:/app/prusa-mcp:ro
```

Or set it in your `.env` file:

```bash
PRUSA_MCP_LOCAL_PATH=/home/your-user/deployments/prusa-mcp
```

### Step 5: Build and Start the Application

```bash
# Build the Docker images
docker-compose -f docker-compose.mcp.yml build

# Start all services
docker-compose -f docker-compose.mcp.yml up -d

# Check that all services are running
docker-compose -f docker-compose.mcp.yml ps
```

Expected output:
```
NAME                          STATUS                   PORTS
engineer-assistant-chatbot    Up (healthy)             0.0.0.0:8501->8501/tcp
engineer-assistant-postgres   Up (healthy)             0.0.0.0:5432->5432/tcp
prusa-mcp-server              Up (healthy)             0.0.0.0:8765->8000/tcp
```

### Step 6: Access the Application

The application will be accessible at:
- **Streamlit UI**: http://localhost:8501
- From other machines: http://server-ip:8501

## Accessing from Windows Host

If you want to access the application from the Windows host (not just WSL):

1. Find your WSL IP address:
```bash
hostname -I
```

2. Access the application at:
```
http://<wsl-ip>:8501
```

Or configure port forwarding to make it accessible via Windows localhost:

```powershell
# In PowerShell (as Administrator)
netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=<wsl-ip>
```

## Firewall Configuration

If you need to access the application from external machines:

```powershell
# In PowerShell (as Administrator)
# Allow inbound traffic on port 8501
New-NetFirewallRule -DisplayName "Engineer Assistant" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```

## Managing the Application

### View Logs

```bash
# View all logs
docker-compose -f docker-compose.mcp.yml logs

# View specific service logs
docker-compose -f docker-compose.mcp.yml logs chatbot
docker-compose -f docker-compose.mcp.yml logs prusa-mcp-server

# Follow logs in real-time
docker-compose -f docker-compose.mcp.yml logs -f
```

### Stop the Application

```bash
docker-compose -f docker-compose.mcp.yml down
```

### Restart the Application

```bash
docker-compose -f docker-compose.mcp.yml restart
```

### Update the Application

```bash
# Stop the current version
docker-compose -f docker-compose.mcp.yml down

# Pull/extract new version
cd ~/deployments
unzip engineer-assistant-vX.X.X.zip -d engineer-assistant-new
cd engineer-assistant-new

# Copy your .env file from the old version
cp ../engineer-assistant/.env .

# Rebuild and start
docker-compose -f docker-compose.mcp.yml build
docker-compose -f docker-compose.mcp.yml up -d
```

## Persistence and Backups

### Data Persistence

The application stores data in:
- **Database**: PostgreSQL volume `postgres_data`
- **Conversations**: `./data/conversations.db` (if using SQLite)
- **Papers**: `./data/` directory
- **Outputs**: `./outputs/` directory

### Backup Data

```bash
# Backup database
docker-compose -f docker-compose.mcp.yml exec postgres pg_dump -U engiai_user engineer_assistant > backup.sql

# Backup data directory
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/

# Backup outputs
tar -czf outputs-backup-$(date +%Y%m%d).tar.gz outputs/
```

### Restore Data

```bash
# Restore database
docker-compose -f docker-compose.mcp.yml exec -T postgres psql -U engiai_user engineer_assistant < backup.sql

# Restore data directory
tar -xzf data-backup-20250117.tar.gz
```

## Troubleshooting

### Services Not Starting

Check logs:
```bash
docker-compose -f docker-compose.mcp.yml logs
```

### Network Issues

Ensure services are on the same network:
```bash
docker network ls
docker network inspect engineer-assistant_engineer-assistant
```

### Container Cannot Resolve Prusa MCP Server

Restart all services:
```bash
docker-compose -f docker-compose.mcp.yml down
docker-compose -f docker-compose.mcp.yml up -d
```

### WSL Memory Issues

Configure WSL memory limits in Windows:

Create `C:\Users\YourUsername\.wslconfig`:

```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

Restart WSL:
```powershell
wsl --shutdown
wsl
```

### Port Already in Use

Check what's using the port:
```bash
sudo lsof -i :8501
```

Or change the port in docker-compose.mcp.yml:
```yaml
ports:
  - "8502:8501"  # Use port 8502 instead
```

## Production Recommendations

### 1. Use a Reverse Proxy

For production, use Nginx as a reverse proxy:

```bash
sudo apt update
sudo apt install nginx

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/engineer-assistant
```

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/engineer-assistant /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 2. Enable HTTPS

Use Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 3. Set Up Auto-Start

Create a systemd service to auto-start on boot:

```bash
sudo nano /etc/systemd/system/engineer-assistant.service
```

```ini
[Unit]
Description=Engineer Assistant
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/your-user/deployments/engineer-assistant
ExecStart=/usr/bin/docker-compose -f docker-compose.mcp.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.mcp.yml down
User=your-user

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable engineer-assistant
sudo systemctl start engineer-assistant
```

### 4. Regular Backups

Set up a cron job for automated backups:

```bash
crontab -e

# Add this line to backup daily at 2 AM
0 2 * * * cd ~/deployments/engineer-assistant && docker-compose -f docker-compose.mcp.yml exec -T postgres pg_dump -U engiai_user engineer_assistant > ~/backups/db-backup-$(date +\%Y\%m\%d).sql
```

## Support

For issues and support:
- GitHub Issues: https://github.com/gioelemo/engineer-assistant/issues
- Documentation: https://gioelemo.github.io/engineer-assistant/

## Version

This deployment guide is for Engineer Assistant v1.0.0
