# Deployment Guide - Windows Server with WSL

This guide covers deploying EngiAI on Windows Server using WSL (Windows Subsystem for Linux) with Docker.

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
scp engiai-v1.0.0.zip user@server-ip:C:\Temp\
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
cp /mnt/c/Temp/engiai-v1.0.0.zip .
cp /mnt/c/Temp/prusa-mcp-server-v1.0.0.zip .

# Extract archives
unzip engiai-v1.0.0.zip -d engiai
unzip prusa-mcp-server-v1.0.0.zip -d .

# Navigate to the application directory
cd EngiAI
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
# Google API Key (required)
GOOGLE_API_KEY=sk-...
# Tavily API Key (for web search)
TAVILY_API_KEY=tvly-...

# LLM Configuration
LLM_MODEL=openai:gpt-4.1

# Database Configuration — ⚠ development default, change before deploying
DATABASE_URL=postgresql://engiai_user:engineer_ai_2025@postgres:5432/engineer_assistant

# Papers Directory (optional, for bulk import)
PAPERS_SOURCE_DIR=/path/to/papers



# MCP Configuration
SKIP_MCP=false
HEADLESS=1
```

### Step 4: Update Prusa MCP Path

Update the `docker-compose.yml` to point to your extracted Prusa MCP directory:

```bash
# Edit docker-compose.yml
nano docker-compose.yml
```

### Step 5: Build and Start the Application

```bash
# Build the Docker images
docker-compose build

# Start all services
docker-compose up -d

# Check that all services are running
docker-compose ps
```

Expected output:
```
NAME                          STATUS                   PORTS
engiai-chatbot    Up (healthy)             0.0.0.0:8501->8501/tcp
engiai-postgres   Up (healthy)             0.0.0.0:5432->5432/tcp
prusa-mcp-server              Up (healthy)             0.0.0.0:8765->8765/tcp
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
New-NetFirewallRule -DisplayName "EngiAI" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```

## Managing the Application

### View Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs chatbot
docker-compose logs prusa-mcp-server

# Follow logs in real-time
docker-compose logs -f
```

### Stop the Application

```bash
docker-compose down
```

### Restart the Application

```bash
docker-compose restart
```

### Update the Application

```bash
# Stop the current version
docker-compose down

# Pull/extract new version
cd ~/deployments
unzip engiai-vX.X.X.zip -d engiai-new
cd EngiAI-new

# Copy your .env file from the old version
cp ../engiai/.env .

# Rebuild and start
docker-compose build
docker-compose up -d
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
docker-compose exec postgres pg_dump -U engiai_user engineer_assistant > backup.sql

# Backup data directory
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/

# Backup outputs
tar -czf outputs-backup-$(date +%Y%m%d).tar.gz outputs/
```

### Restore Data

```bash
# Restore database
docker-compose exec -T postgres psql -U engiai_user engineer_assistant < backup.sql

# Restore data directory
tar -xzf data-backup-20250117.tar.gz
```

## Troubleshooting

### Services Not Starting

Check logs:
```bash
docker-compose logs
```

### Network Issues

Ensure services are on the same network:
```bash
docker network ls
docker network inspect engiai_engiai
```

### Container Cannot Resolve Prusa MCP Server

Restart all services:
```bash
docker-compose down
docker-compose up -d
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

Or change the port in docker-compose.yml:
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
sudo nano /etc/nginx/sites-available/engiai
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
sudo ln -s /etc/nginx/sites-available/engiai /etc/nginx/sites-enabled/
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
sudo nano /etc/systemd/system/engiai.service
```

```ini
[Unit]
Description=EngiAI
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/your-user/deployments/engiai
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
User=your-user

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable engiai
sudo systemctl start engiai
```

### 4. Regular Backups

Set up a cron job for automated backups:

```bash
crontab -e

# Add this line to backup daily at 2 AM
0 2 * * * cd ~/deployments/engiai && docker-compose exec -T postgres pg_dump -U engiai_user engineer_assistant > ~/backups/db-backup-$(date +\%Y\%m\%d).sql
```

## Support

For issues and support:
- GitHub Issues: https://github.com/gioelemo/EngiAI/issues
- Documentation: https://gioelemo.github.io/EngiAI/

## Version

This deployment guide is for EngiAI v1.0.0
