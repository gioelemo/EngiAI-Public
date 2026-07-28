# Installation

## 🚀 Option 1: Docker (Recommended)

Docker provides the easiest, most reliable installation with all dependencies isolated.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- Git installed on your system
- **MMORE RAG Service** running locally (see step 1 below)

### Installation Steps

#### 1. Set Up External Services

Before installing EngiAI, you need to set up the required external services:

**MMORE RAG Service** (Required):
```bash
# Clone and start MMORE service
# Contact the MMORE team or check internal documentation for repository access
git clone <mmore-repository-url>
cd mmore
# Follow MMORE setup and deployment instructions
# Ensure it's running at http://localhost:8000
```

**Prusa Connect MCP** (Optional - for 3D printer integration):

The Prusa MCP server is installed as a pip dependency from git
(`prusa-mcp @ git+https://github.com/gioelemo/prusa-mcp.git`),
declared in `services/prusa_mcp_server/requirements-mcp.txt`.
To run it standalone, use `./services/prusa_mcp_server/run.sh`,
or start it via `make docker-up` (which builds the
`prusa-mcp-server` container from `services/prusa_mcp_server/Dockerfile`).
See `services/prusa_mcp_server/README.md` for details.

#### 2. Clone the Repository

```bash
git clone https://github.com/gioelemo/EngiAI-Public.git
cd EngiAI-Public
```

#### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required configuration:
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Google API Key**: Get from [Google Platform](https://google.com)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/)
- **MMORE_RAG_URL**: URL to MMORE service (default: `http://localhost:8000`)

Optional (for Prusa 3D printer integration):
- **PRUSA_MCP_PATH**: Path to Prusa MCP server
- **SKIP_MCP**: Set to `false` to enable (default: `true`)

#### 4. Start with Docker Compose

```bash
docker-compose up -d
```

#### 5. Verify Installation

Access the UI at: **http://localhost:8501**

Check containers are running:
```bash
docker-compose ps
```

#### 6. View Logs

```bash
docker-compose logs -f chatbot
```

### Benefits of Docker

- ✅ Isolated environment with all dependencies
- ✅ Consistent behavior across machines
- ✅ Easy to scale and deploy
- ✅ Automatic restarts on failure
- ✅ Volume persistence for data
- ✅ No conda/Python environment conflicts

### Updating

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

---

## 💻 Option 2: Local Development (Conda)

For developers who want to modify the code or run without Docker.

### Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda installed
- Python 3.13 or higher
- Git installed on your system
- VS Code (recommended) with Python extension

### One-Command Setup

**First, navigate to the project directory:**
```bash
cd EngiAI-Public
```

**Then follow the manual installation steps below.** Automated setup scripts are not currently provided.

### Manual Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/gioelemo/EngiAI-Public.git
cd EngiAI-Public
```

#### Step 2: Create Conda Environment

```bash
conda create -n engiai python=3.13 -y
```

This will create an environment named `engiai` with Python 3.13.

#### Step 3: Activate Environment

```bash
conda activate engiai
```

#### Step 4: Install Dependencies

```bash
pip install -e .[dev]
```

This installs the package in editable mode along with all dependencies from `pyproject.toml`.

#### Step 5: Install Pre-commit Hooks

```bash
pre-commit install
```

#### Step 6: Configure VS Code (Optional)

1. Open the project in VS Code
2. Install recommended extensions when prompted:
   - `charliermarsh.ruff` (Ruff formatter)
   - `ms-python.mypy-type-checker` (MyPy type checker)
   - `ms-python.python` (Python support)

3. Copy template settings:
   - macOS/Linux: Copy `.vscode/settings_template.json` to `.vscode/settings.json`
   - Windows: Copy to `%APPDATA%\Code\User\settings.json`

### Verify Installation

Test your installation:

```bash
# Run tests
pytest

# Check code quality tools
ruff check .
mypy .
```

If all commands run without errors, you're ready to go! 🎉

### Setting Up API Keys

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys. See [Configuration](configuration.md) for details.

**Important for local development**: Use SQLite instead of PostgreSQL:
```bash
# In .env file
DATABASE_URL=sqlite:///data/conversations.db
```

---

## 🎯 Quick Comparison

| Feature | Docker | Local (Conda) |
|---------|---------|---------------|
| **Setup Time** | 5 minutes | 10-15 minutes |
| **Isolation** | ✅ Complete | ⚠️ Shared environment |
| **Portability** | ✅ Run anywhere | ❌ Needs conda setup |
| **Production Ready** | ✅ Yes | ❌ Development only |
| **Hot Reload** | ❌ Requires rebuild | ✅ Instant changes |
| **Resource Usage** | Moderate | Light |
| **Best For** | Production, demos | Active development |

---

## Troubleshooting

### Docker Issues

#### Port already in use
```bash
# Check what's using port 8501
lsof -i :8501

# Stop conflicting service or change port in docker-compose.yml
```

#### Container won't start
```bash
# Check logs
docker-compose logs -f chatbot

# Common causes:
# - Missing API keys in .env
# - Port conflicts
# - Insufficient resources
```

#### Permission errors (Linux)
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Conda Issues

#### Conda environment creation fails

If you get errors during environment creation:

1. Update conda: `conda update -n base conda`
2. Clear conda cache: `conda clean --all`
3. Try creating the environment again

#### Import errors

If you get import errors when running Python:

1. Make sure the environment is activated: `conda activate engiai`
2. Verify you're in the project root directory
3. Try reinstalling: `pip install -e .`

#### Pre-commit hooks fail

If pre-commit hooks are causing issues:

1. Update pre-commit: `pre-commit autoupdate`
2. Run manually: `pre-commit run --all-files`
3. If still failing, temporarily skip: `git commit --no-verify`

### Getting Help

If you encounter issues not covered here:

1. Check [Troubleshooting Guide](troubleshooting.md)
2. Check existing [GitHub Issues](https://github.com/gioelemo/EngiAI-Public/issues)
3. Open a new issue with:
   - Your operating system
   - Python version (`python --version`) or Docker version
   - Error messages
   - Steps to reproduce

## Next Steps

- [Quick Start Guide](quickstart.md) - Learn the basics
- [Docker Deployment](docker_deployment.md) - Production deployment
- [Configuration](configuration.md) - Set up API keys and preferences
- [Usage Guide](usage/agents.md) - Start using the agents
- [Troubleshooting](troubleshooting.md) - Common issues
