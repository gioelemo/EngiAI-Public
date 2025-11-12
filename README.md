# AI Assistant for Mechanical Engineering Design

[![Python tests](https://github.com/gioelemo/engineer-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/gioelemo/engineer-assistant/actions/workflows/test.yml)
[![pre-commit](https://github.com/gioelemo/engineer-assistant/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/gioelemo/engineer-assistant/actions/workflows/pre-commit.yaml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Jarvis-Style Multimodal AI Assistant for Closed-Loop Design-for-Manufacturing Correction.

## 🎯 Quick Start

### 🚀 Fastest Way: Docker (Recommended)

```bash
# 1. Clone and navigate to the repository
git clone https://github.com/gioelemo/engineer-assistant.git
cd engineer-assistant

# 2. Configure environment
cp .env.example .env
# Edit .env with your OpenAI and Tavily API keys

# 3. Start the application
docker-compose up -d

# 4. Open your browser
# Visit: http://localhost:8501
```

That's it! The complete AI assistant is now running in Docker with all dependencies isolated.

**To stop:** `docker-compose down`

---

## Local Development Setup (Conda)

For developers who want to modify the code or run without Docker:

### Prerequisites
- [Miniforge](https://github.com/conda-forge/miniforge) installed
- VS Code with Python extension (recommended)

### One-Command Setup

**First, navigate to the project directory:**
```bash
cd engineer-assistant
```

**Then run the setup script:**

**macOS/Linux:**
```bash
./setup.sh
```

**Windows:**
```bash
setup.bat
```

### Manual Setup (if scripts don't work)

**Make sure you're in the project directory first:**
```bash
cd engineer-assistant
```

1. **Create environment:**
   ```bash
   conda env create -f environment.yml
   conda activate engineer-assistant
   ```

2. **Install pre-commit:**
   ```bash
   pre-commit install
   ```

3. **Configure VS Code:**
   - Open this project in VS Code - it will prompt you to install recommended extensions
   - Install: `charliermarsh.ruff` (Ruff) and `ms-python.mypy-type-checker` (MyPy) extensions in VS Code
   - Copy settings of `.vscode/settings_template.json` at bottom of existing `.vscode/settings.json` (macOS/Linux) or to `%APPDATA%\Code\User\settings.json` (Windows)

## Usage

**Make sure you're in the project directory and environment is activated:**
```bash
cd engineer-assistant
conda activate engineer-assistant
```

### Code Quality
```bash
ruff check .          # Check for issues
ruff check --fix .    # Fix auto-fixable issues
ruff format .         # Format code
mypy .                # Type checking
```

### Testing

**Run all tests:**
```bash
pytest
```

**Run with coverage:**
```bash
pytest --cov=src --cov-report=term-missing
```

**Run only fast tests (skip slow integration tests):**
```bash
pytest -m "not slow"
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

### Running the Application

## 🚀 Deployment Options

### 📦 Option 1: Docker Compose (⭐ RECOMMENDED for Production)

Docker provides the most reliable, isolated, and portable deployment. All dependencies are containerized with consistent behavior across environments.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running

**Quick Start:**

1. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see configuration section below)
   ```

2. **Choose your deployment mode:**

   **a) Basic Deployment (No Prusa MCP):**
   ```bash
   docker-compose up -d
   ```

   **b) Full Deployment with Prusa 3D Printer Integration:**
   ```bash
   # Requires prusa-mcp folder at ~/Desktop/prusa-mcp
   docker-compose -f docker-compose.mcp.yml up -d
   ```

3. **Access the application:**
   - **Web UI**: http://localhost:8501
   - **Prusa MCP Server** (if enabled): http://localhost:8765

4. **View logs:**
   ```bash
   docker-compose logs -f chatbot          # Basic deployment
   docker-compose -f docker-compose.mcp.yml logs -f  # Full deployment
   ```

5. **Stop services:**
   ```bash
   docker-compose down
   # or
   docker-compose -f docker-compose.mcp.yml down
   ```

**Benefits:**
- ✅ Isolated environment with all dependencies
- ✅ Consistent behavior across machines
- ✅ Easy to scale and deploy
- ✅ Automatic restarts on failure
- ✅ Volume persistence for data
- ✅ No conda/Python environment conflicts

---

### 💻 Option 2: Local Development (Conda)

Best for active development and testing new features.

**Prerequisites:**
- Conda environment activated: `conda activate engineer-assistant`

**Configuration:**

1. **Create your environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your API keys:**
   ```bash
   # Open .env in your preferred editor
   code .env  # VS Code
   # or
   nano .env  # Terminal editor
   ```

**Running Modes:**

#### a) Streamlit Web UI
```bash
# Using convenience script
./run_ui.sh

# Or directly with streamlit
streamlit run src/ui/streamlit_app.py

# Or via Makefile
make run-ui
```

**Features:**
- 💬 Clean chat interface
- 🔧 Real-time tool usage visualization
- 📥 Direct file downloads
- 🗑️ Conversation management
- 📊 Multi-chat support with database

**Access:** http://localhost:8501 (or port shown in terminal)

#### b) Command Line Interface
```bash
# Start interactive CLI
python -m src.main

# Or via Makefile
make run-cli
```

**Features:**
- Terminal-based chat
- Direct command execution
- Tool confirmation prompts
- Lightweight and fast

#### c) Standalone Prusa MCP Server
```bash
# Start external MCP server
./prusa_mcp_server/run.sh

# Or via Makefile
make run-mcp
```

Then in another terminal, run the main app with MCP enabled:
```bash
export SKIP_MCP=false
export PRUSA_MCP_URL=http://localhost:8765
streamlit run src/ui/streamlit_app.py
```

---

### 🎯 Quick Comparison

| Feature | Docker Compose | Local Development |
|---------|----------------|-------------------|
| **Setup Time** | 5 minutes | 10-15 minutes |
| **Isolation** | ✅ Complete | ⚠️ Shared environment |
| **Portability** | ✅ Run anywhere | ❌ Needs conda setup |
| **Production Ready** | ✅ Yes | ❌ Development only |
| **Hot Reload** | ❌ Requires rebuild | ✅ Instant changes |
| **Resource Usage** | Moderate | Light |
| **Prusa MCP** | ✅ Integrated | ⚠️ Manual setup |
| **Best For** | Production, demos | Active development |

---

### ⚙️ Environment Configuration

**Required API Keys:**
```env
OPENAI_API_KEY=your-actual-openai-api-key-here
TAVILY_API_KEY=your-actual-tavily-api-key-here
```

**Optional Configuration:**
```env
# LLM Model Selection (defaults to gpt-4.1)
LLM_MODEL=openai:gpt-4.1

# Database (SQLite default, PostgreSQL recommended for production)
DATABASE_URL=sqlite:///data/conversations.db
# or for PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/engineer_assistant

# Prusa MCP Integration (set to false to disable)
SKIP_MCP=true
PRUSA_MCP_URL=http://localhost:8765

# CLI Tools
PRUSA_SLICER_PATH=prusa-slicer
```

**Available LLM Models:**
- `openai:gpt-4.1` (default, fast and cost-effective)
- `openai:gpt-4o` (most capable OpenAI model)
- `openai:gpt-3.5-turbo` (legacy, cheaper option)
- `anthropic:claude-3-5-sonnet-20241022` (requires Anthropic API key)

**PrusaSlicer Path by Platform:**
- **macOS:** `/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer`
- **Windows:** `C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe`
- **Linux:** `prusa-slicer` (if in PATH)

---

### 🤖 Available Agents

The system includes specialized agents coordinated by a supervisor:

- **Engineering Agent**: Structural optimization with EngiBench
- **Search Agent**: Web research and information gathering
- **RAG Agent**: Document Q&A with vector store
- **ArXiv Agent**: Scientific paper search and analysis
- **Prusa Agent**: 3D printer management via Prusa Connect (optional)
- **HPC Agent**: HPC cluster job management via SSH
- **CLI Agent**: Local command-line tool execution

The supervisor intelligently routes your requests to the appropriate agent!

---

### 🛠️ Makefile Commands (Local Development)

For convenience, common commands are available via Makefile:

```bash
make help          # Show all available commands
make install       # Install/update conda environment
make run-ui        # Start Streamlit web interface
make run-cli       # Start CLI chat interface
make run-mcp       # Start standalone Prusa MCP server
make test          # Run all tests
make test-fast     # Run only fast tests
make test-cov      # Run tests with coverage report
make lint          # Check code quality (ruff + mypy)
make format        # Format code with ruff
make clean         # Clean cache and build files
make docs          # Build documentation
make docs-watch    # Serve docs with live reload
```

---

### Engineering Agent with EngiBench

The engineering agent uses [EngiBench](https://engibench.ethz.ch), a library for engineering design benchmarking and optimization.

**Install EngiBench:**
```bash
pip install engibench
```

**What you can do:**
- Optimize 2D beam structures for minimum compliance
- Simulate structural designs under various load conditions
- Run topology optimization with volume constraints
- Explore trade-offs between stiffness and material usage

**Example conversation:**
```
You: I need to design a beam that minimizes weight while maximizing stiffness.
     Can you help me optimize it with a 35% volume fraction?

Engineering Assistant: I'll help you optimize a beam design! Let me set up
the problem and run topology optimization...

[Uses EngiBench tools to optimize design]

The optimized design achieved a compliance of 0.0234, which is 67% better
than the initial random design! This means the structure is significantly
stiffer while using only 35% of the available material.
```


## HPC Cluster Integration with Fabric

This project includes integration with HPC clusters (like ETH Zurich's Euler cluster) for submitting and monitoring large-scale training jobs.

### SSH Key Setup

Before using HPC integration, set up SSH key authentication:

#### 1. Generate SSH Key (if you don't have one)

**macOS/Linux:**
```bash
# Generate Ed25519 key (recommended)
ssh-keygen -t ed25519 -C "your_email@example.com"

# When prompted:
# - Save to: ~/.ssh/id_ed25519 (press Enter for default)
# - Enter passphrase (optional but recommended)
# - Confirm passphrase

# Add key to SSH agent
ssh-add ~/.ssh/id_ed25519

# (macOS only) Add key to keychain
ssh-add --apple-load-keychain ~/.ssh/id_ed25519
```

#### 2. Copy Public Key to HPC Cluster

```bash
# Copy your public key to HPC
ssh-copy-id -i ~/.ssh/id_ed25519.pub username@hpc.example.com

# Or manually:
ssh username@hpc.example.com << 'EOF'
mkdir -p ~/.ssh
cat >> ~/.ssh/authorized_keys << 'PUBKEY'
$(cat ~/.ssh/id_ed25519.pub)
PUBKEY
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
EOF
```

#### 3. Configure SSH Config

Edit `~/.ssh/config` to add your HPC host (or append to existing config):

```bash
Host euler
    HostName euler.ethz.ch
    User your_username
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    UseKeychain yes
    AddKeysToAgent yes
```

Then test the connection:
```bash
ssh euler  # Should connect without password
```

### Using the HPC Connection Tool

#### Python API

```python
from src.tools.connection import HPCConnection

# Initialize connection (reads ~/.ssh/config)
hpc = HPCConnection(host_alias="euler")

# Generate a SLURM script (from generate_training_command tool)
from src.tools.engiopt import generate_training_command

result = generate_training_command(
    algorithm="cgan_cnn_2d",
    epochs=100,
    seed=42,
    wandb_entity="your_wandb_entity",
    problem_id="beams2d"
)
# This saves a SLURM script to outputs/

# Submit the job
job_id = hpc.submit_job(slurm_file=result["slurm_file"])
print(f"Job {job_id} submitted!")

# Check job status
status = hpc.get_job_status(job_id)
print(status)

# Download results (after job completes)
hpc.get_job_output(job_id, remote_dir="~/slurm_jobs", local_dir="outputs")

# Cancel job if needed
hpc.cancel_job(job_id)
```

#### Command Line Interface

```bash
# Test SSH connection
python connection.py test

# Output:
# ✅ SSH connection successful!
#    Remote directory: /home/username

# Submit a SLURM job
python connection.py submit outputs/train_cgan_cnn_2d_beams2d_seed1.slurm

# Check job status
python connection.py status 12345

# Cancel a job
python connection.py cancel 12345

# Download job outputs
python connection.py download 12345
```

#### Complete Workflow Example

```bash
# 0. Test connection (verify SSH is working)
python connection.py test

# Output:
# ✅ SSH connection successful!
#    Remote directory: /home/username

# 1. Generate SLURM training script
streamlit run src/ui/streamlit_app.py
# → Use "Generate Training Command" in the UI with parameters
# → Script saved to outputs/train_*.slurm

# 2. Submit to HPC
python connection.py submit outputs/train_cgan_cnn_2d_beams2d_seed1.slurm

# Output: ✅ Job submitted with ID: 45678

# 3. Monitor progress
python connection.py status 45678

# Output:
#    JOBID PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
#    45678      gpu.4d  train_ca  username  RUNNING   0:15:32    6:00:00      1 node-name

# 4. Once complete, download outputs
python connection.py download 45678

# Output:
# ✅ Downloaded engiopt_cgan_cnn_2d_45678.out to outputs/
# ✅ Downloaded engiopt_cgan_cnn_2d_45678.err to outputs/
```

### Configuration

The HPC connection uses environment variables for SLURM configuration. Edit `.env`:

```env
# SLURM Job Configuration
SLURM_TIME=6:00:00              # Max job duration (HH:MM:SS)
SLURM_NTASKS=1                 # Number of tasks
SLURM_CPUS_PER_TASK=8          # CPUs per task
SLURM_MEM_PER_CPU=4G           # Memory per CPU
SLURM_GPUS=rtx4090:1           # GPU specification
SLURM_PARTITION=gpu.4d         # Cluster partition
SLURM_EMAIL_USER=your@email.com # Email for job notifications

# Module/Environment Configuration
SLURM_PYTHON_MODULE=gcc/12.2.0
SLURM_CUDA_MODULE=cuda/12.2.2
SLURM_VENV_PATH=/cluster/scratch/$USER/venv
SLURM_PROJECT_PATH=/cluster/scratch/$USER/engineer-assistant
```

### Features

- **Automatic SSH config parsing**: Reads from `~/.ssh/config`
- **Key-based authentication**: Secure, no passwords
- **SSH agent integration**: Works with macOS keychain
- **SFTP file transfer**: Reliable file uploads/downloads
- **Job monitoring**: Track SLURM job status
- **Output collection**: Automatically download logs and results
- **Pattern matching**: Download files matching job ID patterns

### Troubleshooting

**Connection refused:**
```bash
# Test SSH connection
ssh -v euler

# Verify SSH config
cat ~/.ssh/config

# Check key permissions
ls -la ~/.ssh/id_ed25519
# Should be: -rw------- (600)
```

**Key permission denied:**
```bash
# Fix SSH key permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/authorized_keys
```

**Job not found:**
```bash
# List all your jobs
ssh euler squeue -u your_username

# Check specific job
ssh euler squeue -j 12345
```

**Transfer issues:**
```bash
# Verify remote directory exists
ssh euler ls -la ~/slurm_jobs

# Manual file transfer test
python connection.py submit outputs/test.slurm
```

## What's Included

### Core Features
- **🐳 Docker Deployment**: Production-ready containerized deployment with Docker Compose
- **🤖 Multi-Agent System**: Supervisor coordinates specialized agents (Engineering, Search, RAG, HPC, Prusa, CLI)
- **💬 Interactive UI**: Streamlit web interface with chat, file uploads, and visualization
- **🔧 Engineering Tools**: EngiBench integration for structural optimization and topology design
- **🔍 RAG System**: Document Q&A with vector store (ChromaDB) and multimodal support
- **🖨️ 3D Printer Integration**: Prusa Connect integration via MCP (Model Context Protocol)
- **🖥️ HPC Integration**: SLURM job management for remote compute clusters
- **📊 Database Support**: PostgreSQL and SQLite for conversation persistence

### Development Tools
- **Python 3.11+** with conda-forge
- **Ruff** for fast linting and formatting
- **MyPy** for static type checking
- **Pre-commit hooks** for automated quality checks
- **VS Code integration** with consistent settings
- **Comprehensive test suite** with pytest
- **Environment variable management** with `.env` support

### Architecture
- **Modular design** with clear separation of agents, tools, and CLI
- **LangGraph workflows** for agent orchestration
- **LangChain integration** for LLM interactions
- **Extensible tool system** for easy feature additions
- **Type-safe** with full type annotations

## Project Structure

```
├── .vscode/
│   ├── extensions.json          # Recommended VS Code extensions
│   └── settings_template.json   # VS Code settings template
├── prusa_mcp_server/            # Standalone Prusa MCP server
│   ├── __init__.py
│   ├── server.py                # HTTP/SSE server wrapper
│   ├── client.py                # HTTP client for MCP
│   ├── Dockerfile               # MCP server Docker image
│   ├── run.sh                   # Standalone server script
│   └── README.md                # MCP deployment guide
├── src/                         # Source code (modular structure)
│   ├── agents/                  # Agent implementations
│   │   ├── arxiv_agent.py       # Scientific paper search
│   │   ├── cli_agent.py         # CLI command execution
│   │   ├── engineering_agent.py # Engineering optimization
│   │   ├── hpc_agent.py         # HPC cluster management
│   │   ├── prusa_agent.py       # 3D printer control
│   │   ├── rag_agent.py         # Document Q&A
│   │   ├── search_agent.py      # Web search
│   │   └── supervisor_agent.py  # Coordinates specialized agents
│   ├── cli/                     # Command-line interfaces
│   │   └── chat.py              # Interactive CLI chat
│   ├── models/                  # State definitions
│   │   └── state.py             # Conversation state
│   ├── tools/                   # Custom tools
│   │   ├── cli.py               # CLI tools
│   │   ├── connection.py        # HPC SSH connection (Fabric)
│   │   ├── engibench.py         # Engineering optimization
│   │   ├── engiopt.py           # EngiOpt integration
│   │   ├── hpc.py               # HPC monitoring
│   │   ├── rag_chain.py         # RAG pipeline
│   │   ├── search.py            # Web search
│   │   ├── stl_export.py        # 3D model export
│   │   └── vector_store.py      # ChromaDB integration
│   ├── ui/                      # Streamlit web interface
│   │   ├── streamlit_app.py     # Main Streamlit app
│   │   ├── chat.py              # Chat page
│   │   ├── home.py              # Home page
│   │   ├── settings.py          # Settings page
│   │   └── chat_management.py   # Multi-chat DB management
│   ├── utils/                   # Utilities
│   │   └── prompts.py           # System prompts
│   └── main.py                  # CLI entry point
├── scripts/                     # Utility scripts
│   ├── 2D_heatmap_to_stl_extruded.py  # Convert heatmaps to 3D
│   ├── 2D_heatmap_to_stl.py     # Convert heatmaps to 3D STL
│   ├── import_local_papers.py   # Import PDFs to vector store
│   ├── generate_architecture_diagram.py  # Agent system diagram
│   └── generate_docker_mcp_diagram.py    # Docker MCP deployment diagram
├── data/                        # Data directory (gitignored)
│   ├── chroma_db/               # Vector store database
│   └── conversations.db         # SQLite chat history
├── outputs/                     # Generated outputs
│   ├── *.slurm                  # Generated SLURM job scripts
│   └── *.npy, *.png, *.stl      # Engineering design outputs
├── tests/                       # Unit and integration tests
│   ├── test_agents/             # Agent tests
│   ├── test_tools/              # Tool tests
│   └── conftest.py              # Test configuration
├── docker-compose.yml           # Basic Docker deployment
├── docker-compose.mcp.yml       # Full deployment with MCP
├── Dockerfile                   # Main application Docker image
├── .env.example                 # Environment variables template
├── config.py                    # Configuration management
├── environment.yml              # Conda environment
├── pyproject.toml               # Project config & dependencies
├── Makefile                     # Convenient command shortcuts
├── requirements-mcp.txt         # MCP server dependencies
├── setup.sh / setup.bat         # One-command local setup
└── README.md                    # This file
```

See `src/README.md` for detailed architecture documentation and `prusa_mcp_server/README.md` for MCP deployment options.

### 📊 Architecture Diagrams

Generate visual diagrams of the system architecture:

```bash
# Generate agent system architecture diagram
python scripts/generate_architecture_diagram.py
# Output: outputs/agent_architecture.png

# Generate Docker MCP deployment diagram
python scripts/generate_docker_mcp_diagram.py
# Output: outputs/docker_mcp_architecture.png
```

The diagrams show:
- **Agent Architecture**: Multi-agent system with supervisor and specialized agents
- **Docker MCP**: Container deployment with HTTP/SSE communication between services

## Adding Dependencies

Edit `environment.yml`:
```yaml
dependencies:
  - python=3.11.8
  - numpy  # Add conda packages here
  - pip
  - pip:
    - ruff>=0.1.0
    - pre-commit>=3.0.0
    - mypy>=1.17.1  # Add pip packages after here
```

Then run:
```bash
conda env update -f environment.yml
```

## API Keys Setup

To use the chatbot functionality, you'll need:

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Tavily API Key**: Get from [Tavily](https://tavily.com/) for web search functionality

Both are required for the chatbot to work properly.

## Troubleshooting

### Docker Issues
- **Port already in use:** Stop conflicting services or change ports in `docker-compose.yml`
  ```bash
  # Check what's using port 8501
  lsof -i :8501
  # Kill the process or change the port mapping
  ```
- **Container won't start:** Check logs with `docker-compose logs -f chatbot`
- **Can't access UI:** Ensure Docker Desktop is running and containers are up: `docker-compose ps`
- **Permission errors (Linux):** Add your user to docker group: `sudo usermod -aG docker $USER`
- **Out of disk space:** Clean up Docker: `docker system prune -a`

### Local Development Issues
- **Ruff not found in VS Code:** Restart VS Code after activating the conda environment
- **Pre-commit not working:** Run `pre-commit install` again
- **Environment issues:** Delete and recreate:
  ```bash
  conda env remove -n engineer-assistant
  conda env create -f environment.yml
  ```
- **Streamlit command not found:** Ensure conda environment is activated: `conda activate engineer-assistant`

### API & Configuration Issues
- **Missing API keys error:** Ensure `.env` file exists with valid keys (copy from `.env.example`)
- **Chatbot not responding:** Verify OpenAI API key is valid and has sufficient credits
- **Web search not working:** Check Tavily API key in `.env` file
- **Database errors:** Check `DATABASE_URL` in `.env` or delete `data/conversations.db` to reset

### Prusa MCP Issues
- **MCP connection failed:** Ensure MCP server is running on port 8765
  ```bash
  # Check if server is running
  curl http://localhost:8765/sse
  ```
- **Tools not loading:** Set `SKIP_MCP=true` in `.env` to disable MCP integration
- **prusa-mcp folder not found:** Ensure `~/Desktop/prusa-mcp` exists or update `PRUSA_MCP_PATH`

### Getting Help
- Check the [GitHub Issues](https://github.com/gioelemo/engineer-assistant/issues)
- Review logs in `data/*.log` files
- For Docker: `docker-compose logs -f`
- For local: Check terminal output for error messages
