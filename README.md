<p align="center">
<img src="assets/logo_readme.png" align="center" width="50%"/>
</p>

[![Python tests](https://github.com/gioelemo/EngiAI/actions/workflows/test.yml/badge.svg)](https://github.com/gioelemo/EngiAI/actions/workflows/test.yml)
[![pre-commit](https://github.com/gioelemo/EngiAI/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/gioelemo/EngiAI/actions/workflows/pre-commit.yaml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EngiAI provides an AI-powered assistant for mechanical engineering design, integrating topology optimization, research retrieval, HPC simulation, and 3D printer control in a single conversational interface.

## 🎯 Quick Start

### Prerequisites

Before starting, ensure you have these external services set up:

1. **MMORE RAG Service** (Required) - Multimodal document retrieval
   ```bash
   # Contact the MMORE team or check internal documentation for repository access
   git clone <mmore-repository-url>
   cd mmore
   # Follow MMORE setup instructions to run at http://localhost:8000
   ```

2. **Prusa MCP Server** (Optional) - Only needed for 3D printer integration
   - Included as a git submodule at `services/prusa_mcp_server/prusa-mcp`
   - Automatically cloned with `--recurse-submodules` flag
   - See [Prusa MCP README](services/prusa_mcp_server/README.md) for configuration

3. **Excalidraw Whiteboard** - Interactive drawing canvas in the chat UI
   - Included as a git submodule at `src/ui/components/excalidraw`
   - Standalone repo: [gioelemo/streamlit-excalidraw](https://github.com/gioelemo/streamlit-excalidraw)
   - Automatically cloned with `--recurse-submodules` flag

### 🚀 Fastest Way: Docker (Recommended)

```bash
# 1. Clone and navigate to the repository (with submodules)
git clone --recurse-submodules https://github.com/gioelemo/EngiAI.git
cd EngiAI

# If you already cloned without --recurse-submodules:
# git submodule update --init --recursive

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys and MMORE_RAG_URL

# 3. Start the application
docker-compose up -d

# 4. (Optional) Start host service for GUI app integration
# In a separate terminal:
python host_service.py

# 5. Open your browser
# Visit: http://localhost:8501
```

That's it! The complete AI assistant is now running in Docker with all dependencies isolated.

**Note:** The host service (step 4) is optional and only needed if you want to open GUI applications like PrusaSlicer from within the chatbot.

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
cd EngiAI
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
cd EngiAI
```

1. **Create environment:**
   ```bash
   conda create -n engiai python=3.11 -y
   conda activate engiai
   pip install -e .[dev]
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
cd EngiAI
conda activate engiai
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

2. **Start the application:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - **Web UI**: http://localhost:8501
   - **Prusa MCP Server**: http://localhost:8765

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop services:**
   ```bash
   docker-compose down
   ```

6. **For GUI Application Integration (Optional):**

   If you want the Docker container to open GUI applications (like PrusaSlicer, etc.) on your host machine:

   ```bash
   # In a separate terminal, run the host service:
   python host_service.py
   ```

   This allows the containerized assistant to:
   - ✅ Open PrusaSlicer for 3D model slicing
   - ✅ Launch other GUI applications (VS Code, Terminal, etc.)
   - ✅ Execute commands on your host machine

   The host service runs on `http://localhost:9999` and provides a secure bridge between the Docker container and your local system.

   **Note:** This is only needed if you plan to use commands like "open PrusaSlicer" from within the chatbot.

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
- Conda environment activated: `conda activate engiai`

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
# Using Makefile (recommended)
make run-ui

# Or directly with streamlit
streamlit run src/ui/streamlit_app.py
```

**Features:**
- 💬 Clean chat interface
- 🎨 Interactive whiteboard with Excalidraw canvas
- 🔧 Real-time tool usage visualization
- 📥 Direct file downloads
- 🗑️ Conversation management
- 📊 Multi-chat support with database

**Access:** http://localhost:8501 (or port shown in terminal)

#### b) Standalone Prusa MCP Server
```bash
# Start external MCP server
./services/prusa_mcp_server/run.sh

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
GOOGLE_API_KEY=your-actual-google-api-key-here
```

**Database Configuration:**

The database configuration differs between Docker and local development:

- **Docker Deployment**: Uses PostgreSQL container (hostname: `postgres`)
  ```env
  DATABASE_URL=postgresql://engiai_user:engineer_ai_2025@postgres:5432/engineer_assistant
  ```

- **Local Development**: Use SQLite (recommended) or local PostgreSQL
  ```env
  # SQLite (recommended for local development)
  DATABASE_URL=sqlite:///data/conversations.db

  # OR local PostgreSQL (if you have it installed)
  # DATABASE_URL=postgresql://user:password@localhost:5432/engineer_assistant
  ```

**⚠️ Important:** When running locally with `streamlit run`, make sure your `.env` uses `sqlite:///` or `localhost` (NOT `postgres` hostname, which only exists in Docker).

**Optional Configuration:**
```env
# LLM Model Selection (defaults to gpt-4.1)
LLM_MODEL=openai:gpt-4.1

# Prusa MCP Integration (set to false to disable)
SKIP_MCP=true
PRUSA_MCP_URL=http://localhost:8765
```

**Available LLM Models:**
- `openai:gpt-4.1` (default, fast and cost-effective)
- `openai:gpt-4o` (most capable OpenAI model)
- `openai:gpt-4o-mini` (faster, cheaper option)

**LLM Tracing and Monitoring:**

The system supports two options for tracking LLM calls and performance:

1. **LangSmith Tracing** (LangChain's official tracing tool):
   ```env
   LANGCHAIN_TRACING=true
   LANGSMITH_API_KEY=your-langsmith-api-key
   LANGCHAIN_PROJECT=engiai
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```

2. **Weave Tracing** (Weights & Biases Weave for LLM benchmarking):
   ```env
   USE_WEAVE=true
   USE_WEAVE_CHATBOT=false
   WEAVE_PROJECT="your-wandb-entity/engiai-benchmarks"
   WANDB_API_KEY=your-wandb-api-key
   ```

**Note:** Weave uses a separate project from the `engiopt` W&B project to keep LLM benchmark traces isolated from model training experiments. The `USE_WEAVE` flag controls evaluation/benchmark tracing, while `USE_WEAVE_CHATBOT` separately controls chatbot interaction tracing (default: `false`).

Both tracing systems automatically capture:
- Input and output data from LLM calls
- Latency and token usage
- Model parameters and configurations
- Full execution traces

See [docs/source/weave_integration.md](docs/source/weave_integration.md) for detailed Weave usage and benchmarking examples.

---

### 🤖 Available Agents

The system includes specialized agents coordinated by a supervisor:

- **Engineering Agent**: Structural optimization with EngiBench
- **Search Agent**: Web research and information gathering
- **RAG Agent**: Document Q&A with MMORE multimodal RAG
- **ArXiv Agent**: Scientific paper search and analysis
- **Prusa Agent**: 3D printer management via Prusa Connect (optional)
- **HPC Agent**: HPC cluster job management via SSH

The supervisor intelligently routes your requests to the appropriate agent!

---

### 🛠️ Makefile Commands (Local Development)

For convenience, common commands are available via Makefile:

```bash
make help          # Show all available commands
make install       # Install/update conda environment
make run-ui        # Start Streamlit web interface
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

The engineering agent uses [EngiBench](https://engibench.ethz.ch), a library for engineering design benchmarking and optimization (automatically installed with the project dependencies).

**What you can do:**
- Optimize 2D beam structures for minimum compliance
- Simulate structural designs under various load conditions
- Run topology optimization with volume constraints
- Explore trade-offs between stiffness and material usage
- Perform multi-physics optimization balancing structural and thermal performance (ThermoElastic2D)
- Design optical devices for wavelength multiplexing (Photonics2D)
- Seamlessly work with any problem type - the system automatically adapts

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

### SSH Authentication Setup

The HPC integration supports two authentication methods:

#### Option 1: SSH Key Authentication (Recommended)

This is the most secure and convenient method for regular use:

#### 1. Generate SSH Key (if you don't have one)

**macOS/Linux:**
```bash
# Generate Ed25519 key (recommended)
ssh-keygen -t ed25519 -C "your_email@example.com"

# When prompted:
# - Save to: ~/.ssh/id_ed25519 (press Enter for default)
# - Enter passphrase (recommended for security)
# - Confirm passphrase
```

**Note:** You cannot add the private key itself to Keychain, but you can store the passphrase for the private key in Keychain (Step 3 below).

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

#### 3. Store Passphrase in Keychain (macOS)

**macOS 12.0 Monterey and later:**
```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

**macOS older than 12.0 Monterey:**
```bash
ssh-add -K ~/.ssh/id_ed25519
```

Enter your key passphrase when prompted - you won't be asked for it again!

**Note:** If this fails, make sure you're using Apple's version of `ssh-add`: `which ssh-add` should return `/usr/bin/ssh-add` (not a Homebrew version).

#### 4. Configure SSH to Always Use Keychain (macOS Sierra and later)

**Important:** macOS Sierra removed the convenient behavior of persisting keys between logins. You need to configure SSH to use the Keychain by default.

Create or edit `~/.ssh/config`:

```bash
# Default settings for all hosts
Host *
    UseKeychain yes
    AddKeysToAgent yes
    IdentityFile ~/.ssh/id_ed25519

# Your HPC Cluster
Host euler
    HostName euler.ethz.ch
    User your_username
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

**What each setting does:**
- `UseKeychain yes` - **Critical!** Tells SSH to look in macOS Keychain for the passphrase
- `AddKeysToAgent yes` - Automatically adds keys to SSH agent when first used
- `IdentityFile` - Specifies which SSH key to use
- `ForwardAgent yes` - Allows SSH agent forwarding for multi-hop connections

**Note:** If you have multiple private keys (e.g., `id_rsa`, `id_ed25519`), add an `IdentityFile` line for each one.

#### 5. Configure Shell to Auto-Load Keys (macOS)

Add this line to your `~/.zshrc` file to automatically load keys from Keychain on each login:

```bash
# Auto-load SSH keys from Keychain
ssh-add --apple-load-keychain -q 2>/dev/null
```

The `-q` flag suppresses the success message. After adding this, reload your shell:
```bash
source ~/.zshrc
```

**That's it!** Next time you load any SSH connection, it will:
- ✅ Try the private keys you've specified
- ✅ Look for their passphrase in the macOS Keychain
- ✅ Auto-load keys on each login (via ~/.zshrc)
- ✅ **No passphrase typing required!**

#### 6. Verify SSH Setup

```bash
# Check that your key is loaded
ssh-add -l
# Should show: 256 SHA256:... your_email@example.com (ED25519)

# Test HPC connection
ssh euler
# Should connect without asking for passphrase!
```

#### 7. Restart Docker (if using Docker deployment)

If you're running the application in Docker, restart the container to pick up the SSH agent:

```bash
docker-compose restart chatbot
```

Now your SSH connections will work seamlessly from both your terminal and the Docker container!

#### Option 2: Password Authentication (Alternative)

For environments where SSH key setup is not possible, you can use password authentication:

**Python API:**
```python
from src.tools.connection import HPCConnection

# Initialize connection with password
hpc = HPCConnection(
    host="euler.ethz.ch",      # Explicit hostname
    user="your_username",       # HPC username
    password="your_password",   # HPC password
    port=22                     # SSH port (default: 22)
)
```

**Important Notes:**
- Password authentication is less secure than SSH keys
- Passwords are not stored and must be provided each time
- Some HPC clusters may require SSH keys and not accept passwords
- Not recommended for production use
- Consider using SSH keys (Option 1) for better security

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

# Module/Environment Configuration
SLURM_PYTHON_MODULE=gcc/12.2.0
SLURM_CUDA_MODULE=cuda/12.2.2
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
- **🤖 Multi-Agent System**: Supervisor coordinates specialized agents (Engineering, Search, RAG, HPC, Prusa)
- **💬 Interactive UI**: Streamlit web interface with chat, Excalidraw whiteboard, file uploads, and visualization
- **🔧 Engineering Tools**: EngiBench integration for structural, multi-physics, and photonics topology optimization (beams2d, ThermoElastic2D, Photonics2D)
- **🔍 RAG System**: Document Q&A with MMORE multimodal RAG service
- **🖨️ 3D Printer Integration**: Prusa Connect integration via MCP (Model Context Protocol)
- **🖥️ HPC Integration**: SLURM job management for remote compute clusters
- **📊 Database Support**: PostgreSQL and SQLite for conversation persistence
- **📈 LLM Tracing**: Built-in support for LangSmith and Weave (W&B) for tracking LLM performance and benchmarking

### Development Tools
- **Python 3.11+** with conda-forge
- **Ruff** for fast linting and formatting
- **MyPy** for static type checking
- **Pre-commit hooks** for automated quality checks
- **VS Code integration** with consistent settings
- **Comprehensive test suite** with pytest
- **Environment variable management** with `.env` support

### Architecture
- **Modular design** with clear separation of agents, tools, and UI
- **LangGraph workflows** for agent orchestration
- **LangChain integration** for LLM interactions
- **Extensible tool system** for easy feature additions
- **Type-safe** with full type annotations

## Project Structure

```
├── .vscode/
│   ├── extensions.json          # Recommended VS Code extensions
│   └── settings_template.json   # VS Code settings template
├── services/                    # Standalone services
│   ├── host_service.py          # Host service for GUI integration
│   └── prusa_mcp_server/        # Prusa MCP server
│       ├── prusa-mcp/           # Git submodule (MCP implementation)
│       ├── server.py            # HTTP/SSE server wrapper
│       ├── client.py            # HTTP client for MCP
│       ├── Dockerfile           # MCP server Docker image
│       ├── run.sh               # Standalone server script
│       └── README.md            # MCP deployment guide
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
│   ├── models/                  # State definitions
│   │   └── state.py             # Conversation state
│   ├── tools/                   # Custom tools
│   │   ├── cli.py               # CLI tools
│   │   ├── connection.py        # HPC SSH connection (Fabric)
│   │   ├── engibench.py         # Engineering optimization
│   │   ├── engiopt.py           # EngiOpt integration
│   │   ├── hpc.py               # HPC monitoring
│   │   ├── mmore_client.py      # MMORE RAG service client
│   │   ├── rag_chain.py         # RAG pipeline
│   │   ├── search.py            # Web search
│   ├── ui/                      # Streamlit web interface
│   │   ├── streamlit_app.py     # Main Streamlit app
│   │   ├── chat.py              # Chat page
│   │   ├── home.py              # Home page
│   │   ├── settings.py          # Settings page
│   │   ├── chat_management.py   # Multi-chat DB management
│   │   └── components/
│   │       └── excalidraw/      # Git submodule (streamlit-excalidraw)
│   ├── utils/                   # Utilities
│   │   └── prompts.py           # System prompts
├── scripts/                     # Utility scripts
│   ├── 2D_heatmap_to_stl_extruded.py  # Convert heatmaps to 3D
│   ├── 2D_heatmap_to_stl.py     # Convert heatmaps to 3D STL
│   ├── import_local_papers.py   # Import PDFs to MMORE
│   ├── inspect_mmore.py         # Inspect MMORE documents
│   ├── generate_architecture_diagram.py  # Agent system diagram
│   └── generate_docker_mcp_diagram.py    # Docker MCP deployment diagram
├── data/                        # Data directory (gitignored)
│   └── conversations.db         # SQLite chat history
├── outputs/                     # Generated outputs
│   ├── *.slurm                  # Generated SLURM job scripts
│   └── *.npy, *.png, *.stl      # Engineering design outputs
├── tests/                       # Unit and integration tests
│   ├── test_agents/             # Agent tests
│   ├── test_tools/              # Tool tests
│   └── conftest.py              # Test configuration
├── docker-compose.yml           # Docker deployment with MCP
├── Dockerfile                   # Main application Docker image
├── .env.example                 # Environment variables template
├── config.py                    # Configuration management
├── pyproject.toml               # Project config & dependencies
├── Makefile                     # Convenient command shortcuts
├── requirements-mcp.txt         # MCP server dependencies
├── setup.sh / setup.bat         # One-command local setup
└── README.md                    # This file
```

See `src/README.md` for detailed architecture documentation and `services/prusa_mcp_server/README.md` for MCP deployment options.

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

Edit `pyproject.toml` under the `[project.dependencies]` section:
```toml
[project]
dependencies = [
    "langchain>=1.1.0",
    "your-new-package>=1.0.0",  # Add packages here
]
```

Then run:
```bash
pip install -e .
```

## API Keys Setup

To use the chatbot functionality, you'll need:

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Google API Key**: Get from [Google Platform](https://google.com)
3. **Tavily API Key**: Get from [Tavily](https://tavily.com/) for web search functionality

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
  conda env remove -n engiai
  conda create -n engiai python=3.11 -y
  conda activate engiai
  pip install -e .[dev]
  ```
- **Streamlit command not found:** Ensure conda environment is activated: `conda activate engiai`

### API & Configuration Issues
- **Missing API keys error:** Ensure `.env` file exists with valid keys (copy from `.env.example`)
- **Chatbot not responding:** Verify OpenAI API key and Google API key are valid and has sufficient credits
- **Web search not working:** Check Tavily API key in `.env` file
- **Database errors:** Check `DATABASE_URL` in `.env` or delete `data/conversations.db` to reset
- **"could not translate host name 'postgres'" error when running locally:**
  Your `.env` file is configured for Docker. Change `DATABASE_URL` to use SQLite:
  ```env
  DATABASE_URL=sqlite:///data/conversations.db
  ```
  The `postgres` hostname only works inside Docker containers. Use `localhost` or SQLite for local development.

### Prusa MCP Issues
- **MCP connection failed:** Ensure MCP server is running on port 8765
  ```bash
  # Check if server is running
  curl http://localhost:8765/sse
  ```
- **Tools not loading:** Set `SKIP_MCP=true` in `.env` to disable MCP integration
- **prusa-mcp folder not found:** Initialize the git submodule with `git submodule update --init --recursive`

### Host Service & GUI Integration Issues
- **"Cannot connect to host service" error:** Start the host service on your local machine:
  ```bash
  python host_service.py
  ```
  The service should show: `Starting Host Service on http://localhost:9999`

- **PrusaSlicer/GUI apps won't open:**
  1. Verify host service is running: `lsof -i :9999`
  2. Check Docker can reach host: `docker exec engiai-chatbot curl http://host.docker.internal:9999/health`
  3. Ensure the application is installed on your host machine

- **Port 9999 already in use:** Change the port in `.env`:
  ```bash
  HOST_SERVICE_PORT=9998  # Use a different port
  ```
  Then restart both the host service and Docker containers.

- **Security concerns:** The host service only accepts connections from localhost and has a whitelist of allowed applications. See `host_service.py` for the whitelist.

### SSH & HPC Connection Issues
- **"SSH key is encrypted" error in Docker:**
  ```bash
  # Add your key to SSH agent
  ssh-add --apple-use-keychain ~/.ssh/id_ed25519

  # Verify it's loaded
  ssh-add -l

  # Restart Docker container
  docker-compose restart chatbot
  ```
- **"Agent has no identities" error:**
  ```bash
  # Check if SSH agent is running
  echo $SSH_AUTH_SOCK
  # Should show a path like: /private/tmp/com.apple.launchd.*/Listeners

  # Add your key
  ssh-add --apple-use-keychain ~/.ssh/id_ed25519
  ```
- **Connection works locally but fails in Docker:**
  - Ensure SSH agent is running on host: `ssh-add -l`
  - Verify `SSH_AUTH_SOCK` environment variable is set
  - Check that `~/.ssh/config` has `AddKeysToAgent yes` and `UseKeychain yes`
  - Restart Docker after adding keys: `docker-compose restart chatbot`
- **SSH key permissions errors:**
  ```bash
  # Fix SSH key permissions
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/id_ed25519
  chmod 644 ~/.ssh/id_ed25519.pub
  chmod 600 ~/.ssh/config
  chmod 644 ~/.ssh/known_hosts
  ```

### Getting Help
- Check the [GitHub Issues](https://github.com/gioelemo/EngiAI/issues)
- Review logs in `data/*.log` files
- For Docker: `docker-compose logs -f`
- For local: Check terminal output for error messages
