# AI Assistant for Mechanical Engineering Design

[![Python tests](https://github.com/gioelemo/engineer-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/gioelemo/engineer-assistant/actions/workflows/test.yml)
[![pre-commit](https://github.com/gioelemo/engineer-assistant/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/gioelemo/engineer-assistant/actions/workflows/pre-commit.yaml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Jarvis-Style Multimodal AI Assistant for Closed-Loop Design-for-Manufacturing Correction.

## Quick Setup

### Prerequisites
- [Miniforge](https://github.com/conda-forge/miniforge) installed
- VS Code with Python extension

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

**Before running any applications, configure your environment variables:**

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

   Add your actual API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your-actual-openai-api-key-here
   TAVILY_API_KEY=your-actual-tavily-api-key-here

   # Optional: Choose your preferred LLM model (defaults to gpt-4.1)
   LLM_MODEL=openai:gpt-4.1
   ```

   **Available model options:**
   - `openai:gpt-4.1` (default, fast and cost-effective)
   - `openai:gpt-4o` (most capable OpenAI model)
   - `openai:gpt-3.5-turbo` (legacy, cheaper option)
   - `anthropic:claude-3-5-sonnet-20241022` (requires Anthropic API key)

   **CLI Tools configuration:**
   - `PRUSA_SLICER_PATH`: Path to PrusaSlicer executable (default: `prusa-slicer`)
     - **macOS:** `/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer`
     - **Windows:** `C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe`
     - **Linux:** `prusa-slicer` (if in PATH)

**Run the Multi-Agent Assistant:**

#### Option 1: Web UI (Recommended)
```bash
# Launch the Streamlit web interface
./run_ui.sh

# Or manually:
streamlit run src/ui/streamlit_app.py
```

The web UI provides:
- 💬 Clean chat interface
- 🔧 Real-time tool usage visualization
- 📥 Direct file downloads
- 🗑️ Conversation management

See [src/ui/README.md](src/ui/README.md) for more details.

#### Option 2: Command Line Interface
```bash
# Starts the supervisor system in CLI mode
python -m src.main
```

The system includes:
- **Engineering Agent**: Structural optimization with EngiBench
- **Search Agent**: Web research and information gathering
- **Prusa Agent**: 3D printer management via Prusa Connect
- **HPC Agent**: HPC cluster job management via SSH
- **CLI Agent**: Local command-line tool execution

The supervisor intelligently routes your requests to the appropriate agent!

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

- **Python 3.11.8** via conda-forge
- **Ruff** for fast linting/formatting
- **Pre-commit hooks** for automated quality checks
- **VS Code integration** with consistent settings
- **Modular architecture** with agents, tools, and CLI separation
- **Interactive AI chatbot** with arithmetic and web search
- **Engineering optimization** with EngiBench integration
- **Environment variable management** with `.env` support
- **Extensible tool system** for easy feature additions

## Project Structure

```
├── .vscode/
│   ├── extensions.json          # Recommended VS Code extensions
│   └── settings_template.json   # VS Code settings template
├── src/                         # Source code (modular structure)
│   ├── agents/                  # Agent implementations
│   │   ├── cli_agent.py         # CLI agent
│   │   ├── engineering_agent.py # Engineering optimization
│   │   ├── hpc_agent.py         # Agent to connect to SSH machine
│   │   ├── search_agent.py      # Web search
│   │   └── supervisor_agent.py  # Coordinates specialized agents
│   ├── cli/                     # Command-line interfaces
│   │   └── chat.py              # Interactive chat with supervisor
│   ├── models/                  # State definitions
│   │   └── state.py             # Conversation state
│   ├── tools/                   # Custom tools
│   │   ├── cli.py               # CLI tools
│   │   ├── connection.py        # Functions to connect to HPC cluster via ssh
│   │   ├── engibench.py         # Engineering optimization tools
│   │   ├── engiopt.py           # Engineering optimization tools
│   │   ├── hpc.py               # Functions to monitor the HPC cluster
│   │   ├── job_monitor.py       # Functions to monitor the HPC cluster
│   │   ├── search.py            # Web search
│   │   └── stl_export.py        # 3D model export
│   ├── ui/                      # User Interface
│   │   └── streamlit_app.py     # Streamlit App
│   ├── utils/                   # Utilities
│   │   └── prompts.py           # System prompts
│   ├── main.py                  # Main entry point
│   ├── example.py               # Example usage
│   └── README.md                # Detailed architecture docs
├── scripts/                     # Utility scripts
│   ├── 2D_heatmap_to_stl_extruded.py     # Convert heatmaps to 3D STL files
│   ├── 2D_heatmap_to_stl.py     # Convert heatmaps to 3D STL files
│   └── generate_architecture_diagram.py  # Generate system diagrams
├── outputs/                     # Generated outputs
│   ├── agent_architecture.png   # System architecture diagram
│   ├── workflow_example.png     # Example workflow diagram
│   ├── *.slurm                  # Generated SLURM job scripts
│   └── *.npy, *.png, *.stl      # Engineering design outputs
├── tests/                       # Unit and integration tests
│   ├── test_example.py
│   ├── test_optimization_workflow.py
│   └── test_notebook_comparison.py
├── .env.example                 # Environment variables template
├── config.py                    # Configuration management
├── connection.py                # HPC cluster SSH connection (Fabric-based)
├── environment.yml              # Conda environment
├── pyproject.toml               # Project config & ruff settings
├── setup.sh / setup.bat         # One-command setup
└── README.md                    # This file
```

See `src/README.md` for detailed architecture documentation and how to add new features.

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

- **Ruff not found in VS Code:** Restart VS Code after activating the conda environment
- **Pre-commit not working:** Run `pre-commit install` again
- **Environment issues:** Delete and recreate: `conda env remove -n engineer-assistant && conda env create -f environment.yml`
- **Missing API keys error:** Make sure you've created `.env` file and added your actual API keys
- **Chatbot not responding:** Verify your OpenAI API key is valid and has sufficient credits
- **Web search not working:** Check your Tavily API key in the `.env` file
