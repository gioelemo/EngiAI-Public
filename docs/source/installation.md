# Installation

## Prerequisites

Before installing Engineer Assistant, make sure you have:

- [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda installed
- Python 3.11 or higher
- Git installed on your system
- VS Code (recommended) with Python extension

## Quick Installation

### One-Command Setup

The easiest way to get started is using our setup scripts:

#### macOS/Linux

```bash
cd engineer-assistant
./setup.sh
```

#### Windows

```bash
cd engineer-assistant
setup.bat
```

This will:
1. Create a conda environment
2. Install all dependencies
3. Set up pre-commit hooks
4. Configure VS Code settings

## Manual Installation

If the automated setup doesn't work for your system, follow these steps:

### Step 1: Clone the Repository

```bash
git clone https://github.com/gioelemo/engineer-assistant.git
cd engineer-assistant
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
```

This will create an environment named `engineer-assistant` with all required dependencies.

### Step 3: Activate Environment

```bash
conda activate engineer-assistant
```

### Step 4: Install Pre-commit Hooks

```bash
pre-commit install
```

### Step 5: Configure VS Code (Optional)

1. Open the project in VS Code
2. Install recommended extensions when prompted:
   - `charliermarsh.ruff` (Ruff formatter)
   - `ms-python.mypy-type-checker` (MyPy type checker)
   - `ms-python.python` (Python support)

3. Copy template settings:
   - macOS/Linux: Copy `.vscode/settings_template.json` to `.vscode/settings.json`
   - Windows: Copy to `%APPDATA%\Code\User\settings.json`

## Verify Installation

Test your installation:

```bash
# Run tests
pytest

# Check code quality tools
ruff check .
mypy .
```

If all commands run without errors, you're ready to go! 🎉

## Setting Up API Keys

The assistant requires API keys to function. See [Configuration](configuration.md) for details.

## Troubleshooting

### Common Issues

#### Conda environment creation fails

If you get errors during environment creation:

1. Update conda: `conda update -n base conda`
2. Clear conda cache: `conda clean --all`
3. Try creating the environment again

#### Import errors

If you get import errors when running Python:

1. Make sure the environment is activated: `conda activate engineer-assistant`
2. Verify you're in the project root directory
3. Try reinstalling: `pip install -e .`

#### Pre-commit hooks fail

If pre-commit hooks are causing issues:

1. Update pre-commit: `pre-commit autoupdate`
2. Run manually: `pre-commit run --all-files`
3. If still failing, temporarily skip: `git commit --no-verify`

### Getting Help

If you encounter issues not covered here:

1. Check existing [GitHub Issues](https://github.com/gioelemo/engineer-assistant/issues)
2. Open a new issue with:
   - Your operating system
   - Python version (`python --version`)
   - Error messages
   - Steps to reproduce

## Next Steps

- [Quick Start Guide](quickstart.md) - Learn the basics
- [Configuration](configuration.md) - Set up API keys and preferences
- [Usage Guide](usage/agents.md) - Start using the agents
