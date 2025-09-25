# AI Assistant for Mechanical Engineering Design

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
   conda activate python-ruff-template
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
```bash
python src/example.py  # Run example
```

## What's Included

- **Python 3.11.8** via conda-forge
- **Ruff** for fast linting/formatting
- **Pre-commit hooks** for automated quality checks
- **VS Code integration** with consistent settings
- **Example code** (intentionally messy to demonstrate ruff)

## Project Structure

```
├── .vscode/
│   ├── extensions.json          # Recommended VS Code extensions
│   └── settings_template.json   # VS Code settings template
├── src/                         # Your source code
├── tests/                       # Your tests
├── environment.yml              # Conda environment
├── pyproject.toml               # Project config & ruff settings
├── setup.sh / setup.bat         # One-command setup
└── README.md                    # This file
```

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

## Troubleshooting

- **Ruff not found in VS Code:** Restart VS Code after activating the conda environment
- **Pre-commit not working:** Run `pre-commit install` again
- **Environment issues:** Delete and recreate: `conda env remove -n python-ruff-template && conda env create -f environment.yml`