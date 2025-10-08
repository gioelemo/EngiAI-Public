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
   ```

**Run the chatbot:**
```bash
python -m src.chatbot  # Interactive AI chatbot with web search
```

## What's Included

- **Python 3.11.8** via conda-forge
- **Ruff** for fast linting/formatting
- **Pre-commit hooks** for automated quality checks
- **VS Code integration** with consistent settings
- **Interactive AI chatbot** with web search capabilities
- **Environment variable management** with `.env` support
- **Example code** (intentionally messy to demonstrate ruff)

## Project Structure

```
├── .vscode/
│   ├── extensions.json          # Recommended VS Code extensions
│   └── settings_template.json   # VS Code settings template
├── src/                         # Your source code
│   ├── chatbot.py               # Interactive AI chatbot
│   └── example.py               # Example code
├── scripts/                     # Utility scripts
│   └── 2D_heatmap_to_stl.py     # Convert heatmaps to 3D STL files
├── tests/                       # Your tests
├── .env.example                 # Environment variables template
├── config.py                    # Configuration management
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
