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

   # Optional: Choose your preferred LLM model (defaults to gpt-4o-mini)
   LLM_MODEL=openai:gpt-4o-mini
   ```

   **Available model options:**
   - `openai:gpt-4o-mini` (default, fast and cost-effective)
   - `openai:gpt-4o` (most capable OpenAI model)
   - `openai:gpt-3.5-turbo` (legacy, cheaper option)
   - `anthropic:claude-3-5-sonnet-20241022` (requires Anthropic API key)

**Run the chatbot:**
```bash
# General assistant (both math and search) - default
python -m src.main

# Math assistant (arithmetic only)
python -m src.main math

# Search assistant (web search only)
python -m src.main search

# Engineering assistant (structural optimization with EngiBench)
python -m src.main engineering

# Legacy version (for reference)
python -m src.chatbot
```

### Engineering Agent with EngiBench

The engineering agent uses [EngiBench](https://engibench.ethz.ch), a library for engineering design benchmarking and optimization.

**Install EngiBench:**
```bash
pip install engibench
```

**Example usage:**
```bash
python -m src.main engineering
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
│   │   ├── math_agent.py        # Math operations only
│   │   ├── search_agent.py      # Web search only
│   │   └── general_agent.py     # Both math and search
│   ├── cli/                     # Command-line interfaces
│   │   ├── chat.py              # Interactive chat (legacy)
│   │   └── chat_v2.py           # Interactive chat with agent selection
│   ├── models/                  # State definitions
│   │   └── state.py             # Conversation state
│   ├── tools/                   # Custom tools
│   │   ├── arithmetic.py        # Math operations
│   │   └── search.py            # Web search
│   ├── utils/                   # Utilities
│   │   └── prompts.py           # System prompts
│   ├── main.py                  # Main entry point
│   ├── chatbot.py               # Legacy chatbot
│   ├── chatbot_new.py           # Original version (pre-refactor)
│   └── README.md                # Detailed architecture docs
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
