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

**Run all tests:**
```bash
pytest
```

**Run with coverage:**
```bash
pytest --cov=src --cov-report=term-missing
```

**Run specific test file:**
```bash
pytest tests/test_notebook_comparison.py
pytest tests/test_optimization_workflow.py
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

**Run the Multi-Agent Assistant:**
```bash
# Starts the supervisor system that coordinates all specialized agents
python -m src.main
```

The system includes:
- **Code Execution Agent**: Python code execution and calculations
- **Engineering Agent**: Structural optimization with EngiBench
- **Search Agent**: Web research and information gathering

The supervisor intelligently routes your requests to the appropriate agent!

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

### Code Execution Agent

The code execution agent allows you to run Python code and perform calculations interactively.

**Example usage:**
```bash
python test_code_execution_agent.py
```

**What you can do:**
- Execute arbitrary Python code snippets
- Perform mathematical calculations
- Test functions and code logic
- Run data analysis tasks
- Use NumPy, Pandas, and other Python libraries

**Example conversation with the agent:**
```
You: What is the sum of squares from 1 to 10?

Code Execution Agent: I'll calculate that for you!

[Executes: sum([i**2 for i in range(1, 11)])]

The sum of squares from 1 to 10 is 385.

You: Generate the first 10 Fibonacci numbers

Code Execution Agent: [Runs Fibonacci code]

The first 10 Fibonacci numbers are: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**Available tools:**
- `execute_python_code`: Run multi-line Python code
- `execute_python_expression`: Quickly evaluate a single expression

**Note:** Code runs in a persistent REPL environment, so variables persist between executions!

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
│   │   ├── code_execution_agent.py  # Python code execution
│   │   ├── engineering_agent.py     # Engineering optimization
│   │   ├── search_agent.py      # Web search
│   │   └── supervisor_agent.py  # Coordinates specialized agents
│   ├── cli/                     # Command-line interfaces
│   │   └── chat.py              # Interactive chat with supervisor
│   ├── models/                  # State definitions
│   │   └── state.py             # Conversation state
│   ├── tools/                   # Custom tools
│   │   ├── code_execution.py    # Python REPL execution
│   │   ├── engibench.py         # Engineering optimization tools
│   │   ├── search.py            # Web search
│   │   └── stl_export.py        # 3D model export
│   ├── utils/                   # Utilities
│   │   └── prompts.py           # System prompts
│   ├── main.py                  # Main entry point
│   ├── example.py               # Example usage
│   └── README.md                # Detailed architecture docs
├── scripts/                     # Utility scripts
│   ├── 2D_heatmap_to_stl.py     # Convert heatmaps to 3D STL files
│   └── generate_architecture_diagram.py  # Generate system diagrams
├── outputs/                     # Generated outputs
│   ├── agent_architecture.png   # System architecture diagram
│   ├── workflow_example.png     # Example workflow diagram
│   └── *.npy, *.png, *.stl      # Engineering design outputs
├── tests/                       # Unit and integration tests
│   ├── test_example.py
│   ├── test_optimization_workflow.py
│   └── test_notebook_comparison.py
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
