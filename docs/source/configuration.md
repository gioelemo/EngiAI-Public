# Configuration

Learn how to configure Engineer Assistant for your environment.

## Environment Variables

The assistant uses environment variables for configuration. These are stored in a `.env` file in the project root.

### Creating Your Configuration

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   nano .env  # or use your preferred editor
   ```

### Required Variables

These must be set for the assistant to function:

```bash
# OpenAI API Key (Required for all LLM operations)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Tavily API Key (Required for web search)
TAVILY_API_KEY=tvly-your-tavily-api-key-here

# MMORE RAG Service (Required for document retrieval)
MMORE_RAG_URL=http://localhost:8000
```

**Where to get keys:**
- **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Tavily**: [https://tavily.com/](https://tavily.com/)

**External Services:**
- **MMORE**: Clone and deploy the MMORE service locally before starting Engineer Assistant
  ```bash
  # Contact the MMORE team or check internal documentation for repository access
  git clone <mmore-repository-url>
  cd mmore
  # Follow MMORE setup and deployment instructions
  ```

### Optional Variables

These enhance functionality but aren't required:

```bash
# LLM Configuration
LLM_MODEL=openai:gpt-4o  # Default model
LLM_TEMPERATURE=0.0      # Temperature for responses (0.0-2.0)

# HPC Cluster Configuration
HPC_HOST_ALIAS=euler     # SSH alias for your HPC cluster
HPC_USERNAME=username    # Your cluster username
HPC_WORKSPACE=/cluster/scratch/username  # Workspace directory

# Weights & Biases (for ML tracking)
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=engineer-assistant
WANDB_ENTITY=your-team

# Prusa 3D Printer Integration (optional)
SKIP_MCP=true                    # Set to false to enable Prusa integration
PRUSA_MCP_PATH=/path/to/prusa-mcp  # Path to cloned Prusa MCP server
PRUSA_SLICER_PATH=/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer

# Database (optional, uses SQLite by default)
DATABASE_URL=sqlite:///data/engineer_assistant.db
```

**For Prusa 3D printer integration:**
- Use the Prusa MCP server from this repository:
  ```bash
  # Available in prusa_mcp_server/ directory
  # See prusa_mcp_server/README.md for setup
  ```
- Set `SKIP_MCP=false` and configure `PRUSA_MCP_PATH`

### Document Processing Variables

For PDF and document processing features:

```bash


# Paper Import Configuration
PAPERS_SOURCE_DIR=/path/to/papers  # Directory containing PDFs to import
PAPERS_STATE_FILE=data/local_import_state.json  # Tracks imported papers
```

**Paper Import:**
- `PAPERS_SOURCE_DIR`: Point to your local papers directory (or network share)
- Papers are automatically uploaded to MMORE service
- Progress tracked in `PAPERS_STATE_FILE`
- See [Paper Import Guide](paper_import_guide.md) for details

### Observability & Debugging

For monitoring and debugging:

```bash
# LangSmith Tracing (optional)
LANGSMITH_TRACING=false  # Set to true to enable
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=engineer-assistant

# Host Service (Docker GUI integration)
HOST_SERVICE_PORT=9999  # Port for opening GUI apps from Docker
```

**LangSmith Setup:**
1. Create account at [smith.langchain.com](https://smith.langchain.com/)
2. Get API key from settings
3. Monitor LLM calls, costs, and performance
4. Debug conversation flows and tool usage

### Hugging Face Integration

For ML model downloads:

```bash
# Hugging Face
HF_TOKEN=your-huggingface-token
HF_HOME=$HOME/.cache/huggingface/hub
HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
```

**Getting HF Token:**
1. Create account at [huggingface.co](https://huggingface.co/)
2. Go to Settings → Access Tokens
3. Create token with "Read" permission

### Advanced SLURM Configuration

For HPC job submission (advanced users):

```bash
# SLURM Job Defaults
SLURM_TIME=00:45:00           # Max job duration
SLURM_NTASKS=1                # Number of tasks
SLURM_CPUS_PER_TASK=4         # CPUs per task
SLURM_MEM_PER_CPU=7GB         # Memory per CPU
SLURM_GPUS=rtx_4090:1         # GPU specification

# Module Loading
SLURM_STACK_MODULE=stack/2024-06
SLURM_GCC_MODULE=gcc/12.2.0
SLURM_PYTHON_MODULE=python_cuda/3.11.6
SLURM_CUDA_MODULE=cuda/12.8.0

# Email Notifications
SLURM_EMAIL_USER=your@email.com
```

**Note:** Many SLURM settings can be configured via the Settings UI (⚙️ Settings > SLURM Cluster Configuration). Environment variables serve as fallback defaults.

## LLM Models

You can configure which language model to use:

### Supported Models

```bash
# OpenAI Models
LLM_MODEL=openai:gpt-4o          # GPT-4 Optimized (recommended)
LLM_MODEL=openai:gpt-4-turbo     # GPT-4 Turbo
LLM_MODEL=openai:gpt-3.5-turbo   # GPT-3.5 (faster, cheaper)

# Anthropic Models (if you have an API key)
LLM_MODEL=anthropic:claude-3-opus
LLM_MODEL=anthropic:claude-3-sonnet
```

### Model Parameters

```bash
# Temperature: Controls randomness in responses (0.0 to 2.0)
LLM_TEMPERATURE=0.0   # Deterministic, focused (recommended for engineering)
LLM_TEMPERATURE=0.7   # Balanced creativity (default)
LLM_TEMPERATURE=1.0   # More creative and varied responses

# Note: Higher temperature increases creativity but may reduce accuracy
# For engineering tasks, use 0.0-0.3 for precision
# For brainstorming, use 0.7-1.0 for variety
```

## HPC Configuration

For cluster computing features:

### SSH Setup

1. Configure SSH access to your cluster:
   ```bash
   # In ~/.ssh/config
   Host euler
       HostName euler.ethz.ch
       User your-username
       IdentityFile ~/.ssh/id_rsa
   ```

2. Test connection:
   ```bash
   ssh euler
   ```

3. Set in `.env`:
   ```bash
   HPC_HOST_ALIAS=euler
   HPC_WORKSPACE=/cluster/scratch/your-username
   ```

### Fabric Configuration

The assistant uses [Fabric](http://www.fabfile.org/) for HPC interactions. Configuration is automatic based on your `.env` settings.

## Weights & Biases

For experiment tracking:

1. Create account at [wandb.ai](https://wandb.ai/)
2. Get API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Add to `.env`:
   ```bash
   WANDB_API_KEY=your-key-here
   WANDB_PROJECT=engineer-assistant
   WANDB_ENTITY=your-username-or-team
   ```

## Database Setup

See the [Database Setup Guide](database_setup.md) for detailed instructions on:

- Setting up PostgreSQL or SQLite for conversation history
- Configuring the MMORE RAG service for document retrieval

## VS Code Settings

Recommended settings are in `.vscode/settings_template.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "mypy-type-checker.args": ["--config-file", "pyproject.toml"]
}
```

## Validating Configuration

Check your configuration:

```bash
# Test API keys
python -c "import config; print('OpenAI Key:', config.config.openai_api_key[:10])"

# Test database connection
python -c "from src.ui.database import DatabaseManager; db = DatabaseManager(); print('DB OK')"

# Run all tests
pytest tests/test_config.py
```

## Troubleshooting

### API Key Issues

**Error**: `openai.AuthenticationError`
- Check your API key in `.env`
- Ensure no extra spaces or quotes
- Verify key is active in OpenAI dashboard

### HPC Connection Issues

**Error**: `Connection refused`
- Verify SSH config
- Test manual SSH connection
- Check firewall/VPN requirements

### Database Issues

**Error**: `database is locked`
- Close other applications using the database
- Use PostgreSQL for concurrent access

## Security Best Practices

1. **Never commit `.env`** - It's in `.gitignore` for a reason
2. **Use environment-specific keys** - Different keys for dev/prod
3. **Rotate keys regularly** - Change API keys periodically
4. **Limit key permissions** - Use minimal required scopes
5. **Use secrets manager** - For production deployments (AWS Secrets Manager, etc.)

## Next Steps

- [Quick Start](quickstart.md) - Start using the assistant
- [Usage Guide](usage/agents.md) - Learn about agents
- [API Reference](api/agents.rst) - Developer docs
