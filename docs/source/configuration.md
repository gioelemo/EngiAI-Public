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
```

**Where to get keys:**
- **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Tavily**: [https://tavily.com/](https://tavily.com/)

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

# Prusa Slicer (for 3D printing)
PRUSA_SLICER_PATH=/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer

# Database (optional, uses SQLite by default)
DATABASE_URL=sqlite:///data/engineer_assistant.db
```

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
LLM_TEMPERATURE=0.0   # Deterministic (recommended for engineering)
LLM_TEMPERATURE=0.7   # More creative responses
LLM_TEMPERATURE=1.0   # Maximum creativity

LLM_MAX_TOKENS=4096   # Maximum response length
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

- Setting up ChromaDB

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
