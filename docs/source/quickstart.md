# Quick Start Guide

This guide will get you up and running with Engineer Assistant in 5 minutes.

## 1. Activate Environment

First, make sure your conda environment is activated:

```bash
conda activate engineer-assistant
```

## 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here

# Optional
WANDB_API_KEY=your-wandb-key
HPC_HOST_ALIAS=your-cluster-name
```

Get API keys from:
- [OpenAI](https://platform.openai.com/api-keys)
- [Tavily](https://tavily.com/)
- [Weights & Biases](https://wandb.ai/authorize) (optional)

## 3. Start the UI

Launch the Streamlit interface:

```bash
streamlit run src/ui/streamlit_app.py
```

Or use the convenience script:

```bash
./run_ui.sh
```

The UI will open in your browser at `http://localhost:8501`

## 4. Try Your First Query

In the chat interface, try these example queries:

### Search for Papers
```
Find recent papers on topology optimization
```

### Optimize a Design
```
Create an optimized 2D beam design with 40% material
```

### Export to STL
```
Export the current design to STL format
```

## 5. Using the CLI

For command-line usage:

```bash
python -m src.cli.chat
```

Then interact with the assistant in your terminal.

## Common Workflows

### Research Workflow

1. **Search ArXiv**: "Find papers on generative design"
2. **Analyze Papers**: "Download and analyze the first paper"
3. **Ask Questions**: "What methods do they use for optimization?"

### Design Workflow

1. **Generate Design**: "Create a 2D beam design"
2. **Optimize**: "Optimize for minimal compliance"
3. **Export**: "Convert to STL for 3D printing"

### HPC Training Workflow

1. **Prepare**: "Set up training for beams2d problem"
2. **Submit**: "Submit training job to HPC cluster"
3. **Monitor**: "Check training status"

## What's Next?

- [Configuration Guide](configuration.md) - Customize your setup
- [Agent Documentation](usage/agents.md) - Learn about each agent
- [Tool Reference](usage/tools.md) - Explore available tools
- [API Reference](api/agents.rst) - Developer documentation

## Tips

- Use **natural language** - the assistant understands conversational queries
- Ask for **clarification** if you're unsure about something
- Use **context** - reference previous designs or papers in follow-up questions
- Check the **chat history** in the UI to review past interactions

## Need Help?

- Documentation: [Full Docs](index.rst)
- Issues: [GitHub Issues](https://github.com/gioelemo/engineer-assistant/issues)
- Examples: Check the `scripts/` directory for code examples
