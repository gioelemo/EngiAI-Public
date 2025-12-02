# Quick Start Guide

This guide will get you up and running with Engineer Assistant in 5 minutes.

## Prerequisites

Before starting, you need to set up the **MMORE RAG service** (required for document retrieval):

```bash
# Contact the MMORE team or check internal documentation for repository access
git clone <mmore-repository-url>
cd mmore
# Follow MMORE setup and deployment instructions
# Ensure it's running at http://localhost:8000
```

**What is MMORE?** A multimodal RAG (Retrieval-Augmented Generation) service that enables the assistant to search and answer questions about your uploaded documents (PDFs, papers, etc.).

---

## 🚀 Fastest Way: Docker (Recommended)

The easiest and most reliable way to get started:

### 1. Clone and Navigate

```bash
git clone https://github.com/gioelemo/engineer-assistant.git
cd engineer-assistant
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your OpenAI and Tavily API keys
```

Get API keys from:
- [OpenAI](https://platform.openai.com/api-keys)
- [Tavily](https://tavily.com/)

**Important:** Verify `MMORE_RAG_URL=http://localhost:8000` in your `.env` file points to your running MMORE service.

### 3. Start the Application

```bash
docker-compose up -d
```

### 4. Access the UI

Open your browser: **http://localhost:8501**

That's it! The complete AI assistant is now running with all dependencies isolated.

**To stop**: `docker-compose down`

---

## 💻 Local Development (Alternative)

For developers who want to modify the code:

### 1. Set Up Environment

If you haven't already, set up the conda environment:

```bash
# One-command setup (macOS/Linux)
./setup.sh

# Or manually
conda env create -f environment.yml
conda activate engineer-assistant
pre-commit install
```

See the [Installation Guide](installation.md) for detailed instructions.

### 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here

# Database (for local development, use SQLite)
DATABASE_URL=sqlite:///data/conversations.db

# Optional
WANDB_API_KEY=your-wandb-key
HPC_HOST_ALIAS=your-cluster-name
```

### 3. Start the UI

Launch the Streamlit interface:

```bash
streamlit run src/ui/streamlit_app.py
```

Or use the convenience script:

```bash
make run-ui
```

The UI will open in your browser at `http://localhost:8501`

---

## Next Steps

### Import Your Documents (Optional)

To enable document Q&A with your own PDFs:

```bash
# 1. Configure paper directory in .env
echo "PAPERS_SOURCE_DIR=/path/to/your/pdfs" >> .env

# 2. Import papers to MMORE
python scripts/import_local_papers.py

# 3. Ask questions about your documents
# In the UI: "What does my research say about topology optimization?"
```

See the [Paper Import Guide](paper_import_guide.md) for detailed instructions.

### Try Your First Query

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

## Optional: Host Service for GUI Integration

If you're using Docker and want to open GUI applications (like PrusaSlicer) on your host machine:

```bash
# In a separate terminal:
python host_service.py
```

This allows the containerized assistant to:
- ✅ Open PrusaSlicer for 3D model slicing
- ✅ Launch other GUI applications (VS Code, Terminal, etc.)
- ✅ Execute commands on your host machine

**Note:** Only needed for Docker deployments and only if you want GUI app integration.

## What's Next?

- [Docker Deployment Guide](docker_deployment.md) - Production deployment
- [Configuration Guide](configuration.md) - Customize your setup
- [Agent Documentation](usage/agents.md) - Learn about each agent
- [Tool Reference](usage/tools.md) - Explore available tools
- [Architecture Overview](architecture.md) - System design
- [Troubleshooting](troubleshooting.md) - Common issues

## Tips

- Use **natural language** - the assistant understands conversational queries
- Ask for **clarification** if you're unsure about something
- Use **context** - reference previous designs or papers in follow-up questions
- Check the **chat history** in the UI to review past interactions

## Need Help?

- [Troubleshooting Guide](troubleshooting.md)
- [GitHub Issues](https://github.com/gioelemo/engineer-assistant/issues)
- Check the `scripts/` directory for code examples
