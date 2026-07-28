<p align="center">
<img src="assets/logo_readme.png" align="center" width="50%"/>
</p>

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Python tests](https://github.com/gioelemo/EngiAI-Public/actions/workflows/test.yml/badge.svg)](https://github.com/gioelemo/EngiAI-Public/actions/workflows/test.yml)
[![pre-commit](https://github.com/gioelemo/EngiAI-Public/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/gioelemo/EngiAI-Public/actions/workflows/pre-commit.yaml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

> ⚠️ **Status (2026-05-19):** This is a public snapshot of EngiAI accompanying the IDETC 2026 paper. The codebase is actively evolving in a private repository and will be made fully public as soon as the journal extension is published. For the smoothest experience, follow the demo setup in [README.demo.md](README.demo.md). The full version works but the RAG (MMORE) integration currently requires manual setup — official Docker images for RAG will be released soon.

EngiAI provides an AI-powered assistant for mechanical engineering design, integrating topology optimization, research retrieval, HPC simulation, and 3D printer control in a single conversational interface.

📄 **Paper (arXiv):** https://arxiv.org/abs/2605.19743

🎥 **Demo video:** https://www.youtube.com/watch?v=QbQVZFCq3X0

## Quick start

**New here?** Follow the lightweight demo build in [README.demo.md](README.demo.md) — chatbot + Postgres only, no external dependencies, works on Windows / macOS / Linux.

**Full stack** (Docker):

```bash
git clone https://github.com/gioelemo/EngiAI-Public.git
cd EngiAI-Public
cp .env.example .env       # add OPENAI_API_KEY / GOOGLE_API_KEY / TAVILY_API_KEY
docker compose up -d
```

Then open <http://localhost:8501>. The RAG (MMORE) service requires separate setup — see [docs/source/installation.md](docs/source/installation.md).

## Agents

The supervisor routes requests to specialized agents:

- **Engineering** — topology optimization with EngiBench (beams2d, ThermoElastic2D, Photonics2D)
- **Search** — web research via Tavily
- **RAG** — document Q&A with MMORE multimodal RAG
- **ArXiv** — scientific paper search and analysis
- **Prusa** — 3D printer control via MCP (optional)
- **HPC** — SLURM job management over SSH
- **CLI** — local shell with human-in-the-loop confirmation

Details and examples: [docs/source/usage/agents.md](docs/source/usage/agents.md).

## Documentation

Full documentation lives in [`docs/`](docs/source/). A hosted documentation site is planned (in the style of <https://engibench.ethz.ch>).

| Topic | Link |
| --- | --- |
| Installation (Docker + conda) | [installation.md](docs/source/installation.md) |
| Quick start | [quickstart.md](docs/source/quickstart.md) |
| Configuration (env vars, LLM models, tracing) | [configuration.md](docs/source/configuration.md) |
| Architecture | [architecture.md](docs/source/architecture.md) |
| Using agents | [usage/agents.md](docs/source/usage/agents.md) |
| HPC integration (SLURM / SSH) | [usage/hpc.md](docs/source/usage/hpc.md) |
| 3D printer (Prusa) | [usage/prusa.md](docs/source/usage/prusa.md) |
| Web UI | [usage/ui.md](docs/source/usage/ui.md) |
| Benchmarking with Weave | [weave_integration.md](docs/source/weave_integration.md) |
| Deployment | [deployment.md](docs/source/deployment.md) |
| Database setup | [database_setup.md](docs/source/database_setup.md) |
| Paper import (RAG) | [paper_import_guide.md](docs/source/paper_import_guide.md) |
| Troubleshooting | [troubleshooting.md](docs/source/troubleshooting.md) |
| Contributing | [contributing.md](docs/source/contributing.md) |

To build the docs locally: `make docs` (or `make docs-watch` for live preview).

## Repository layout

```
src/          # agents, tools, UI, models
benchmarks/   # evaluation framework and per-problem prompts
services/     # standalone services (host_service, Prusa MCP)
docs/         # full Sphinx documentation
scripts/      # utility scripts
tests/        # unit and integration tests
```

## Getting help

- [Troubleshooting guide](docs/source/troubleshooting.md)
- [GitHub Issues](https://github.com/gioelemo/EngiAI-Public/issues)
- Container logs: `docker compose logs -f`
