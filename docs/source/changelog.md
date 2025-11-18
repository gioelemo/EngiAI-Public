# Changelog

All notable changes to Engineer Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Script to get deployment zip to be uploaded on server
- ThermoElastic2D problem integration from EngiBench
  - 7 new tools for multi-physics topology optimization
  - Support for balancing structural and thermal performance
  - Unified tools: `create_problem`, `simulate_design`, `optimize_design`
  - Rendering and constraints: `render_design`, `check_constraints`
  - Problem metadata: `get_problem_details`, `get_dataset_info`
  - Weight parameter to control structural vs thermal optimization emphasis

### Changed
- Updated engineering agent system prompts with ThermoElastic2D workflow guidance
- Enhanced prompts with multi-physics optimization concepts and key parameters

### Fixed
- Deployment fix for SSH configuration (do not work for the moment)

## [1.0.0] - 2025-11-17

### Added
- Documentation using Sphinx with EngiBench-style template
- Comprehensive test suite for ArXiv agent
- ArXiv agent for searching and analyzing research papers
- RAG agent for knowledge base queries
- HPC agent for cluster job submission and monitoring
- Improved Web UI using Streamlit
- CLI interface for command-line usage
- Docker build system (2 separate containers: Chatbot + MCP server for Prusa)
- Support to change model temperature
- Warning on API key usage (Tavily and Mathpix)
- Open Application from Docker into local GUI using host_service.py
- Enhanced suggested prompts extraction with fallback patterns for malformed output
- Improved Prusa agent system prompts with explicit formatting requirements

### Changed
- Updated documentation theme to sphinx-book-theme
- Reorganized dependencies alphabetically in configuration files
- Switched to dirhtml builder for cleaner URLs
- Removed non-functional preview links from Prusa print job outputs

### Fixed
- Mock author structure in ArXiv agent tests
- Different small fix
- Wandb Download works again
- Suggested prompts now display correctly in Streamlit when using Prusa agent
- Network connectivity between chatbot and Prusa MCP server containers

## [0.0.1] - 2025-11-06

### Added
- Initial release
- Multi-agent system with LangGraph
- Engineering optimization tools
- Integration with EngiBench and EngiOpt
- 3D printer control via Prusa agent
- Vector store for document retrieval
- Support for various LLM backends

[Unreleased]: https://github.com/gioelemo/engineer-assistant/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/gioelemo/engineer-assistant/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/gioelemo/engineer-assistant/releases/tag/v0.0.1
