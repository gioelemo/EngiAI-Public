# Changelog

All notable changes to Engineer Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed
- Updated documentation theme to sphinx-book-theme
- Reorganized dependencies alphabetically in configuration files
- Switched to dirhtml builder for cleaner URLs

### Fixed
- Mock author structure in ArXiv agent tests
- Different small fix (see full changelog)

## [0.0.1] - 2025-11-06

### Added
- Initial release
- Multi-agent system with LangGraph
- Engineering optimization tools
- Integration with EngiBench and EngiOpt
- 3D printer control via Prusa agent
- Vector store for document retrieval
- Support for various LLM backends

[Unreleased]: https://github.com/gioelemo/engineer-assistant/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/gioelemo/engineer-assistant/releases/tag/v0.0.1
