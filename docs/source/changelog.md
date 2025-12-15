# Changelog

All notable changes to Engineer Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-12-15

### Added
- **Dual Voice Provider Support (ElevenLabs + OpenAI)**
  - Extended existing ElevenLabs voice integration with OpenAI as alternative provider
  - Voice provider selection in settings to switch between ElevenLabs and OpenAI
  - **ElevenLabs** (existing provider):
    - High-quality STT/TTS using `eleven_multilingual_v2` model
    - 6 voices: Rachel, Domi, Bella, Antoni, Josh, George
  - **OpenAI** (new provider):
    - OpenAI Whisper for speech-to-text (STT)
    - OpenAI TTS for text-to-speech with 6 voices (alloy, echo, fable, onyx, nova, shimmer)
    - Two TTS quality models: `tts-1` (standard, faster, cheaper) and `tts-1-hd` (high definition)
  - Per-conversation voice provider and voice selection persistence
  - Database schema update with `voice_provider` column for conversations
  - Environment variables for OpenAI voice configuration (`VOICE_PROVIDER`, `OPENAI_TTS_MODEL`, `OPENAI_TTS_VOICE`, `OPENAI_STT_MODEL`)
  - Provider-agnostic routing system maintaining backward compatibility
  - Comprehensive test suite (`test_voice.py`) covering all voice functionality
- **Interactive Whiteboard Feature (Excalidraw Integration)**
  - Native Excalidraw canvas integrated into chat interface
  - Real-time drawing and sketching capabilities for design discussions
  - Export canvas drawings as PNG images to chat
  - Custom message input for canvas exports with optional descriptions
  - Smart empty state detection with disabled send button when canvas is empty
  - Visual indicators and helpful tooltips for canvas state
  - Keyboard shortcuts:
    - `Ctrl+Enter` (Cmd+Enter on Mac) to send canvas to chat
    - `Ctrl+K` (Cmd+K on Mac) to create new chat
  - Native Streamlit button replacing iframe button for better UX
  - Automatic canvas clearing after successful send
- Script to get deployment zip to be uploaded on server
- **Photonics2D problem integration from EngiBench**
  - New optical device topology optimization problem for wavelength multiplexing
  - Support for designing photonic structures with multiple objectives
  - Seamlessly integrated with existing unified tool framework
- **Problem Registry System Simplification**
  - Centralized problem registry in `problems.py` as single source of truth
  - Dynamic objective extraction using `problem.objectives` attribute
  - Automatic prompt generation from problem registry
  - Removed all hardcoded problem-specific code from `engibench.py` and `engiopt.py`
  - Adding new problems now only requires updating `problems.py` - all tools and documentation update automatically
- **ThermoElastic2D problem integration from EngiBench**
  - 7 new tools for multi-physics topology optimization
  - Support for balancing structural and thermal performance
  - Unified tools: `create_problem`, `simulate_design`, `optimize_design`
  - Rendering and constraints: `render_design`, `check_constraints`
  - Problem metadata: `get_problem_details`, `get_dataset_info`
  - Weight parameter to control structural vs thermal optimization emphasis
- SSH authentication also via username and password
- Move supported problems in constant.py to avoid name duplicate
- Add button to remove all files in the outputs folder
- Improve connection state managment for Prusa
- MMORE multimodal RAG service integration for document retrieval
  - Replaced ChromaDB with external MMORE service
  - Enhanced document upload with retry logic and progress tracking
  - Improved error handling for large file uploads
  - Database tracking of uploaded documents

### Changed
- **Settings UI Reorganization**
  - Reorganized voice settings layout for better balance (voice selection full-width, options in 2 columns)
  - Moved Media Saving settings next to Voice Interaction for logical grouping
  - Swapped File Management and Media Saving positions for improved layout
- **UI/UX Improvements**
  - Replaced internal canvas button with native Streamlit button for consistency
  - Improved chat layout with better spacing on welcome screen
  - Enhanced delete button in chat sidebar (using ✕ instead of x)
  - Refactored chat.py with helper functions for better code organization
  - Fixed message display order during processing (canvas messages now appear correctly)
- Updated engineering agent system prompts with ThermoElastic2D workflow guidance
- Enhanced prompts with multi-physics optimization concepts and key parameters
- Migrated RAG system from ChromaDB to MMORE service
- Updated all documentation to reflect MMORE integration

### Fixed
- **Audio Playback Fix**
  - Fixed MP3 audio playback by using correct MIME type (`audio/mpeg` instead of `audio/mp3`)
  - Added proper MIME type mapping for all audio formats (mp3, wav, ogg, flac)
- **Code Quality Improvements**
  - Refactored OpenAI transcription function to reduce complexity (reduced branches from 13 to acceptable level)
  - Fixed linting issues: moved imports to top-level, used context managers for file operations
  - Replaced deprecated `tempfile.mktemp()` with `NamedTemporaryFile(delete=False)`
  - Used `Path.open()` and `Path.unlink()` instead of built-in `open()` and `os.unlink()`
- Deployment fix for SSH configuration (do not work for the moment)
- Fixed canvas export message ordering in chat display
- Corrected Docker build warning (changed `as` to `AS` in Dockerfile for consistency)
- Fixed complexity warnings in chat.py (reduced statements and branches)
- **Fixed all unit tests for generalized problem system**
  - Updated tests in `test_engibench.py` to work with dynamic objective extraction
  - Fixed `test_engiopt.py` to use problem registry instead of hardcoded imports
  - All tests passing with new problem-agnostic architecture

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
- Warning on API key usage (Tavily)
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

[Unreleased]: https://github.com/gioelemo/engineer-assistant/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/gioelemo/engineer-assistant/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/gioelemo/engineer-assistant/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/gioelemo/engineer-assistant/releases/tag/v0.0.1
