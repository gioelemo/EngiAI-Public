# Engineer Assistant v1.1.0 Release Notes

**Release Date:** December 15, 2025

## 🎉 Highlights

### Dual Voice Provider Support
- Choose between **ElevenLabs** and **OpenAI** for voice interactions
- Seamlessly switch providers in settings
- Per-conversation voice and provider persistence

### Interactive Whiteboard (Excalidraw)
- Draw and sketch directly in the chat interface
- Export drawings as images with descriptions
- Keyboard shortcuts for quick actions

### New Engineering Problems
- **Photonics2D**: Optical device topology optimization
- **ThermoElastic2D**: Multi-physics structural-thermal optimization

### MMORE RAG Integration
- Enhanced document retrieval with multimodal capabilities
- Replaces ChromaDB for more powerful semantic search

## 📋 What's New

### Voice Interaction Enhancements
- **ElevenLabs Provider** (High-quality, existing)
  - 6 voice options: Rachel, Domi, Bella, Antoni, Josh, George
  - `eleven_multilingual_v2` model for STT/TTS

- **OpenAI Provider** (New alternative)
  - Whisper for speech-to-text
  - 6 TTS voices: alloy, echo, fable, onyx, nova, shimmer
  - Two quality models: `tts-1` (fast) and `tts-1-hd` (high quality)

### Interactive Design Canvas
- Native Excalidraw integration for real-time drawing
- Export sketches directly to chat with custom messages
- Smart empty state detection
- Keyboard shortcuts:
  - `Ctrl+Enter` / `Cmd+Enter`: Send canvas to chat
  - `Ctrl+K` / `Cmd+K`: Create new chat

### Engineering Capabilities
- **Photonics2D Problem**
  - Wavelength multiplexing optimization
  - Multi-objective optical device design

- **ThermoElastic2D Problem**
  - Structural and thermal performance balancing
  - 7 unified tools for multi-physics optimization

- **Problem Registry System**
  - Simplified architecture with centralized problem registry
  - Dynamic objective extraction
  - Automatic tool and documentation generation

### Document Management
- MMORE service integration for advanced RAG
- Improved upload handling with retry logic
- Better error handling for large files
- Database tracking of uploaded documents

## 🔧 Improvements

### UI/UX
- Reorganized settings layout for better usability
- Improved chat layout and spacing
- Enhanced sidebar with better delete buttons
- Better message ordering during processing

### Code Quality
- Fixed audio playback with correct MIME types
- Reduced code complexity in multiple modules
- Improved test coverage for new problem system
- Better error handling throughout

### Configuration
- Moved SLURM settings to Settings UI (database-backed)
- Removed deprecated environment variables
- Simplified Docker configuration

## 🐛 Bug Fixes
- Fixed MP3 audio playback with correct MIME type
- Corrected canvas export message ordering
- Fixed Docker build warnings
- Resolved complexity warnings in chat module
- Updated all unit tests for new problem system

## 📦 Deployment

### Docker
Two deployment options:
1. **Basic**: `docker-compose up -d` (no Prusa integration)
2. **Full**: `docker-compose -f docker-compose.mcp.yml up -d` (with Prusa)

### Requirements
- Docker & Docker Compose
- OpenAI API key
- Tavily API key
- MMORE service (for RAG)
- Optional: Prusa MCP server (for 3D printing)

## 🔄 Migration Notes

### Breaking Changes
None - this release is backward compatible with v1.0.0

### Configuration Updates
If upgrading from v1.0.0:
1. Configure SLURM settings in Settings UI instead of `.env`
2. Optional: Add voice provider settings to `.env`
3. Optional: Configure MMORE service URL

### Removed Variables
The following env vars are now managed via Settings UI:
- `SLURM_VENV_PATH`
- `SLURM_PROJECT_PATH`
- `SLURM_EMAIL_USER`
- `SLURM_LOGS_DIR`
- `SLURM_WANDB_ENTITY`
- `SLURM_WANDB_PROJECT`
- `HF_HOME_REMOTE`
- `HF_DATASETS_CACHE_REMOTE`

## 📚 Documentation
- Updated configuration guide
- Enhanced Docker deployment docs
- New voice provider documentation
- Improved troubleshooting guide

## 🙏 Acknowledgments
- MMORE team for the RAG service
- EngiBench team for the problem library
- Community contributors and testers

## 📖 Full Changelog
See [CHANGELOG.md](CHANGELOG.md) for complete details.

## 🔗 Links
- [Installation Guide](docs/source/installation.md)
- [Configuration](docs/source/configuration.md)
- [Docker Deployment](docs/source/docker_deployment.md)
- [Troubleshooting](docs/source/troubleshooting.md)

---

**Install:** `git clone https://github.com/gioelemo/engineer-assistant && cd engineer-assistant`

**Quick Start:** `docker-compose up -d`
