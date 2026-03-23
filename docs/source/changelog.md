# Changelog

All notable changes to Engineer Assistant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Thesis Visualization and Analysis Tools**
  - Complete evaluation data extraction pipeline in `benchmarks/evaluations/extract_data.py`
    - Extracts tool usage, token consumption, latency, and metadata from Weave evaluation traces
    - Support for both `Evaluation.predict_and_score` and direct `EngineeringAgent.predict` calls
    - Automatic metadata extraction from WeaveDict objects (example_id, seed, problem_id)
    - Token usage extraction from nested model-specific summary data
    - Latency tracking from Weave execution metadata
    - CSV export with configurable column ordering (seed first for grouping analysis)
    - Missing tool columns automatically filled with 0 for clean data analysis
    - Refactored with helper functions for improved maintainability (reduced complexity)
  - Comprehensive visualization suite in `benchmarks/evaluations/plots/`
    - `plot_tool_usage.py`: Tool usage analysis with frequency charts, heatmaps, and per-model comparisons
    - `plot_design_quality.py`: Design quality distribution visualization with KDE overlays
    - `plot_metrics_comparison.py`: Side-by-side model performance comparison across all metrics
    - `plot_iou_vs_objective.py`: Correlation analysis between design quality (IoU) and objective scores
    - `plot_dpp_vs_mmd.py`: Diversity vs distribution similarity scatter plot analysis
    - `plot_dpp_vs_fog.py`: Design diversity vs optimization quality analysis
    - `generate_summary_table.py`: LaTeX-formatted summary statistics table generation
    - `run_all.py`: Automated execution of all visualization scripts with configuration
    - `utils.py`: Shared utilities for data loading, preprocessing, and plot styling (308 lines)
  - LaTeX-quality plot styling with custom fonts and consistent formatting
  - Automatic figure export to `benchmarks/evaluations/plots/figures/`
  - Documentation in `benchmarks/evaluations/TOOL_USAGE.md` (170 lines)

### Changed
- **Data Extraction Improvements**
  - Renamed output files from `tool_usage.csv` to `data.csv` for more general nomenclature
  - Enhanced metadata extraction with multiple fallback strategies for robustness
  - Replaced try-except-pass with `contextlib.suppress` for cleaner error handling
  - Added constant `REF_EXTRA_MIN_LENGTH` to replace magic values
  - Optimized dictionary iteration using `.values()` instead of `.items()` when keys unused
  - Refactored complex functions into focused helper functions:
    - `_extract_metadata_from_example()`: Handles all metadata extraction logic
    - `_extract_tokens_and_latency()`: Handles token and latency extraction
  - Improved code maintainability (reduced from 485 to 490 lines with better organization)

### Fixed
- VS Code Pylance import resolution for `benchmarks/evaluations/plots/` modules
  - Added `python.analysis.extraPaths` configuration to `.vscode/settings_template.json`
  - Created `.vscode/settings.json` from template with proper Python path configuration
  - Resolved false positive import errors in editor while maintaining clean lint output

### Dependencies
- Added visualization dependencies to `pyproject.toml`:
  - `matplotlib`: Core plotting library
  - `seaborn`: Statistical data visualization
  - `scipy`: Scientific computing and statistics

---

## [Previous Releases]

### Added
- **Complete Benchmarking Infrastructure**
  - Built entire `benchmarks/` directory structure from scratch for systematic agent evaluation
  - Comprehensive documentation in `benchmarks/README.md` with quick start guide and examples
  - Support for multiple problem types, models, and configurations with results tracking via Weave
  - ~2,800 lines of new code implementing evaluation framework, prompt generation, and scoring

- **Weave Evaluation Framework Integration**
  - Unified evaluation script (`benchmarks/evaluations/evaluate_agent.py`) for benchmarking across problem types
  - `EngineeringAgent` Weave Model wrapper for multi-agent supervisor system
  - Dataset creation and management with `get_or_create_dataset()` for prompt datasets
  - Automatic evaluation results tracking and visualization in Weave dashboard
  - Evaluation results organized by model and problem type
  - Documentation in `benchmarks/evaluations/README.md`

- **Beams2D Problem Benchmarks**
  - Complete beams2d topology optimization benchmark suite
  - Integration with HuggingFace dataset `IDEALLab/beams_2d_50_100_v0` (3.88k examples)
  - Prompt generation script (`generate_prompts.py`) with Weave integration and templating
  - Comprehensive prompt validation script (`validate_prompts.py`) with numerical accuracy checks, parameter range validation, and completeness verification
  - Dataset exploration tool (`explore_dataset.py`) for analyzing beam design parameters and statistics
  - Problem-specific scoring functions in `benchmarks/problems/beams2d/scorers.py`:
    - `score_design_match()` - Multi-metric scorer with IoU, pixel accuracy, volume fraction, and compliance
    - Compliance extraction from tool messages with regex parsing
    - Design comparison visualization with side-by-side agent vs ground truth images
  - Documentation in `benchmarks/problems/beams2d/README.md`

- **Shared Benchmarking Utilities**
  - `benchmarks/shared/utils.py` with reusable functions:
    - `get_hf_dataset()` - HuggingFace dataset loading with caching
    - `extract_design_from_tool_messages()` - Design array extraction from agent messages
    - `create_design_comparison()` - Side-by-side visualization of designs
  - Modular architecture for adding future problem types (thermoelastic2d, 3D structures, etc.)

- **Agent Evaluation Mode**
  - `eval_mode` parameter in `SupervisorAgent` for reduced token costs during benchmarking
  - Minimal agent prompts optimized for evaluation tasks
  - Skip MMORE and MCP server requirements during evaluation (`SKIP_MMORE`, `SKIP_MCP` env vars)
  - Parallel design generation support with proper isolation

- **Independent Weave Tracking Control**
  - `USE_WEAVE_CHATBOT` configuration flag to control Weave tracking for chatbot interactions separately from evaluations
  - Allows disabling chatbot tracking overhead while keeping evaluation tracking enabled
  - Documented in `.env.example`, `README.md`, and `docs/source/weave_integration.md`

- **Generic Problem Registry System (v2.0 Architecture)**
  - Centralized problem registry in `benchmarks/shared/problem_registry.py` as single source of truth
  - Declarative problem configuration using `ProblemConfig`, `ObjectiveConfig`, and `ConditionConfig` dataclasses
  - Automatic prompt generation, scoring, and evaluation for all problems from unified configuration
  - Adding new problems now only requires updating `PROBLEMS` dictionary - all tools and scorers adapt automatically
  - Eliminates all problem-specific hardcoded logic from evaluation framework
  - ~1,350 lines of new infrastructure code implementing generic architecture

- **Generic Scoring Framework**
  - `output_quality_visual_scorer.py`: Universal per-design metrics working across all topology optimization problems
    - IoU (Intersection over Union) for topology matching
    - Pixel accuracy for density field comparison
    - MSE (Mean Squared Error) for continuous design similarity
    - Generic constraint validation from problem configuration
    - Generic objective extraction and scoring from problem configuration
  - `output_quality_engibench_scorer.py`: Global metrics computed after full evaluation
    - MMD (Maximum Mean Discrepancy) for distribution similarity (configurable sigma parameter)
    - DPP (Determinantal Point Process) diversity metric
    - RVC (Ratio of Violated Constraints) with detailed violation reporting
    - IOG (Initial Optimality Gap) for assessing design initialization quality
    - COG (Current Optimality Gap) for mid-optimization performance
    - FOG (Final Optimality Gap) for end-state optimization quality
  - `objective_extractor.py`: Problem-agnostic objective value extraction from tool messages
    - Regex-based extraction from JSON and plain text outputs
    - Support for field aliases and multiple extraction strategies
    - Automatic objective score calculation based on optimization direction and error thresholds
  - `problem_config.py`: Declarative configuration system for problems, objectives, and conditions
  - Configurable metric weights per problem via `design_metrics_weights` in problem config

- **Advanced Metrics and Analysis**
  - `benchmarks/shared/metrics.py`: Implementation of statistical metrics
    - MMD with Gaussian RBF kernel for comparing design distributions
    - DPP diversity using determinant-based measure
    - Optimality gap calculations (Initial, Current, Final) using EngiBench optimization history
    - Multiprocessing support for efficient parallel computation
  - `compute_output_quality_design_stats.py`: Per-design statistics aggregation across multiple seeds
    - Mean ± std computation for design-level metrics (IoU, pixel accuracy, constraint/objective scores)
    - Supports both core metrics and problem-specific extensions
    - CSV output with statistical summaries
  - `compute_output_quality_global_stats.py`: Global statistics aggregation
    - Summary statistics for MMD, DPP, RVC, and optimality gaps
    - Aggregation across multiple optimization seeds
    - Statistical validation of model performance

- **Photonics2D Problem Benchmarks**
  - Complete photonics2d optical device topology optimization benchmark suite
  - Integration with HuggingFace dataset `IDEALLab/photonics_2d_120_120_v0`
  - Wavelength multiplexing optimization with multiple eigenvalue parameters
  - Objective: Maximize total_overlap between target and simulated field distributions
  - Conditions: lambda1, lambda2 (wavelength parameters), blur_radius (smoothing parameter)
  - Prompt generation script with problem-specific templating
  - Improved design visualization with proper colormap and rendering
  - Documentation in `benchmarks/problems/photonics2d/README.md`
  - ~357 lines of new code for photonics2d support

- **ThermoElastic2D Problem Benchmarks** (Experimental)
  - Multi-objective coupled physics topology optimization benchmark
  - Integration with HuggingFace dataset `IDEALLab/thermoelastic_2d_v0`
  - Three competing objectives:
    - Structural compliance (minimize for stiffness)
    - Thermal compliance (minimize for heat dissipation)
    - Volume fraction (minimize for material efficiency)
  - Support for boundary condition arrays (fixed elements, force elements, heatsink elements)
  - Weight parameter to control structural vs thermal optimization emphasis
  - Prompt generation script with multi-objective templating
  - Documentation in `benchmarks/problems/thermoelastic2d/README.md`
  - ~536 lines of new code for thermoelastic2d support
  - Note: Not fully functional yet, marked as experimental

- **Evaluation Infrastructure Enhancements**
  - Multiple seed evaluation support for statistical robustness
    - Run evaluations with multiple random seeds and aggregate results
    - Standardized seed handling across all problem types
    - Automatic seed-specific result tracking in Weave
  - Session isolation for parallel evaluation
    - `set_session_id()` and `clear_session_state()` functions in `engibench.py`
    - Prevents cross-contamination between concurrent optimization runs
    - Improved caching and state management
  - Random sampling for prompt generation with reproducibility
    - `--seed` parameter in prompt generation scripts for reproducible dataset sampling
    - Random sample selection instead of sequential for better dataset variability
    - Sorted indices ensure consistent iteration order after random selection
  - Generic dataset exploration tool (`benchmarks/shared/explore_dataset.py`)
    - Works with any problem in the registry
    - Statistical analysis of design parameters and conditions
    - Visualization of design distributions
  - Prompt generation utilities (`benchmarks/shared/prompt_generation.py`)
    - Shared prompt generation logic across all problems
    - Templating system for consistent prompt structure
    - Validation and error handling for generated prompts
  - Comprehensive test suite additions:
    - `test_objective_extractor.py`: Tests for generic objective extraction
    - `test_problem_config.py`: Tests for problem configuration validation
    - `test_evaluate_agent.py`: Tests for evaluation framework
    - `test_session_isolation.py`: Tests for parallel execution isolation

- **Model Evaluation and Analysis Tools**
  - CSV-based metrics tracking for model comparison
  - `cgan_cnn_2d_beams2d_metrics.csv`: Baseline metrics for generative models
  - Scripts for computing statistics across model evaluations
  - Support for comparing agent performance against generative baselines

### Changed
- **Agent System Enhancements for Evaluation**
  - Updated `supervisor_agent.py`, `engineering_agent.py`, and `rag_agent.py` to support eval mode
  - Added `skip_mmore` parameter to RAG agent for running without Docker
  - Improved logging in engineering agent for evaluation debugging
  - Enhanced `engibench.py` to handle parallel design requests correctly
  - Updated agent prompts to reduce token usage during evaluation
  - Improved agent prompts with explicit instructions to not mention tool names to users
  - Simplified prompts for better model understanding and reduced verbosity

- **Benchmark Code Architecture**
  - Complete refactoring from problem-specific to generic architecture
  - Removed all problem-specific scoring modules (`benchmarks/problems/beams2d/scorers.py`)
  - Replaced with unified generic scorers that work across all problems
  - Separated problem-specific scoring logic from shared utilities
  - Organized code by problem type in `benchmarks/problems/` directory structure
  - Shared utilities in `benchmarks/shared/` for cross-problem functionality
  - Evaluation framework in `benchmarks/evaluations/` for unified benchmarking
  - Problem configurations now auto-generate from central registry
  - All 85+ commits focused on generalization and standardization

- **Scoring System Evolution and Standardization**
  - Unified scoring interface across all problems via problem registry
  - Consistent metric naming conventions across problems
  - Standardized constraint checking using tolerance-based approach from EngiBench paper
  - Metric weights now configurable per problem in registry
  - Iteratively improved scoring metrics based on evaluation results:
    - Added IoU (Intersection over Union) for topology matching
    - Added pixel accuracy for density comparison
    - Added MSE for continuous similarity
    - Added constraint score for volume fraction and other constraints
    - Added objective score for optimization performance
  - Optimized default scoring weights: 40% IoU, 25% pixel accuracy, 15% constraint match, 20% objective match
  - Moved from problem-specific parsers to generic objective extraction
  - Support for multiple objective aliases for robust extraction
  - Improved objective extraction from tool messages using multiple strategies (JSON, regex, dict)

- **Dataset Management**
  - Improved dataset creation workflow with validation and error handling
  - Added comparison images saved to `evaluations/results/models/{model}/{problem}/{prompt_style}/{rag_status}/comparisons/`
  - Better organization of evaluation data and results
  - Dataset split support (train/val/test) in prompt generation
  - Validation reports for prompt quality assurance
  - Moved dataset exploration to shared utilities for reusability

- **Evaluation Workflow**
  - Added `--scorers` flag to control which metrics are computed
    - `generic` or `all`: Full per-design and global metrics
    - `engibench`: Lightweight design extraction only for faster evaluation
  - Disabled Weave tracing for prompt generation to reduce overhead
  - Suppressed Pydantic deprecation warnings for cleaner output
  - Improved evaluation logging with progress indicators
  - Better error handling and recovery in evaluation pipeline
  - Async/await optimization for parallel evaluation
  - Added time tracking for full evaluation uploads to Weave

- **EngiBench Tool Integration**
  - Enhanced `src/tools/engibench.py` with session management
  - Removed sparse and uniform design initialization options (focused on optimized designs)
  - Added optimization history extraction for optimality gap metrics
  - Improved tool output formatting for objective values
  - Better caching and state management for parallel executions
  - Fixed parameter handling for multi-objective problems

- **Documentation and Code Quality**
  - Updated benchmarks README files for all problem types
  - Improved docstrings with detailed parameter descriptions
  - Added type hints throughout evaluation framework
  - Comprehensive inline comments for complex logic
  - Ruff and mypy compliance across all new code
  - Excluded `metrics.py` from some linting rules due to external dependencies
  - Improved naming conventions for consistency (e.g., scorer function names)
  - Better error messages and logging throughout

- **Configuration and Settings**
  - Updated `pyproject.toml` with new dependencies and configurations
  - Enhanced `config.py` with evaluation-specific settings
  - Updated Weave integration documentation (`docs/source/weave_integration.md`)
  - Better environment variable handling for evaluation vs chatbot modes

### Fixed
- Fixed parallel design generation issue preventing concurrent optimization runs
- Fixed design configuration not being used correctly in evaluation
- Fixed mypy type error in scorers by explicitly typing result dictionary as `dict[str, Any]`
- Fixed test suite to support new evaluation infrastructure
- Removed unused imports and cleaned up code (~90 lines of dead code removed)
- Fixed caching issues in EngiBench tool with proper session isolation
- Fixed test session isolation to prevent state contamination between tests
- Fixed parameter ordering and naming inconsistencies in tool calls
- Fixed Ruff linting issues throughout benchmarks codebase
- Fixed mypy type checking errors in new evaluation infrastructure
- Improved printed output formatting in evaluation results
- Fixed compliance extraction edge cases with better regex patterns
- Fixed dataset split handling (using test split for evaluation, not training)
- Fixed constraint violation detection to match EngiBench paper convention
- Fixed volume fraction constraint checking with proper tolerance handling
- Fixed photonics2d design rendering and visualization
- Fixed CSV generation for statistics scripts
- Fixed format inconsistencies in metrics output
- Fixed missing docstrings in several modules
- Removed redundant logging statements
- Cleaned up old evaluation result files and data artifacts
- Fixed order of samples in evaluation to ensure reproducibility
- Fixed complexity warnings in evaluation code by refactoring large functions
- Fixed comment and docstring inconsistencies mentioning removed metrics
- Fixed small issues in statistics computation scripts

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
- Improve connection state management for Prusa
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
