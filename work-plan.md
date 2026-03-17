# Engineer Assistant — Post-Thesis Work Plan

## Context

The Engineer Assistant thesis (76 pages, compiled) needs minor polishing before the **April 13** deadline. After that, the focus shifts to the May 8 presentation, potential IDETC paper revision (notification April 27), code cleanup, and system improvements — all the way through the IDETC conference at end of August.

**Work schedule:**
- **Full-time (~40h/week)** until at least April 13, likely until May 8
- **Part-time (2 full days + 1 half day ≈ 20h/week)** from May 9 onwards

This gives roughly **560 usable hours** total. Since thesis polishing is minor, the full-time period before April 13 can also be used for code cleanup and early improvements. The plan below totals ~484h.

**Closed-loop manufacturing** is out of scope. Everything else the user listed is included.

---

## Fixed Milestones

| Date | Event |
|------|-------|
| Apr 3-6 | Easter break (Good Friday – Easter Monday) |
| **Apr 13** | **Thesis written report deadline** |
| Apr 20-23 | HPC AI conference (tentative, unavailable) |
| Apr 27 | IDETC paper notification |
| May 1 (Fri) | Labour Day (Swiss holiday) |
| **May 8** | **Final presentation (30 min)** |
| May 14-15 | Ascension Day + bridge day |
| **May 19** | **IDETC final paper submission + author registration** |
| May 25 (Mon) | Whit Monday (Swiss holiday) |
| Jun 29 – Jul 7 | Conference + holidays (unavailable) |
| Aug 1 (Sat) | Swiss National Day (weekend, no impact) |
| Aug 23-26 | IDETC conference |

---

## Phase 1: Thesis + Video + Early Cleanup (Mar 17 – Apr 13) ~120h (full-time)

Thesis content is largely final (mirrors the submitted IDETC paper). Polishing is light. Focus is: thesis, video, test fixes, and start deployment improvements.

| Week | Task | Hours | Details |
|------|------|-------|---------|
| Mar 17-20 (Tue-Fri, 4d) | Thesis polish + **YouTube video script** | 24h | Quick thesis read-through (content already solid from paper). Fix typos/formatting. Write video script and prepare demo scenarios. |
| Mar 23-27 (Mon-Fri, 5d) | **Record + edit YouTube video** | 30h | Record demo: all agents, optimization, STL export, Prusa, RAG, Excalidraw. 5-8 min. Edit and upload. This is the main focus for the week. |
| Mar 30 – Apr 2 (Mon-Thu, 4d) | **Fix tests** + **update deps** | 28h | Fix 2 pre-existing test failures (temperature assertion in arxiv_agent, rag_agent). Audit full test suite: verify correctness, remove obsolete tests, add missing coverage. Update dependencies one at a time with pytest after each. **Add Python 3.12+ support** (test compatibility, update Dockerfile, fix deprecations). |
| Apr 3-6 | Easter break (Fri-Mon) | 0h | No work. |
| Apr 7-10 (Tue-Fri, 4d) | **Start deployment simplification** | 32h | **Prerequisite:** Ensure MMORE PR #240 is merged before MMORE integration changes. (1) **MMORE as dependency**: Replace external `~/Desktop/mmore` folder with pip package or git submodule (depends on PR #240 merge). (2) Docker Compose profiles (`--profile full/dev/eval`). Replace env-var feature flags with profiles. (3) Docker secrets or auto-generated random credentials for DB (no plaintext). |
| Apr 13 (Mon, 1d) | **Submit thesis** | 8h | Final compile, submit. Remove thesis and paper submodules from main repo (they're separate repos, no longer needed in the codebase). Prusa MCP submodule sync. |

**Deliverables:** Thesis submitted. YouTube video published. Tests green. Dependencies updated. Deployment restructured (Docker profiles + MMORE).

---

## Phase 2: Presentation + Standardization + arXiv (Apr 14 – May 7) ~120h (full-time)

Presentation is the main focus. Standardization and GUI launcher (deferred from Phase 1) fill the gaps.

> **Note:** Apr 20-23 is blocked by the HPC AI conference (tentative). Presentation first draft must be done by Apr 17.

| Week | Task | Hours | Details |
|------|------|-------|---------|
| Apr 14-17 (Tue-Fri, 4d) | Presentation first draft + **rename to EngiAI** | 32h | ~25-30 slides, 30 min talk. Structure: motivation → architecture → tool integration → benchmarks → demo video clip → conclusions. Reuse thesis figures. **Must have a complete first draft by Apr 17.** **Rename repo** from `engineer-assistant` to `EngiAI` (package name, imports, Docker images, README, docs, CI). |
| Apr 20-23 | **HPC AI conference** (tentative) | 0h | Unavailable. |
| Apr 24 (Fri, 1d) | Presentation refinement | 8h | Polish slides post-conference. Fresh eyes review. |
| Apr 27-30 (Mon-Thu, 4d) | **Standardization (EngiBench/EngiOpt style)** + rehearsal | 32h | Match EngiBench/EngiOpt structure exactly: conventional commits via `conventional-commit-lint` pre-commit hook, Ruff config (line-length 124, Google docstrings, `select = ["ALL"]`), Pyright, Codespell, dynamic version from `__init__.py`, `CITATION.cff`, `CONTRIBUTING.md`, organized pyproject.toml extras, GPL-3.0, **same CI/CD pipeline** (GitHub Actions: automated testing, linting, releases, changelog, Docker/PyPI publishing, GitHub Pages docs). **Update Sphinx docs** to match current features + new name. 1 rehearsal. **Apr 27 = paper notification** — if revision needed → see Phase 3a. |
| ~~May 1 (Fri)~~ | **Labour Day** (Swiss holiday) | 0h | Off. |
| May 4-7 (Mon-Thu, 4d) | **arXiv paper** + rehearsals + **start GUI launcher** | 32h | Convert ASME paper to arXiv format (strip class deps, self-contained figures, arXiv metadata). Upload preprint. 2 full timed rehearsals + Q&A prep. Backup plan: pre-recorded demo video. Start **cross-platform GUI launcher** (Python + customtkinter/PySide6, PyInstaller): start/stop Docker, show health, open browser. **Admin panel in launcher** (password-protected): API keys (encrypted, masked), Docker service config. **Streamlit settings remain open**: model selection, voice, theme — accessible to all users. |
| **May 8 (Fri)** | **Presentation** | 8h | Present. |

**Deliverables:** Presentation delivered. arXiv preprint submitted. Codebase standardized (EngiBench/EngiOpt style). GUI launcher + secure settings working.

---

## Phase 3: System Improvements + Paper Revision (May 9 – Jun 27) ~110h (part-time, 20h/week)

Most code cleanup was done in Phases 1-2. This phase focuses on system improvements and paper revision.

### 3a. IDETC Paper Revision + Final Submission + GUI Launcher — May 9-19, ~20h (HARD DEADLINE)

Paper notification is Apr 27, final submission is **May 19**. This takes priority. GUI launcher finish runs in parallel.

> **Note:** May 14 (Thu) = Ascension Day, May 15 (Fri) = bridge day (likely off).

| Week | Task | Hours | Details |
|------|-------|---------|---------|
| May 9-13 (Mon-Wed before Ascension, 3d) | Paper revision + **finish GUI launcher** | 12h | Address reviewer feedback (if needed). Finish and test GUI launcher + admin panel from Phase 2. |
| ~~May 14-15~~ | **Ascension + bridge day** | 0h | Off. |
| May 16-19 (Mon only before deadline) | Camera-ready + author registration | 8h | Final formatting, submit by **May 19**. Register as author. |

*If paper accepted without revision, these hours go to cleanup. Either way, the May 19 deadline must be met.*

### 3b. Deep Code Cleanup — May 19 – Jun 6, ~35h

| Week | Task | Hours | Details |
|------|-------|---------|---------|
| May 19-23 (Mon-Fri) | **Dead code removal + deduplication** | 10h | Audit codebase for unused imports, functions, dead code paths, and duplicate logic. Use ruff's `F401`/`F841`. Check for duplicate patterns across agents. Remove stale files. Prune unused deps. |
| May 19-23 | **Prusa MCP cleanup + standardization** | 10h | Sync submodule (currently detached HEAD). Refactor and standardize to EngiBench/EngiOpt conventions: conventional commits, ruff/pyright config, CITATION.cff, CONTRIBUTING.md, pyproject.toml structure, GPL-3.0. |
| ~~May 25~~ | **Whit Monday** (Swiss holiday) | 0h | Off. |
| May 26 – Jun 6 (Tue-Fri + Mon-Fri, 9d) | **(Extract Excalidraw component)** | 15h | Extract `src/ui/components/excalidraw/` to standalone `streamlit-excalidraw` repo with own CI. Standardize to EngiBench/EngiOpt conventions. Do before MCP Excalidraw work (Phase 4). |

### 3c. System Improvements — Jun 8 – Jun 26, ~55h

| Week | Task | Hours | Details |
|------|------|-------|---------|
| Jun 8-12 (Mon-Fri) | **Full Ollama support in chatbot** | 15h | Ollama model selection in Streamlit settings. Test with llama3, mistral. Handle tool-calling quirks. Add Ollama container to Docker Compose. |
| Jun 15-19 (Mon-Fri) | **Better outputs for long-running tasks** + **Error recovery** | 20h | Streaming progress in Streamlit during optimization. Intermediate design visualization. HPC job polling with real-time status. Retry with exponential backoff for API calls. LangGraph error nodes. Graceful degradation (MMORE down → local, Prusa unreachable → save locally). |
| Jun 22-26 (Mon-Fri) | **Experiment recording system** + **Google audio model support** | 20h | Set up recording infrastructure for human-AI experiments (Opencast/Tobira or OBS-based): screen capture + webcam, session storage/management. Gemini audio integration (1:1 with OpenAI voices, extend `voices.py`). |

**Jun 29 – Jul 7: Conference + Holidays (unavailable)**

---

## Phase 4: Feature Development (Jul 8 – Aug 15) ~120h (part-time, 20h/week)

### 4a. Core Features — Jul 8-24, ~55h

| Week | Task | Hours | Details |
|------|------|-------|---------|
| Jul 8-10 + Jul 13-17 (8d) | **Interactive MCP Excalidraw whiteboard** | 35h | Integrate the official excalidraw-mcp server with the existing Streamlit Excalidraw component (`src/ui/components/excalidraw/`). Add as a new MCP service (like Prusa MCP). Connect agent to whiteboard: agent can draw topology results, annotate designs; user sketches constraints that agent interprets. Bridge the MCP server ↔ Streamlit component communication. |
| Jul 20-24 (Mon-Fri) | **More RAG file formats** + **Extend to other problems/models** | 20h | Extend MMORE client: .pptx, .ipynb, video files, and other common formats. Update rag_tools.py. Containerized problem solvers (input/output JSON interface). Example container for new problem type. Update problem registry. |

### 4b. New Agents + Repo Transfer — Jul 27 – Aug 14, ~65h

| Week | Task | Hours | Details |
|------|------|-------|---------|
| Jul 27-31 + Aug 3 (Mon-Mon, 6d) | **Self-writing training code agent** | 30h | New `TrainingAgent` (BaseAgent pattern). Tools: generate_training_script, run_training, evaluate_model. Automatic validation: generated code is compiled/syntax-checked before submission, no manual review step. HPC integration for Euler. Enable extended HPC benchmarks (longer training runs, multi-model evaluation). Full testing: unit tests for agent/tools, integration test with mock HPC, end-to-end test on Euler. |
| Aug 4-5 (Tue-Wed) | **Repo transfer to lab GitHub org** | 8h | Repo already renamed to EngiAI (Phase 2). Clean any remaining hardcoded paths/credentials. Final verification of EngiBench/EngiOpt-style structure. Transfer/fork under lab org. Handoff document. Tag v2.0.0. |
| Aug 6-7 (Thu-Fri) | **New computer setup** | 10h | Set up dev environment on new machine. Verify Docker deployment from clean clone (serves as deployment validation). Install deps, test all services via GUI launcher. |
| Aug 10-14 (Mon-Fri) | **Buffer / overflow** | 17h | Catch up on any tasks that slipped. Polish features. Final testing. |

---

## Phase 5: IDETC + Wrap-Up (Aug 17 – Aug 28) ~20h

| Date | Task | Hours |
|------|------|-------|
| Aug 17-21 (Mon-Fri) | IDETC prep + travel | 10h |
| Aug 23-26 (Sun-Wed) | IDETC conference | — |
| Aug 27-28 (Thu-Fri) | Final wrap-up | 10h |

Wrap-up: update README/docs with all new features, final release tag, archive, status document for the lab.

---

## Hours Summary

| Phase | Period | Pace | Hours | Focus |
|-------|--------|------|-------|-------|
| 1. Thesis + Video + Tests | Mar 17 – Apr 13 | Full-time | 122h | Thesis, video, test fixes, deps, deployment restructure |
| 2. Presentation + Standardization + arXiv | Apr 14 – May 8 | Full-time | 112h | Slides, rename, standardization, CI/CD, arXiv, GUI launcher (minus HPC AI conf + May 1) |
| 3. Cleanup + Improvements | May 9 – Jun 27 | Part-time | 110h | Paper revision, dead code, Excalidraw extraction, Prusa MCP, Ollama, outputs, errors, recording, audio |
| 4. Features | Jul 8 – Aug 15 | Part-time | 120h | MCP whiteboard, training agent, RAG formats, containers, repo transfer, new computer |
| 5. IDETC | Aug 17 – Aug 28 | — | 20h | Conference + wrap-up |
| **Total** | | | **484h** | |

**Available hours:**
- Full-time (Mar 17 – May 8, ~7 weeks minus Easter/HPC AI conf): ~230h
- Part-time (May 9 – Aug 31, ~16 usable weeks × 20h): ~320h
- **Total available: ~550h → buffer of ~66h**

### Is This Feasible?

**Yes, comfortably.** The full-time period (Phases 1-2) handles all the deadline-critical work plus most code cleanup. By May 8, the thesis is submitted, presentation done, arXiv uploaded, docs updated, and the codebase is clean. Phases 3-4 at part-time are modular — individual tasks can be paused, reordered, or dropped if paper revision demands more time.

If IDETC paper is accepted without revision, you gain ~30h extra buffer.

---

## Risk Factors

| Risk | Impact | Mitigation |
|------|--------|------------|
| Paper rejected (Apr 27) | Frees ~30h; need to decide on resubmission venue | arXiv preprint ensures visibility. Redirect hours to features. |
| Heavy paper revision | Eats into Phase 3b cleanup time | Defer some cleanup tasks to Phase 4. |
| MCP Excalidraw harder than expected | 30h may not be enough | Ship a simpler version (display-only, no bidirectional drawing) and document full vision. |
| Ollama models can't do tool calling | Feature limited in scope | Document which models are compatible. Fall back to "Ollama for Q&A, API for tools." |
| Dependency updates break things | Could cascade | Update one at a time, test after each. Keep pre-update tag. |

---

## Optional / Deferred Tasks

These can be scheduled if buffer hours open up (e.g., paper accepted without revision):

- **Ablation studies** (prompt/temperature/model tests) — ~15-20h. May be useful for arXiv paper or future publication. Run if paper reviewers request it or if time permits.

## Tasks NOT Included

- **Closed-loop manufacturing feedback** — out of scope per user decision.
- **User studies** — mentioned in thesis as future work, not planned here.
