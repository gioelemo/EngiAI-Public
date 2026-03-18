# EngiAI — Post-Thesis Work Plan

**Author:** Gioele Molinari<br>
**Date:** March 2026<br>
**Project:** EngiAI — A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design

---

## Goals

1. **Make the system production-ready and easy to deploy** — Ensure full test coverage, update all dependencies, simplify the Docker-based deployment with a graphical launcher for non-technical users, standardize the codebase to match the lab's EngiBench/EngiOpt conventions, and set up automated CI/CD pipelines.

2. **Extend the system with new capabilities** — Add support for local language models (Ollama) in the chat interface, develop an interactive whiteboard for collaborative design (Excalidraw integration), implement a new agent for automated training code generation with HPC support, extend the document retrieval system to additional file formats, set up experiment recording infrastructure, improve error handling, and configure a new dedicated workstation.

---


## Key Milestones

| Date | Event |
|------|-------|
| Apr 3–6, 2026 | Easter break |
| **Apr 13, 2026** | **Thesis written report deadline** |
| Apr 20–23, 2026 | HPC AI conference (tentative) |
| Apr 27, 2026 | IDETC paper notification |
| **May 8, 2026** | **Final presentation (30 min + questions)** |
| **May 19, 2026** | **IDETC final paper submission + author registration** |
| Jun 29 – Jul 7, 2026 | PASC conference + holidays |
| Aug 23–26, 2026 | IDETC conference |

Swiss public holidays (Labour Day, Ascension, Whit Monday) are accounted for in the schedule below.

---

## Phase 1: Thesis Completion + Demo Video + Code Quality (Mar 17 – Apr 13) — ~125h

| Period | Task | Hours |
|--------|------|-------|
| Mar 17–20 | Thesis polishing + YouTube video script | 25h |
| Mar 23–27 | Record and edit demo video (5–8 min, all system features) | 30h |
| Mar 30 – Apr 2 | Fix test suite + update dependencies + Python 3.12 support | 30h |
| Apr 7–10 | Start presentation slides | 30h |
| Apr 13 | Final thesis compilation and submission | 10h |

**Deliverables:** Thesis submitted. Demo video published. Test suite fully passing. Dependencies up to date. Presentation draft started.

---

## Phase 2: Presentation + Standardization + arXiv Preprint (Apr 14 – May 8) — ~110h

| Period | Task | Hours |
|--------|------|-------|
| Apr 14–17 | Finalize presentation + rename repository to EngiAI + deployment simplification (Docker profiles, secure credentials, dependency management) | 30h |
| Apr 20–23 | HPC AI conference (unavailable) | 0h |
| Apr 24 | Presentation refinement | 10h |
| Apr 27–30 | Codebase standardization (EngiBench/EngiOpt conventions, CI/CD pipelines, documentation update) + rehearsal | 30h |
| May 4–7 | arXiv preprint preparation + rehearsals + begin GUI launcher development | 30h |
| **May 8** | **Final presentation** | 10h |

**Deliverables:** Presentation delivered. arXiv preprint submitted. Codebase standardized with automated CI/CD. GUI launcher prototype with secure settings management.

---

## Phase 3: Paper Revision + Code Cleanup + System Improvements (May 9 – Jun 27) — ~110h

### 3a. IDETC Paper Revision (May 9–19) — 20h

Address reviewer feedback (if needed), prepare camera-ready version, and complete author registration by the **May 19 deadline**. Finalize the GUI launcher in parallel.

### 3b. Code Cleanup (May 19 – Jun 6) — 35h

- Remove dead code, duplicate logic, and unused dependencies
- Standardize the Prusa 3D printer integration module to lab conventions
- Extract the Excalidraw whiteboard component into a standalone reusable package

### 3c. System Improvements (Jun 8–26) — 55h

| Period | Task | Hours |
|--------|------|-------|
| Jun 8–12 | Local model support (Ollama integration in chat interface) | 15h |
| Jun 15–19 | Improved progress feedback for long-running tasks + error recovery mechanisms | 20h |
| Jun 22–26 | Experiment recording infrastructure (screen + camera capture) + Google audio model support | 20h |

*Jun 29 – Jul 7: PASC conference + holidays (unavailable)*

---

## Phase 4: New Features + Handover (Jul 8 – Aug 7) — ~105h

### 4a. Core Features (Jul 8–24) — 55h

| Period | Task | Hours |
|--------|------|-------|
| Jul 8–17 | Interactive Excalidraw whiteboard with AI agent integration (bidirectional: agent draws results, user sketches constraints) | 35h |
| Jul 20–24 | Additional file format support for document retrieval + containerized problem solver framework | 20h |

### 4b. New Agent + Handover (Jul 27 – Aug 7) — 50h

| Period | Task | Hours |
|--------|------|-------|
| Jul 27 – Aug 3 | Training code generation agent (automated code validation, HPC integration for Euler, full test suite) | 30h |
| Aug 4–5 | Repository transfer to lab GitHub organization (v2.0.0 release) | 10h |
| Aug 6–7 | New workstation setup and deployment verification | 10h |

---

## Phase 5: IDETC Conference + Wrap-Up (Aug 10–28) — ~60h

| Period | Task | Hours |
|--------|------|-------|
| Aug 10–21 | IDETC conference presentation preparation, rehearsals + travel | 30h |
| Aug 23–26 | IDETC conference | 20h |
| Aug 27–28 | Final documentation, release tagging, status report for the lab | 10h |

---

## Summary

| Phase | Period | Pace | Hours |
|-------|--------|------|-------|
| 1. Thesis + Video + Code Quality | Mar 17 – Apr 13 | Full-time | ~125h |
| 2. Presentation + Standardization | Apr 14 – May 8 | Full-time | ~110h |
| 3. Paper Revision + Cleanup + Improvements | May 9 – Jun 27 | Part-time | ~110h |
| 4. New Features + Handover | Jul 8 – Aug 7 | Part-time | ~105h |
| 5. IDETC + Wrap-Up | Aug 10 – Aug 28 | Part-time | ~60h |
| **Total** | **Mar 17 – Aug 28** | | **~510h** |

The full-time period (Phases 1–2) covers all deadline-critical work: thesis, presentation, arXiv preprint, and codebase standardization. The part-time period (Phases 3–4) is modular — tasks can be reordered or adjusted depending on the IDETC paper revision workload.

---

## Contingencies

| Scenario | Response |
|----------|----------|
| Heavy IDETC paper revision required | Defer some Phase 3b cleanup tasks to Phase 4 buffer |
| IDETC paper accepted without revision | Redirect ~30h to feature development |
| Excalidraw integration more complex than estimated | Deliver a simplified version and document the full design |

---

## Stretch Goals

If buffer hours are available:

- **Ablation studies** (prompt/temperature/model comparisons) — ~15–20h. Useful for potential future publications.

---

## Proposed Employment (tentative)

| | |
|---|---|
| **Employment period** | **~May 9 – August 28, 2026** (part-time, after thesis defense) |
| **Employment rate** | **~50%** (~2 full days + 1 half day per week, ~20h/week) |
| **Total part-time effort** | **~275 hours** across ~13 usable weeks |

Exact dates and employment rate to be discussed. Prior to the employment period, Phases 1–2 (Mar 17 – May 8) are covered under the existing full-time arrangement (~40h/week, ~235h). The part-time contract would begin after the final presentation on May 8.

**Total project effort:** ~510 hours (full-time + part-time combined).
