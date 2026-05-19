---
name: refresh-claude-md
description: Audit CLAUDE.md for drift against the current codebase. Re-reads CLAUDE.md, cross-checks every concrete claim (file paths, directory structure, commands, environment variables, architectural descriptions) against the current repo state and recent git history, and reports stale sections as a confirm-before-edit punch list. Use when CLAUDE.md feels outdated, after large refactors, before onboarding a new teammate, or when the user explicitly asks to "refresh CLAUDE.md", "check CLAUDE.md", or "audit project instructions".
allowed-tools: Read, Glob, Grep, Bash(git log:*), Bash(git diff:*), Bash(git show:*), Bash(ls:*), Edit
---

# Refresh CLAUDE.md

Audit [CLAUDE.md](../../../CLAUDE.md) for drift against the current state of the codebase. **Propose** edits — never apply them without explicit user confirmation. CLAUDE.md contains hand-curated nuance (design rationale, "why it breaks LLMs" commentary, worked examples) that is more valuable than anything auto-regeneration could produce, so this skill's job is to catch *factual drift*, not to rewrite prose.

## Process

Run these steps in order. Batch independent read/grep operations in parallel where possible.

### Step 1 — Load CLAUDE.md and extract concrete claims

Read [CLAUDE.md](../../../CLAUDE.md) in full. As you read, build a mental list of every **verifiable claim**, categorized:

- **Path claims** — specific files or directories referenced (e.g., `src/ui/components/excalidraw/`, `benchmarks/shared/problem_registry.py`)
- **Command claims** — shell commands, make targets, CLI invocations (e.g., `make run-ui`, `pytest -m "not slow"`)
- **Environment variable claims** — env vars the project reads (e.g., `SKIP_MCP`, `MMORE_RAG_URL`)
- **Symbol claims** — classes, functions, or module names mentioned by name (e.g., `SupervisorAgent`, `GenericFakeChatModel`)
- **Architectural claims** — prose describing how components relate (e.g., "SupervisorAgent routes requests to specialized agents", "uses LangGraph state machines")
- **Parameter/config claims** — numeric ranges, defaults, tolerances (e.g., "±0.05 float tolerance", "`scale_xy`: 0.5-5.0")

Ignore pure commentary, rationale, and hand-written examples — those aren't "facts" that can drift.

### Step 2 — Verify claims against the current repo

For each category, run the cheapest check that confirms or refutes the claim. Do not do exhaustive code reading — target the claim.

| Claim type | How to verify |
|---|---|
| Path (file) | `Glob` for the exact path. Must exist. |
| Path (directory) | `Bash(ls -la <dir>)` or `Glob` with `**` inside it. |
| Command / make target | `Grep` in `Makefile`, `package.json`, `pyproject.toml`. |
| Environment variable | `Grep` for the var name across `src/` and config files. |
| Symbol (class/function) | `Grep` for `class <Name>` / `def <Name>`. |
| Architectural prose | Spot-check by reading the named entry file (e.g., if it says "SupervisorAgent routes to X/Y/Z agents", verify those agents still exist). |
| Numeric parameter | `Grep` for the constant / config key in the source file it belongs to. |

**Parallelize aggressively.** Dispatch all the grep/glob verifications in a single message with multiple tool calls — these are independent and read-only.

### Step 3 — Consult recent git history for context

Run `git log --since="3 months ago" --name-only --pretty=format:"%h %s"` to see recently touched files. Any CLAUDE.md section describing code in a frequently-changed file is a higher suspicion target — even if the claim still holds, flag it for manual review.

Also run `git log -- CLAUDE.md` to see when CLAUDE.md itself was last updated. Claims from before the last major refactor of a subsystem are suspect.

### Step 4 — Produce a drift report

Output a markdown report with this exact structure. Do **not** edit CLAUDE.md yet.

```
## CLAUDE.md drift audit

### 🔴 Confirmed stale (N)
For each: quote the CLAUDE.md line(s), explain what's wrong, propose the replacement text.

### 🟡 Suspect (N)
For each: quote the CLAUDE.md line(s), explain why it's suspect (e.g., "file was refactored in commit abc123", "symbol exists but signature changed"), ask the user to confirm.

### 🟢 Verified (N)
One-line summary — just a count and category breakdown. No per-claim detail.

### ⚪ Not checked (N)
Anything skipped because it was pure prose/rationale, or because verification would require running code. List briefly.
```

**Tone rules for the report:**
- Be specific. "Stale path" is useless; `src/ui/components/excalidraw/ — directory does not exist; submodule was removed in commit <hash>` is useful.
- Quote CLAUDE.md with line numbers when possible.
- For each 🔴 item, write the *exact replacement text*, not just "update this."
- Keep 🟢 to a count + categories; do not list every verified claim.

### Step 5 — Ask before editing

After presenting the report, stop and ask the user which items to apply. Options to offer:

1. Apply all 🔴 items as-proposed
2. Apply specific items by number
3. Skip editing, user will handle manually
4. Drill into a 🟡 item for deeper investigation

Only use `Edit` on CLAUDE.md after the user picks items. For each edit, use the exact replacement text from the report — do not reword on the fly.

## Known drift hotspots in this repo

Use these as seed suspicions on the first pass; they are places CLAUDE.md has historically drifted:

- **Submodule paths** in the "Core Components" tree — check `.gitmodules` against any submodule references in CLAUDE.md.
- **Agent list** in the Multi-Agent System section — check [src/agents/](../../../src/agents/) for added/removed agents.
- **Prompt style list** in the Benchmarks section — check [benchmarks/problems/beams2d/generate_prompts.py](../../../benchmarks/problems/beams2d/generate_prompts.py) for the actual `--style` choices.
- **Environment variables** — check [.env.example](../../../.env.example) against the "Configuration" section.
- **Make targets** — check [Makefile](../../../Makefile) against any `make X` commands mentioned.

## What this skill deliberately does NOT do

- Does not rewrite prose, rationale, or hand-written examples — those are the *value* of CLAUDE.md.
- Does not apply edits without explicit per-item confirmation.
- Does not add new sections. If the codebase has grown a new subsystem CLAUDE.md doesn't mention, flag it in the report under a separate "Suggested additions" section, but do not draft the content.
- Does not touch memory files under `~/.claude/projects/.../memory/` — those are a separate system.
