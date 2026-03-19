# EngiAI Demo Video Plan

## Context
Create a ~3 minute demo video showcasing EngiAI's multi-agent chatbot for mechanical engineering. The user sends **3 separate messages** to show the conversational, step-by-step nature of the system, then opens the result in **PrusaSlicer** to show it's print-ready. Uses OBS on macOS with Stream Deck. Voiceover + text overlays. 3 OBS scenes: Home, Chat (Chrome), PrusaSlicer.

---

## Pre-Recording Setup

### OBS Configuration
- **Resolution**: 1920×1080 (or retina-scaled equivalent)
- **Streamlit**: Run `make run-ui` → fullscreen Chrome on `localhost:8501`, hide bookmarks bar, clean Chrome profile
- **Zoom**: Browser zoom so chat text is readable at 1080p (~80% of screen width)
- **Scene 1 — "Home"**: EngiAI home page capture
- **Scene 2 — "Chat"**: Chat interface capture (Chrome window)
- **Scene 3 — "PrusaSlicer"**: PrusaSlicer window capture (add after recording starts, or pre-configure with a placeholder window)

**Window management**: Use macOS Spaces or split-screen. Keep Chrome on Space 1 and PrusaSlicer on Space 2. Alternatively, use OBS "Window Capture" sources — one for Chrome, one for PrusaSlicer — and switch scenes via Stream Deck. This avoids visible desktop/dock when switching.

### Stream Deck Buttons
1. **Start recording** (OBS)
2. **Scene: Home** — switch to home page scene
3. **Scene: Chat** — switch to chat scene
4. **Scene: PrusaSlicer** — switch to PrusaSlicer scene
5. **Add marker** — press before/after every agent-thinking section. Markers = "speed up between these"
6. **Stop recording**

**Tip**: Record one continuous take. Never stop/restart — just add markers. Edit in post.

### Environment Prep
- MMORE RAG running on port 8001, EngiBench paper indexed (`file_id='engibench_paper'`)
- Clean chat session (no prior messages)
- macOS Focus mode ON (no notifications)
- Hide Dock, neutral wallpaper
- **Do a full dry run first** to confirm everything works and get a feel for timing

---

## Video Structure & Script

---

### SCENE 1 — Home Page (0:00–0:08)

**On screen**: EngiAI home page — agent grid visible (Engineering, RAG, Search, ArXiv, Prusa, HPC, CLI, Supervisor)

**Voiceover**:
> "EngiAI is a multi-agent assistant for mechanical engineering — it coordinates specialized agents for optimization, research, 3D printing, and more."

**Text overlay**: `EngiAI — Multi-Agent Engineering Assistant`

**Action**: Click "Chat" in sidebar after ~5 seconds.

---

### SCENE 2 — Message 1: RAG Query (0:08–0:50)

**On screen**: Clean empty chat

**Text overlay**: `Step 1 — Find parameters from a research paper`

**Type this prompt** (naturally, ~3-4 seconds of typing):
```
Search the EngiBench paper for the Beams2D API code example that sets volfrac and forcedist conditions.
```

This mirrors the benchmark prompt style — it references the specific API example in the paper rather than asking generically. The expected answer is `volfrac=0.7, forcedist=0.3` (the non-default values from the paper's code snippet `desired_conds = {"volfrac": 0.7, "forcedist": 0.3}`).

**Press Enter.**

**Voiceover** (while agent thinks):
> "We ask the agent to find specific optimization parameters from a research paper's API example — the paper has been indexed in our RAG system."

- ⏩ **Speed up** the thinking spinner (2-4×). Add marker before/after.
- **Voiceover** (when RAG results appear, return to 1×):
> "The RAG agent searches the paper and finds the exact values from the code example — volume fraction 0.7 and force distribution 0.3."

**Text overlay** (brief, on the response): `Found: volfrac = 0.7, forcedist = 0.3`

Let the response sit for ~3 seconds so viewers can read it.

---

### SCENE 3 — Message 2: Optimize (0:50–1:40)

**Text overlay**: `Step 2 — Run topology optimization`

**Type this prompt**:
```
Now optimize a 2D beam with those parameters and default values for the rest.
```

**Press Enter.**

**Voiceover**:
> "Now we ask the engineering agent to run the topology optimization using the values it just found."

- ⏩ **Speed up** the optimization phase (4-8×). This is the longest computation (~30-60s real time). Add markers.
- **Text overlay** during speed-up: `Optimizing... ⏩`

**When the result appears** (topology image + compliance from simulation):
- Return to **1× speed**
- **Voiceover**:
> "The optimization converges — we can see the material distributed along the main load paths."

- Let the 2D topology image sit for ~3-4 seconds
- If compliance is mentioned: "The compliance value tells us how stiff the structure is."

---

### SCENE 4 — Message 3: STL Export (1:40–2:20)

**Text overlay**: `Step 3 — Export to 3D-printable STL`

**Type this prompt**:
```
Export the design as an STL file for 3D printing.
```

**Press Enter.**

**Voiceover**:
> "Finally, we export the 2D result into a 3D-printable STL file."

- ⏩ **Speed up** briefly (2×) during conversion. Add markers.

**Key moment — 3D viewer appears**:
- Return to **1× speed**
- **Mouse**: Slowly rotate the 3D model to show the extruded beam from different angles (~4-5 seconds)
- **Voiceover**:
> "The interactive 3D viewer shows the extruded design — ready to send to a printer."

- **Text overlay**: `Interactive 3D model — ready to print`

---

### SCENE 5 — PrusaSlicer (2:20–2:50)

**Preparation**: Have PrusaSlicer already open (empty or with a default plate). After the STL export, you know the file is at `outputs/beams2d_design_exported_<timestamp>.stl`.

**Action sequence**:
1. Switch to PrusaSlicer scene (Stream Deck button)
2. Drag-and-drop the STL file from Finder into PrusaSlicer, or use File → Import → STL and select the exported file
   - **Alternative**: If you want a cleaner transition, pre-open PrusaSlicer and use `open -a PrusaSlicer outputs/beams2d_design_exported_*.stl` in terminal before switching scenes
3. The beam model appears on the print bed

**Voiceover**:
> "And here's the exported model in PrusaSlicer — ready to slice and send to the printer."

- **Mouse**: Rotate the view slightly, maybe click "Slice now" to show the slicing preview (layers view is visually impressive)
- **Text overlay**: `Ready to print`

**Duration**: ~30 seconds (10s to load + 20s to show the model and optionally slice)

---

### SCENE 6 — Closing (2:50–3:05)

**On screen**: Either stay on PrusaSlicer sliced view, or cut back to the chat to show the full 3-message conversation

**Voiceover**:
> "Three messages — from research paper to printable design. That's EngiAI."

**Text overlay / end card**:
- "EngiAI" (logo)
- `Paper → Parameters → Optimization → 3D Print`
- Optional: GitHub link, your name, thesis context

**Fade to black.**

---

## The 3 Prompts (Summary)

| # | Prompt | What happens |
|---|--------|-------------|
| 1 | "In the EngiBench paper's Section 3.1 API walkthrough, a code example runs a Beams2D optimization using non-default design conditions. Search the paper to find both the volume fraction and force distribution from that example." | RAG agent searches indexed paper, finds volfrac=0.7 & forcedist=0.3 |
| 2 | "Now optimize a 2D beam with those parameters." | Engineering agent runs topology optimization + simulation with the RAG-retrieved values |
| 3 | "Export the design as an STL file for 3D printing." | Engineering agent exports STL, 3D viewer appears |

---

## Post-Production

### Editing (iMovie / DaVinci Resolve / Final Cut)
1. **Speed ramps**: Use OBS markers to locate thinking sections → speed up 2-8×. Add a small "⏩ 4×" indicator in the corner
2. **Jump cuts**: If any thinking section is very long (>10s even at 8×), hard-cut it out entirely
3. **Text overlays**: Clean sans-serif font (SF Pro / Inter). White text + dark shadow. Step labels appear at the start of each section
4. **Transitions**: Simple cuts, max 0.2s cross-dissolve
5. **Voiceover**: Record separately in a quiet room, sync in post. Keep it natural and conversational
6. **Music**: Optional subtle background track, very low volume
7. **Zoom**: If the chat text is small, add a gentle zoom-in crop on key responses

### Voiceover Recording Tips
- Record all narration in one session after the screen recording is done
- Watch your edited screen recording and narrate along with it
- Keep sentences short and declarative — avoid filler words
- Pause between scenes so you have clean edit points

---

## Timing Summary

| Scene | Duration | Speed | Content |
|-------|----------|-------|---------|
| Home page | 8s | 1× | Agent grid overview |
| Message 1 (RAG) | 42s | Mixed | Type → think (⏩) → RAG results (1×) |
| Message 2 (Optimize) | 50s | Mixed | Type → optimize (⏩) → topology result (1×) |
| Message 3 (STL) | 40s | Mixed | Type → export (⏩) → 3D viewer (1×) |
| PrusaSlicer | 30s | 1× | Import STL, show on print bed, optionally slice |
| Closing | 15s | 1× | End card, fade to black |
| **Total** | **~3:05** | | |

---

## Fallback (if MMORE is down)

Skip the RAG step. Use 2 messages instead:

| # | Prompt |
|---|--------|
| 1 | "Design a 2D beam with volume fraction 0.35 and force distribution 0.5. Optimize it and simulate the result." |
| 2 | "Export the design as an STL file for 3D printing." |

Adjust voiceover to skip the paper-search narrative.
