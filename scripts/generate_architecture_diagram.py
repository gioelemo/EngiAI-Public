"""
Generate a simplified, publication-ready architecture diagram.

This creates a clean diagram suitable for conference publications, focusing on
the core multi-agent system without Docker/deployment details.
"""
# mypy: ignore-errors

import sys
from pathlib import Path

# Add benchmarks/evaluations/plots to path to import utils
plots_dir = Path(__file__).parent.parent / "benchmarks" / "evaluations" / "plots"
sys.path.insert(0, str(plots_dir))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from utils import COLOR_PALETTE, PLOT_STYLE, setup_style  # noqa: E402

# Setup publication style
setup_style()

# Create figure with appropriate size for conference paper (full width)
fig, ax = plt.subplots(1, 1, figsize=PLOT_STYLE["figsize_full_width_tall"])
ax.set_xlim(0.3, 9.7)
ax.set_ylim(3.05, 9.05)
ax.axis("off")

# Color scheme — all from COLOR_PALETTE (Okabe-Ito) for consistency with benchmark plots
color_user = COLOR_PALETTE[4]  # Sky blue
color_supervisor = COLOR_PALETTE[1]  # Orange
color_agent = COLOR_PALETTE[0]  # Blue
color_external = COLOR_PALETTE[2]  # Green

# Shared style constants from PLOT_STYLE
box_alpha = 0.4  # lighter fills for text readability
box_linewidth = 1.0  # matches lines.linewidth in setup_style
arrow_linewidth = 1.0
arrow_alpha = 0.7

# Font sizes (using utils style, bumped +1 for readability)
font_sizes = PLOT_STYLE["font_sizes"]
label_size = font_sizes["axes_label"] + 1
small_size = font_sizes["tick_label"] + 1

# ============================================================================
# Uniform vertical layout — all boxes same height, equal gaps between layers
# ============================================================================
box_h = 0.7
top_y = 9.0  # top of User box
bot_y = 3.2  # bottom of Tools box
n_layers = 4
layer_gap = (top_y - bot_y - n_layers * box_h) / (n_layers - 1)

user_y = top_y - box_h
supervisor_y = user_y - layer_gap - box_h
agent_y = supervisor_y - layer_gap - box_h
tools_y = agent_y - layer_gap - box_h

# ============================================================================
# Layer 1: User
# ============================================================================
user_box = FancyBboxPatch(
    (3.5, user_y),
    3,
    box_h,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_user,
    linewidth=box_linewidth,
    alpha=box_alpha,
)
ax.add_patch(user_box)
ax.text(
    5,
    user_y + box_h / 2,
    r"\textbf{User}",
    fontsize=label_size,
    ha="center",
    va="center",
)

# Arrow from user to supervisor
arrow_mid = (user_y + supervisor_y + box_h) / 2
arrow_user = FancyArrowPatch(
    (5.3, user_y),
    (5.3, supervisor_y + box_h),
    arrowstyle="->",
    mutation_scale=15,
    linewidth=arrow_linewidth,
    color="black",
    alpha=arrow_alpha,
)
ax.add_patch(arrow_user)
ax.text(5.5, arrow_mid, r"\textbf{Query}", fontsize=small_size, va="center", ha="left")

# Arrow from supervisor back to user (response)
arrow_response = FancyArrowPatch(
    (4.7, supervisor_y + box_h),
    (4.7, user_y),
    arrowstyle="->",
    mutation_scale=15,
    linewidth=arrow_linewidth,
    color="black",
    alpha=arrow_alpha,
)
ax.add_patch(arrow_response)
ax.text(
    4.5, arrow_mid, r"\textbf{Response}", fontsize=small_size, va="center", ha="right"
)

# ============================================================================
# Layer 2: Supervisor Agent (LLM)
# ============================================================================
supervisor_box = FancyBboxPatch(
    (0.5, supervisor_y),
    9,
    box_h,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_supervisor,
    linewidth=box_linewidth,
    alpha=box_alpha,
)
ax.add_patch(supervisor_box)
ax.text(
    5,
    supervisor_y + box_h / 2,
    r"\textbf{Supervisor Agent (LLM)}",
    fontsize=label_size,
    ha="center",
    va="center",
)

# ============================================================================
# Layer 3: Specialized Agents (name only — no tool sublabels)
# ============================================================================
agent_height = box_h
agent_width = 1.15
n_agents = 7
agent_x_start = 0.5
agent_available = 9.5 - agent_x_start
agent_spacing = (agent_available - n_agents * agent_width) / (n_agents - 1)

agents = [
    {"name": "Engineering"},
    {"name": "RAG"},
    {"name": "Search"},
    {"name": "ArXiv"},
    {"name": "HPC"},
    {"name": "CLI"},
    {"name": "Prusa"},
]

# Compute x positions
for i, agent in enumerate(agents):
    agent["x"] = agent_x_start + i * (agent_width + agent_spacing)

# Draw agents
for agent in agents:
    agent_box = FancyBboxPatch(
        (agent["x"], agent_y),
        agent_width,
        agent_height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=color_agent,
        linewidth=box_linewidth,
        alpha=box_alpha,
    )
    ax.add_patch(agent_box)

    # Agent name — centered in box
    ax.text(
        agent["x"] + agent_width / 2,
        agent_y + agent_height / 2,
        r"\textbf{" + agent["name"] + "}",
        fontsize=small_size,
        ha="center",
        va="center",
    )

    # Bidirectional arrow from supervisor to agent
    cx = agent["x"] + agent_width / 2
    arrow_super = FancyArrowPatch(
        (cx, supervisor_y),
        (cx, agent_y + agent_height),
        arrowstyle="<->",
        mutation_scale=12,
        linewidth=arrow_linewidth,
        color="black",
        alpha=arrow_alpha,
    )
    ax.add_patch(arrow_super)

# ============================================================================
# Layer 4: External Services / Tools (single source of truth for tool names)
# ============================================================================
tools_height = box_h  # same as specialization boxes
tools_width = agent_width  # same as specialization boxes

# Each tool is positioned to align under its corresponding agent(s).
# "links" maps each tool to the agent indices that connect to it.
tools = [
    {"name": "EngiBench\n+EngiOpt", "align": [0], "links": [0]},
    {"name": "MMORE\nRAG", "align": [1], "links": [1]},
    {"name": "Web\nAPIs", "align": [2, 3], "links": [2, 3]},  # shared by Search & ArXiv
    {"name": "HPC\nCluster", "align": [4], "links": [4]},
    {"name": "Local\nShell", "align": [5], "links": [5]},
    {"name": "Prusa\nMCP", "align": [6], "links": [6]},
]

# Compute x position: center tool under the midpoint of its aligned agents
for tool in tools:
    xs = [agents[i]["x"] + agent_width / 2 for i in tool["align"]]
    tool["x"] = sum(xs) / len(xs) - tools_width / 2

# Draw tool boxes and connecting arrows
for tool in tools:
    tool_box = FancyBboxPatch(
        (tool["x"], tools_y),
        tools_width,
        tools_height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=color_external,
        linewidth=box_linewidth,
        alpha=box_alpha,
    )
    ax.add_patch(tool_box)

    # Bold text, same style as agent boxes
    # Wrap each line in \textbf{}
    bold_name = "\n".join(r"\textbf{" + line + "}" for line in tool["name"].split("\n"))
    ax.text(
        tool["x"] + tools_width / 2,
        tools_y + tools_height / 2,
        bold_name,
        fontsize=small_size,
        ha="center",
        va="center",
    )

    # Arrows from each linked agent down to this tool
    tool_cx = tool["x"] + tools_width / 2
    for idx in tool["links"]:
        agent_cx = agents[idx]["x"] + agent_width / 2
        ax.annotate(
            "",
            xy=(tool_cx, tools_y + tools_height),
            xytext=(agent_cx, agent_y),
            arrowprops={
                "arrowstyle": "->",
                "lw": arrow_linewidth,
                "color": "black",
                "alpha": arrow_alpha,
            },
        )

# ============================================================================
# Layer labels (on the left side)
# ============================================================================
# Layer labels removed — described in figure caption instead


# ============================================================================
# Save figure
# ============================================================================
plt.tight_layout()

# Create output directory in assets/
output_dir = Path(__file__).parent.parent / "assets"
output_dir.mkdir(parents=True, exist_ok=True)

# Save both PNG and PDF
output_png = output_dir / "architecture_simplified.png"
output_pdf = output_dir / "architecture_simplified.pdf"

fig.savefig(
    output_png,
    dpi=PLOT_STYLE["dpi"],
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white",
)
fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.05, facecolor="white")

print("Simplified architecture diagram saved:")
print(f"   PNG: {output_png}")
print(f"   PDF: {output_pdf}")

plt.close()
