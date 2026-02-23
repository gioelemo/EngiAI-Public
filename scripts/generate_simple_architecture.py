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

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from utils import COLOR_PALETTE, PLOT_STYLE, setup_style  # noqa: E402

# Setup publication style
setup_style()

# Create figure with appropriate size for conference paper (full width)
fig, ax = plt.subplots(1, 1, figsize=PLOT_STYLE["figsize_full_width_tall"])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Color scheme (using same palette as plots for consistency)
color_user = "#E8F4F8"  # Light blue
color_supervisor = COLOR_PALETTE[1]  # Orange
color_agent = COLOR_PALETTE[0]  # Blue
color_tool = COLOR_PALETTE[2]  # Green
color_external = "#F0F0F0"  # Light gray

# Font sizes (using utils style)
font_sizes = PLOT_STYLE["font_sizes"]
title_size = font_sizes["axes_title"] + 1
label_size = font_sizes["axes_label"]
small_size = font_sizes["tick_label"]

# ============================================================================
# Layer 1: User
# ============================================================================
user_box = FancyBboxPatch(
    (3.5, 8.5),
    3,
    0.8,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_user,
    linewidth=1.5,
)
ax.add_patch(user_box)
ax.text(
    5,
    8.9,
    r"\textbf{User}",
    fontsize=label_size,
    ha="center",
    va="center",
)

# Arrow from user to supervisor
arrow_user = FancyArrowPatch(
    (5.3, 8.5),
    (5.3, 7.8),
    arrowstyle="->",
    mutation_scale=15,
    linewidth=1.5,
    color="black",
)
ax.add_patch(arrow_user)
ax.text(5.8, 8.15, r"\textit{Query}", fontsize=small_size, va="center")

# ============================================================================
# Layer 2: Supervisor Agent (LLM)
# ============================================================================
supervisor_box = FancyBboxPatch(
    (0.5, 6.8),
    9,
    1.0,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_supervisor,
    linewidth=1.5,
    alpha=0.7,
)
ax.add_patch(supervisor_box)
ax.text(
    5,
    7.3,
    r"\textbf{Supervisor Agent (LLM)}",
    fontsize=label_size,
    ha="center",
    va="center",
)

# Arrow from supervisor back to user (response)
arrow_response = FancyArrowPatch(
    (4.7, 7.8),
    (4.7, 8.5),
    arrowstyle="->",
    mutation_scale=15,
    linewidth=1.5,
    color="black",
)
ax.add_patch(arrow_response)
ax.text(4.0, 8.15, r"\textit{Response}", fontsize=small_size, va="center", ha="right")

# ============================================================================
# Layer 3: Specialized Agents
# ============================================================================
agent_y = 4.5
agent_height = 1.2
agent_width = 1.0  # Reduced to fit 7 agents with proper spacing
# Calculate spacing to distribute evenly under supervisor (0.5 to 9.5)
# Total agent width: 7 * 1.0 = 7.0, available space: 9, gaps: 6
agent_spacing = (9 - 7 * agent_width) / 6  # ~0.333

agents = [
    {"name": "Engineering", "x": 0.5, "tools": ["EngiBench", "EngiOpt"]},
    {"name": "RAG", "x": 0.5 + 1 * (agent_width + agent_spacing), "tools": ["MMORE"]},
    {
        "name": "Search",
        "x": 0.5 + 2 * (agent_width + agent_spacing),
        "tools": ["Tavily"],
    },
    {
        "name": "ArXiv",
        "x": 0.5 + 3 * (agent_width + agent_spacing),
        "tools": ["ArXiv API"],
    },
    {"name": "HPC", "x": 0.5 + 4 * (agent_width + agent_spacing), "tools": ["SLURM"]},
    {"name": "CLI", "x": 0.5 + 5 * (agent_width + agent_spacing), "tools": ["Shell"]},
    {"name": "Prusa", "x": 0.5 + 6 * (agent_width + agent_spacing), "tools": ["MCP"]},
]

# Draw agents
for agent in agents:
    agent_box = FancyBboxPatch(
        (agent["x"], agent_y),
        agent_width,
        agent_height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=color_agent,
        linewidth=1,
        alpha=0.7,
    )
    ax.add_patch(agent_box)

    # Agent name
    ax.text(
        agent["x"] + agent_width / 2,
        agent_y + agent_height - 0.3,
        r"\textbf{" + agent["name"] + "}",
        fontsize=small_size,
        ha="center",
        va="center",
    )

    # Tools (smaller text)
    for i, tool in enumerate(agent["tools"]):
        ax.text(
            agent["x"] + agent_width / 2,
            agent_y + agent_height - 0.6 - i * 0.25,
            tool,
            fontsize=small_size - 1,
            ha="center",
            va="center",
            style="italic",
        )

    # Arrow from supervisor to agent
    supervisor_x = agent["x"] + agent_width / 2
    arrow_super = FancyArrowPatch(
        (supervisor_x, 6.8),
        (supervisor_x, agent_y + agent_height),
        arrowstyle="<->",
        mutation_scale=12,
        linewidth=1,
        color="black",
        alpha=0.6,
    )
    ax.add_patch(arrow_super)

# ============================================================================
# Layer 4: External Services / Tools
# ============================================================================
tools_y = 2.5
tools_height = 0.8
tools_width = 1.2  # Reduced to fit 6 tools without overlapping

# Position tools to align with their corresponding agents
tools = [
    {
        "name": "EngiBench\nLibrary",
        "x": 0.5 + agent_width / 2 - tools_width / 2,
    },  # Under Engineering
    {
        "name": "MMORE\nRAG Service",
        "x": agents[1]["x"] + agent_width / 2 - tools_width / 2,
    },  # Under RAG
    {
        "name": "Web\nAPIs",
        "x": (agents[2]["x"] + agents[3]["x"] + agent_width) / 2 - tools_width / 2,
    },  # Between Search & ArXiv
    {
        "name": "HPC\nCluster",
        "x": agents[4]["x"] + agent_width / 2 - tools_width / 2,
    },  # Under HPC
    {
        "name": "Local\nShell",
        "x": agents[5]["x"] + agent_width / 2 - tools_width / 2,
    },  # Under CLI
    {
        "name": "Prusa\nMCP",
        "x": agents[6]["x"] + agent_width / 2 - tools_width / 2,
    },  # Under Prusa
]

for tool in tools:
    tool_box = FancyBboxPatch(
        (tool["x"], tools_y),
        tools_width,
        tools_height,
        boxstyle="round,pad=0.05",
        edgecolor="gray",
        facecolor=color_external,
        linewidth=1,
        linestyle="--",
    )
    ax.add_patch(tool_box)

    ax.text(
        tool["x"] + tools_width / 2,
        tools_y + tools_height / 2,
        tool["name"],
        fontsize=small_size,
        ha="center",
        va="center",
    )

# ============================================================================
# Connecting arrows from agents to tools
# ============================================================================
# Engineering -> EngiBench
tool_0_x = tools[0]["x"] + tools_width / 2
ax.annotate(
    "",
    xy=(tool_0_x, tools_y + tools_height),
    xytext=(agents[0]["x"] + agent_width / 2, agent_y),
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black", "alpha": 0.6},
)

# RAG -> MMORE
tool_1_x = tools[1]["x"] + tools_width / 2
ax.annotate(
    "",
    xy=(tool_1_x, tools_y + tools_height),
    xytext=(agents[1]["x"] + agent_width / 2, agent_y),
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black", "alpha": 0.6},
)

# Search/ArXiv -> Web APIs
tool_2_x = tools[2]["x"] + tools_width / 2
ax.annotate(
    "",
    xy=(tool_2_x, tools_y + tools_height),
    xytext=(agents[2]["x"] + agent_width / 2, agent_y),
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black", "alpha": 0.6},
)
ax.annotate(
    "",
    xy=(tool_2_x, tools_y + tools_height),
    xytext=(agents[3]["x"] + agent_width / 2, agent_y),
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black", "alpha": 0.6},
)

# HPC -> HPC Cluster
tool_3_x = tools[3]["x"] + tools_width / 2
ax.annotate(
    "",
    xy=(tool_3_x, tools_y + tools_height),
    xytext=(agents[4]["x"] + agent_width / 2, agent_y),
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black", "alpha": 0.6},
)

# CLI -> Local Shell
tool_4_x = tools[4]["x"] + tools_width / 2
ax.annotate(
    "",
    xy=(tool_4_x, tools_y + tools_height),
    xytext=(agents[5]["x"] + agent_width / 2, agent_y),
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black", "alpha": 0.6},
)

# Prusa -> Prusa MCP
tool_5_x = tools[5]["x"] + tools_width / 2
ax.annotate(
    "",
    xy=(tool_5_x, tools_y + tools_height),
    xytext=(agents[6]["x"] + agent_width / 2, agent_y),
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black", "alpha": 0.6},
)

# ============================================================================
# Layer labels (on the left side)
# ============================================================================
ax.text(
    -0.3,
    8.9,
    r"\textit{Interface}",
    fontsize=small_size,
    ha="right",
    va="center",
    color="black",
)
ax.text(
    -0.3,
    7.2,
    r"\textit{Orchestration}",
    fontsize=small_size,
    ha="right",
    va="center",
    color="black",
)
ax.text(
    -0.3,
    agent_y + agent_height / 2,
    r"\textit{Specialization}",
    fontsize=small_size,
    ha="right",
    va="center",
    color="black",
)
ax.text(
    -0.3,
    tools_y + tools_height / 2,
    r"\textit{Execution}",
    fontsize=small_size,
    ha="right",
    va="center",
    color="black",
)

# ============================================================================
# Legend at bottom
# ============================================================================
legend_elements = [
    mpatches.Patch(
        facecolor=color_supervisor,
        edgecolor="black",
        label="Supervisor (Orchestration)",
        alpha=0.7,
    ),
    mpatches.Patch(
        facecolor=color_agent, edgecolor="black", label="Domain Agent (LLM)", alpha=0.7
    ),
    mpatches.Patch(
        facecolor=color_external,
        edgecolor="gray",
        label="External Service/Tool",
        linestyle="--",
    ),
]

legend = ax.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    frameon=True,
    fontsize=small_size,
    bbox_to_anchor=(0.5, -0.05),
)
legend.get_frame().set_linewidth(0.5)

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

fig.savefig(output_png, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")

print("✅ Simplified architecture diagram saved:")
print(f"   PNG: {output_png}")
print(f"   PDF: {output_pdf}")

plt.close()
