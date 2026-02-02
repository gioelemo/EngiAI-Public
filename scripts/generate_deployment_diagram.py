"""
Generate a simplified Docker deployment diagram for conference publication.

This shows the container structure and networking, but in a cleaner,
more publication-friendly format.
"""
# mypy: ignore-errors

import sys
from pathlib import Path

# Add benchmarks/evaluations/plots to path to import utils
plots_dir = Path(__file__).parent.parent / "benchmarks" / "evaluations" / "plots"
sys.path.insert(0, str(plots_dir))

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from utils import COLOR_PALETTE, PLOT_STYLE, setup_style  # noqa: E402

# Setup publication style
setup_style()

# Create figure (full width for deployment diagram)
fig, ax = plt.subplots(1, 1, figsize=PLOT_STYLE["figsize_full_width_tall"])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Color scheme
color_container = COLOR_PALETTE[0]  # Blue
color_db = COLOR_PALETTE[2]  # Green
color_external = "#F0F0F0"  # Light gray
color_network = "#FFF8DC"  # Cornsilk (light yellow)

# Font sizes
font_sizes = PLOT_STYLE["font_sizes"]
title_size = font_sizes["axes_title"] + 1
label_size = font_sizes["axes_label"]
small_size = font_sizes["tick_label"]

# ============================================================================
# Docker Network Box (full width)
# ============================================================================
network_box = Rectangle(
    (0.2, 3.5),
    9.6,
    5.0,
    edgecolor=COLOR_PALETTE[1],
    facecolor=color_network,
    linewidth=2,
    linestyle="--",
    alpha=0.3,
)
ax.add_patch(network_box)
ax.text(
    0.4,
    8.1,
    r"\textit{Docker Network: engineer-assistant}",
    fontsize=small_size,
    color=COLOR_PALETTE[1],
    fontweight="bold",
)

# ============================================================================
# User Browser (outside Docker)
# ============================================================================
browser_box = FancyBboxPatch(
    (3.5, 8.8),
    3,
    0.7,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="#E8F4F8",
    linewidth=1.5,
)
ax.add_patch(browser_box)
ax.text(
    5,
    9.15,
    r"\textbf{User Browser}",
    fontsize=label_size,
    ha="center",
    va="center",
)

# Arrow to Streamlit container
arrow_browser = FancyArrowPatch(
    (5, 8.8),
    (5, 7.7),
    arrowstyle="<->",
    mutation_scale=15,
    linewidth=1.5,
    color="black",
)
ax.add_patch(arrow_browser)

# ============================================================================
# Container 1: Streamlit Chatbot
# ============================================================================
chatbot_box = FancyBboxPatch(
    (2.8, 6.5),
    4.4,
    1.2,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_container,
    linewidth=1.5,
    alpha=0.7,
)
ax.add_patch(chatbot_box)

# Container name
ax.text(
    5,
    7.4,
    r"\textbf{engineer-assistant-chatbot}",
    fontsize=label_size,
    ha="center",
    va="center",
)

# Container details
ax.text(
    5,
    7.1,
    r"Streamlit UI + Multi-Agent System",
    fontsize=small_size,
    ha="center",
    va="center",
    style="italic",
)


# ============================================================================
# Container 2: PostgreSQL Database
# ============================================================================
postgres_box = FancyBboxPatch(
    (0.5, 4.0),
    2.6,
    1.0,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_db,
    linewidth=1.5,
    alpha=0.7,
)
ax.add_patch(postgres_box)

ax.text(
    1.8,
    4.7,
    r"\textbf{postgres}",
    fontsize=label_size,
    ha="center",
    va="center",
)

ax.text(
    1.8,
    4.4,
    r"PostgreSQL 15",
    fontsize=small_size,
    ha="center",
    va="center",
    style="italic",
)


# Arrow from chatbot to postgres
arrow_db = FancyArrowPatch(
    (3.8, 6.5),
    (2.5, 5.0),
    arrowstyle="<->",
    mutation_scale=12,
    linewidth=1.2,
    color="black",
    alpha=0.6,
)
ax.add_patch(arrow_db)

# ============================================================================
# Container 3: Prusa MCP Server (Optional)
# ============================================================================
prusa_box = FancyBboxPatch(
    (3.7, 4.0),
    2.6,
    1.0,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_container,
    linewidth=1.5,
    alpha=0.7,
)
ax.add_patch(prusa_box)

ax.text(
    5.0,
    4.7,
    r"\textbf{prusa-mcp-server}",
    fontsize=label_size,
    ha="center",
    va="center",
)

ax.text(
    5.0,
    4.4,
    r"MCP Tools (Optional)",
    fontsize=small_size,
    ha="center",
    va="center",
    style="italic",
)


# Arrow from chatbot to prusa
arrow_prusa = FancyArrowPatch(
    (5.0, 6.5),
    (5.0, 5.0),
    arrowstyle="<->",
    mutation_scale=12,
    linewidth=1.2,
    color="black",
    alpha=0.6,
)
ax.add_patch(arrow_prusa)

# ============================================================================
# Container 4: MMORE RAG Service (inside Docker)
# ============================================================================
mmore_box = FancyBboxPatch(
    (6.9, 4.0),
    2.6,
    1.0,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_container,
    linewidth=1.5,
    alpha=0.7,
)
ax.add_patch(mmore_box)

ax.text(
    8.2,
    4.7,
    r"\textbf{mmore-rag-service}",
    fontsize=label_size,
    ha="center",
    va="center",
)

ax.text(
    8.2,
    4.4,
    r"Multimodal RAG",
    fontsize=small_size,
    ha="center",
    va="center",
    style="italic",
)

# Arrow from chatbot to MMORE
arrow_mmore = FancyArrowPatch(
    (6.4, 6.5),
    (7.5, 5.0),
    arrowstyle="<->",
    mutation_scale=12,
    linewidth=1.2,
    color="black",
    alpha=0.6,
)
ax.add_patch(arrow_mmore)

# ============================================================================
# External Services (outside Docker network)
# ============================================================================

# Unified External Services (APIs + HPC) - moved up closer to legend
external_box = FancyBboxPatch(
    (2.5, 2.5),
    5.0,
    0.8,
    boxstyle="round,pad=0.05",
    edgecolor="gray",
    facecolor=color_external,
    linewidth=1,
    linestyle="--",
)
ax.add_patch(external_box)

ax.text(
    5.0,
    3.1,
    r"\textbf{External Services}",
    fontsize=label_size,
    ha="center",
    va="center",
)

ax.text(
    5.0,
    2.75,
    r"OpenAI, Tavily, ArXiv, HPC Cluster",
    fontsize=small_size - 1,
    ha="center",
    va="center",
    style="italic",
)

# Arrow to External Services - routed to avoid containers
# Use a curved arrow from bottom of chatbot
arrow_external = FancyArrowPatch(
    (4.2, 6.5),
    (4.5, 3.3),
    arrowstyle="->",
    mutation_scale=12,
    linewidth=1.2,
    color="black",
    alpha=0.6,
    connectionstyle="arc3,rad=0.2",
)
ax.add_patch(arrow_external)


# ============================================================================
# Legend
# ============================================================================
legend_elements = [
    mpatches.Patch(
        facecolor=color_container, edgecolor="black", label="Container", alpha=0.7
    ),
    mpatches.Patch(facecolor=color_db, edgecolor="black", label="Database", alpha=0.7),
    mpatches.Patch(
        facecolor=color_external,
        edgecolor="gray",
        label="External Service",
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
output_png = output_dir / "deployment_infrastructure.png"
output_pdf = output_dir / "deployment_infrastructure.pdf"

fig.savefig(output_png, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")

print("✅ Deployment infrastructure diagram saved:")
print(f"   PNG: {output_png}")
print(f"   PDF: {output_pdf}")

plt.close()
