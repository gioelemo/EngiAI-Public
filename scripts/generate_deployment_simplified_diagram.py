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

# Create figure with appropriate size for conference paper (full width)
fig, ax = plt.subplots(1, 1, figsize=PLOT_STYLE["figsize_full_width_tall"])
ax.set_xlim(0, 10)
ax.set_ylim(2.2, 9.6)
ax.axis("off")

# Color scheme — all from COLOR_PALETTE (Okabe-Ito) for consistency with benchmark plots
color_user = COLOR_PALETTE[4]       # Sky blue
color_container = COLOR_PALETTE[0]  # Blue
color_db = COLOR_PALETTE[2]         # Green
color_external = COLOR_PALETTE[3]   # Pink
color_network = COLOR_PALETTE[1]    # Orange (network boundary)

# Shared style constants (matching architecture diagram)
box_alpha = 0.4
box_linewidth = 1.0
arrow_linewidth = 1.0
arrow_alpha = 0.7

# Font sizes (using utils style)
font_sizes = PLOT_STYLE["font_sizes"]
label_size = font_sizes["axes_label"]
small_size = font_sizes["tick_label"]

# ============================================================================
# Uniform vertical layout — all boxes same height, equal gaps between layers
# ============================================================================
box_h = 0.7
top_y = 9.0    # top of User Browser box
bot_y = 3.2    # bottom of External Services box
n_layers = 4
layer_gap = (top_y - bot_y - n_layers * box_h) / (n_layers - 1)

user_y = top_y - box_h
chatbot_y = user_y - layer_gap - box_h
services_y = chatbot_y - layer_gap - box_h
external_y = services_y - layer_gap - box_h

# ============================================================================
# Docker Network boundary (encompasses chatbot + services layers)
# ============================================================================
network_pad_bot = 0.45
network_pad_top = 0.45
network_bot = services_y - network_pad_bot
network_top = chatbot_y + box_h + network_pad_top
network_box = Rectangle(
    (0.3, network_bot),
    9.4,
    network_top - network_bot,
    edgecolor=color_network,
    facecolor=color_network,
    linewidth=box_linewidth,
    alpha=0.1,
)
ax.add_patch(network_box)
ax.text(
    0.5,
    (chatbot_y + box_h + network_top) / 2,
    r"\textbf{Docker Network}",
    fontsize=label_size,
    color="black",
    ha="left",
    va="center",
)

# ============================================================================
# Layer 1: User Browser
# ============================================================================
browser_box = FancyBboxPatch(
    (3.5, user_y),
    3,
    box_h,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_user,
    linewidth=box_linewidth,
    alpha=box_alpha,
)
ax.add_patch(browser_box)
ax.text(
    5,
    user_y + box_h / 2,
    r"\textbf{User Browser}",
    fontsize=label_size,
    ha="center",
    va="center",
)

# Arrow from browser to chatbot
arrow_browser = FancyArrowPatch(
    (5, user_y),
    (5, chatbot_y + box_h),
    arrowstyle="<->",
    mutation_scale=15,
    linewidth=arrow_linewidth,
    color="black",
    alpha=arrow_alpha,
)
ax.add_patch(arrow_browser)

# ============================================================================
# Layer 2: Chatbot Container (full width, like supervisor in architecture)
# ============================================================================
chatbot_box = FancyBboxPatch(
    (0.5, chatbot_y),
    9,
    box_h,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_container,
    linewidth=box_linewidth,
    alpha=box_alpha,
)
ax.add_patch(chatbot_box)
ax.text(
    5,
    chatbot_y + box_h / 2 + 0.1,
    r"\textbf{Chatbot + Multi-Agent System}",
    fontsize=label_size,
    ha="center",
    va="center",
)
ax.text(
    5,
    chatbot_y + box_h / 2 - 0.15,
    r"\texttt{engineer-assistant-chatbot}",
    fontsize=small_size,
    ha="center",
    va="center",
)

# ============================================================================
# Layer 3: Service Containers
# ============================================================================
svc_width = 2.6
svc_height = box_h

services = [
    {"name": "PostgreSQL", "container": "postgres", "x": 0.5, "color": color_db},
    {"name": "Prusa MCP", "container": "prusa-mcp-server", "x": 3.7, "color": color_container},
    {"name": "MMORE RAG", "container": "mmore-rag-service", "x": 6.9, "color": color_container},
]

for svc in services:
    svc_box = FancyBboxPatch(
        (svc["x"], services_y),
        svc_width,
        svc_height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=svc["color"],
        linewidth=box_linewidth,
        alpha=box_alpha,
    )
    ax.add_patch(svc_box)
    ax.text(
        svc["x"] + svc_width / 2,
        services_y + svc_height / 2 + 0.1,
        r"\textbf{" + svc["name"] + "}",
        fontsize=label_size,
        ha="center",
        va="center",
    )
    ax.text(
        svc["x"] + svc_width / 2,
        services_y + svc_height / 2 - 0.15,
        r"\texttt{" + svc["container"] + "}",
        fontsize=small_size,
        ha="center",
        va="center",
    )

# Bidirectional arrows from chatbot to each service
for svc in services:
    cx = svc["x"] + svc_width / 2
    arrow = FancyArrowPatch(
        (cx, chatbot_y),
        (cx, services_y + svc_height),
        arrowstyle="<->",
        mutation_scale=12,
        linewidth=arrow_linewidth,
        color="black",
        alpha=arrow_alpha,
    )
    ax.add_patch(arrow)

# ============================================================================
# Layer 4: External Services (outside Docker network)
# ============================================================================
external_box = FancyBboxPatch(
    (0.5, external_y),
    9,
    box_h,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_external,
    linewidth=box_linewidth,
    alpha=box_alpha,
)
ax.add_patch(external_box)
ax.text(
    5,
    external_y + box_h / 2,
    r"\textbf{External Services}",
    fontsize=label_size,
    ha="center",
    va="center",
)

# Arrow from Docker network to external services
arrow_external = FancyArrowPatch(
    (5, network_bot),
    (5, external_y + box_h),
    arrowstyle="<->",
    mutation_scale=12,
    linewidth=arrow_linewidth,
    color="black",
    alpha=arrow_alpha,
)
ax.add_patch(arrow_external)

# ============================================================================
# Layer labels (on the left side, matching architecture diagram)
# ============================================================================
layer_label_size = label_size
ax.text(
    -0.3, user_y + box_h / 2, r"\textbf{Interface}",
    fontsize=layer_label_size, ha="right", va="center", color="black",
)
ax.text(
    -0.3, chatbot_y + box_h / 2, r"\textbf{Application}",
    fontsize=layer_label_size, ha="right", va="center", color="black",
)
ax.text(
    -0.3, services_y + box_h / 2, r"\textbf{Services}",
    fontsize=layer_label_size, ha="right", va="center", color="black",
)
ax.text(
    -0.3, external_y + box_h / 2, r"\textbf{External}",
    fontsize=layer_label_size, ha="right", va="center", color="black",
)

# ============================================================================
# Legend at bottom
# ============================================================================
legend_elements = [
    mpatches.Patch(
        facecolor=color_container,
        edgecolor="black",
        label=r"\textbf{Container}",
        alpha=box_alpha,
    ),
    mpatches.Patch(
        facecolor=color_db,
        edgecolor="black",
        label=r"\textbf{Database}",
        alpha=box_alpha,
    ),
    mpatches.Patch(
        facecolor=color_external,
        edgecolor="black",
        label=r"\textbf{External Service}",
        alpha=box_alpha,
    ),
]

legend = ax.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    frameon=True,
    fontsize=label_size,
    bbox_to_anchor=(0.5, -0.02),
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
output_png = output_dir / "deployment_simplified.png"
output_pdf = output_dir / "deployment_simplified.pdf"

fig.savefig(output_png, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")

print("Deployment simplified diagram saved:")
print(f"   PNG: {output_png}")
print(f"   PDF: {output_pdf}")

plt.close()
