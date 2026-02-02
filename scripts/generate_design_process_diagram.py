"""
Generate an engineering design process circular diagram for thesis.

This creates a publication-ready circular diagram showing an iterative
engineering design process with stages arranged in a cycle.
"""
# mypy: ignore-errors

import sys
from pathlib import Path

# Add benchmarks/evaluations/plots to path to import utils
plots_dir = Path(__file__).parent.parent / "benchmarks" / "evaluations" / "plots"
sys.path.insert(0, str(plots_dir))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

from utils import COLOR_PALETTE, PLOT_STYLE, setup_style  # noqa: E402

# Setup publication style
setup_style()

# Create figure with white background
fig, ax = plt.subplots(
    figsize=(PLOT_STYLE["figsize_full_width"][0], PLOT_STYLE["figsize_full_width"][0]),
    dpi=PLOT_STYLE["dpi"],
)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal")
ax.axis("off")

# Set white background
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Colors
color_center = COLOR_PALETTE[2]  # Green
color_arrows = "black"  # Black arrows for consistency
label_size = 11  # Larger for readability
title_size = 13
center_size = 12

# Design process stages (8 stages)
stages = [
    "Problem\nDefinition",
    "Requirements\nAnalysis",
    "Concept\nGeneration",
    "Design\nEvaluation",
    "Optimization\nRefinement",
    "Validation\nTesting",
    "Manufacturing\nPlanning",
    "Implementation\nFeedback",
]

# Center circle
center_circle = Circle(
    (0, 0), 0.35, facecolor=color_center, edgecolor="black", linewidth=1.5, alpha=0.3
)
ax.add_patch(center_circle)

# Center text
ax.text(
    0,
    0.05,
    r"\textbf{ENGINEERING}",
    fontsize=center_size,
    ha="center",
    va="center",
    weight="bold",
)
ax.text(
    0,
    -0.05,
    r"\textbf{DESIGN PROCESS}",
    fontsize=center_size,
    ha="center",
    va="center",
    weight="bold",
)

# Outer circle (invisible, for reference)
outer_radius = 0.95  # Increased to push text further out

# Calculate positions for stages in a circle
n_stages = len(stages)
angles = np.linspace(0, 2 * np.pi, n_stages, endpoint=False)
# Start from top (90 degrees)
angles = angles + np.pi / 2

# Draw stages and arrows
for i, (angle, stage) in enumerate(zip(angles, stages, strict=True)):
    # Position for stage label
    x = outer_radius * np.cos(angle)
    y = outer_radius * np.sin(angle)

    # Draw stage text
    ax.text(
        x,
        y,
        stage,
        fontsize=label_size,
        ha="center",
        va="center",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none"},
    )

    # Draw arrow to next stage following circular path
    next_angle = angles[(i + 1) % n_stages]

    # Arrow radius - arrows follow this circle
    arrow_radius = 0.60  # Decreased to avoid overlapping with text

    # Calculate the arc angle (need to handle wrap-around)
    if i == n_stages - 1:  # Last arrow wraps around
        angle_diff = (next_angle + 2 * np.pi) - angle
    else:
        angle_diff = next_angle - angle

    # Adjust the connectionstyle to follow the circular path better
    # Positive rad curves outward (following circle perimeter)
    rad_value = 0.5  # Positive to curve outward along the circle

    start_x = arrow_radius * np.cos(angle)
    start_y = arrow_radius * np.sin(angle)
    end_x = arrow_radius * np.cos(next_angle)
    end_y = arrow_radius * np.sin(next_angle)

    # Create arrow following circular path
    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="->",
        mutation_scale=25,
        linewidth=2.0,
        color=color_arrows,
        connectionstyle=f"arc3,rad={rad_value}",
        alpha=0.6,
    )
    ax.add_patch(arrow)

# No title - removed as requested

# Save figure
plt.tight_layout()
output_dir = Path(__file__).parent.parent / "assets"
output_dir.mkdir(exist_ok=True)

output_png = output_dir / "design_process.png"
output_pdf = output_dir / "design_process.pdf"

plt.savefig(output_png, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
plt.close()

print("✅ Design process diagram saved:")
print(f"   PNG: {output_png}")
print(f"   PDF: {output_pdf}")
