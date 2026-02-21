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
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

from utils import COLOR_PALETTE, PLOT_STYLE, setup_style  # noqa: E402

# Setup publication style
setup_style()

# Create figure — square, matching full-width of other diagrams
fig, ax = plt.subplots(
    figsize=(PLOT_STYLE["figsize_full_width"][0], PLOT_STYLE["figsize_full_width"][0]),
    dpi=PLOT_STYLE["dpi"],
)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Color scheme — all from COLOR_PALETTE (Okabe-Ito)
color_center = COLOR_PALETTE[1]  # Orange
color_stage = COLOR_PALETTE[0]   # Blue

# Shared style constants (matching architecture/deployment diagrams)
box_alpha = 0.4
box_linewidth = 1.0
arrow_linewidth = 3.0
arrow_alpha = 0.7

# Font sizes (using utils style)
font_sizes = PLOT_STYLE["font_sizes"]
label_size = font_sizes["axes_title"] + 2  # 11pt for center text
small_size = font_sizes["axes_label"] + 2  # 10pt for stage labels

# Design process stages (8 stages, counterclockwise from top)
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

n_stages = len(stages)
# Angles starting from top (pi/2), going counterclockwise
angles = np.linspace(0, 2 * np.pi, n_stages, endpoint=False) + np.pi / 2

# ============================================================================
# Center circle
# ============================================================================
center_circle = Circle(
    (0, 0),
    0.30,
    facecolor=color_center,
    edgecolor="black",
    linewidth=box_linewidth,
    alpha=box_alpha,
)
ax.add_patch(center_circle)

ax.text(
    0, 0.05,
    r"\textbf{Engineering}",
    fontsize=label_size,
    ha="center", va="center",
)
ax.text(
    0, -0.07,
    r"\textbf{Design Process}",
    fontsize=label_size,
    ha="center", va="center",
)

# ============================================================================
# Stage labels in uniform-size colored boxes
# ============================================================================
label_radius = 0.88
stage_box_w = 0.40
stage_box_h = 0.20
# Distance from center to nearest box edge (for cardinal N/S boxes)
d_inner = label_radius - stage_box_h / 2

for angle, stage in zip(angles, stages, strict=True):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # For diagonal angles, shift the box center so its inner corner
    # (not the box center) sits on the angle ray from the origin.
    is_diagonal = abs(cos_a) > 0.1 and abs(sin_a) > 0.1
    if is_diagonal:
        # Place the nearest corner at distance d_inner along the angle ray
        x = d_inner * cos_a + np.sign(cos_a) * stage_box_w / 2
        y = d_inner * sin_a + np.sign(sin_a) * stage_box_h / 2
    else:
        # Cardinal: box center on the angle ray
        x = label_radius * cos_a
        y = label_radius * sin_a

    # Fixed-size box centered at (x, y)
    stage_box = FancyBboxPatch(
        (x - stage_box_w / 2, y - stage_box_h / 2),
        stage_box_w,
        stage_box_h,
        boxstyle="round,pad=0.02",
        facecolor=color_stage,
        edgecolor="black",
        linewidth=box_linewidth,
        alpha=box_alpha,
    )
    ax.add_patch(stage_box)

    bold_stage = "\n".join(r"\textbf{" + line + "}" for line in stage.split("\n"))
    ax.text(
        x, y,
        bold_stage,
        fontsize=small_size,
        ha="center", va="center",
    )

# ============================================================================
# Circular arrows between stages (true circular arcs, not Bezier)
# ============================================================================
arrow_radius = 0.54
gap = 0.08  # radians gap near each label

for i in range(n_stages):
    a_start = angles[i] + gap
    if i < n_stages - 1:
        a_end = angles[i + 1] - gap
    else:
        a_end = angles[0] + 2 * np.pi - gap

    # Generate points along the actual circular arc
    n_pts = 50
    theta = np.linspace(a_start, a_end, n_pts)
    xs = arrow_radius * np.cos(theta)
    ys = arrow_radius * np.sin(theta)

    # Draw arc body (stop before end to leave room for arrowhead)
    ax.plot(xs[:-3], ys[:-3], "-", color="black",
            linewidth=arrow_linewidth, alpha=arrow_alpha, solid_capstyle="round")

    # Arrowhead as triangle at the end, oriented along the tangent
    tang = np.array([-np.sin(a_end), np.cos(a_end)])   # tangent (CCW direction)
    norm = np.array([np.cos(a_end), np.sin(a_end)])     # outward normal

    tip = np.array([xs[-1], ys[-1]])
    arrow_len = 0.05
    arrow_half_w = 0.035
    base = tip - arrow_len * tang

    triangle = Polygon(
        [tip, base + arrow_half_w * norm, base - arrow_half_w * norm],
        closed=True,
        facecolor="black",
        edgecolor="black",
        alpha=arrow_alpha,
        zorder=3,
    )
    ax.add_patch(triangle)

# ============================================================================
# Save figure
# ============================================================================
plt.tight_layout()

output_dir = Path(__file__).parent.parent / "assets"
output_dir.mkdir(exist_ok=True)

output_png = output_dir / "design_process.png"
output_pdf = output_dir / "design_process.pdf"

plt.savefig(output_png, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
plt.close()

print("Design process diagram saved:")
print(f"   PNG: {output_png}")
print(f"   PDF: {output_pdf}")
