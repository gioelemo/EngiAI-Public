"""
Generate a visual diagram of the Docker MCP deployment architecture.

This diagram shows how the Prusa MCP server runs as a separate container
and communicates with the main chatbot application via HTTP/SSE.
"""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def draw_container(  # noqa: PLR0913
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
    services: list[dict[str, str]] | None = None,
) -> None:
    """Draw a Docker container with optional internal services."""
    # Container border
    container = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=color,
        linewidth=2,
        alpha=0.3,
    )
    ax.add_patch(container)

    # Container label
    ax.text(
        x + width / 2,
        y + height - 0.3,
        label,
        fontsize=12,
        fontweight="bold",
        ha="center",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "black"},
    )

    # Draw internal services if provided
    if services:
        service_y = y + height - 1.0
        for service in services:
            service_box = FancyBboxPatch(
                (x + 0.2, service_y),
                width - 0.4,
                0.6,
                boxstyle="round,pad=0.05",
                edgecolor="black",
                facecolor="white",
                linewidth=1,
            )
            ax.add_patch(service_box)
            ax.text(
                x + width / 2,
                service_y + 0.3,
                service["name"],
                fontsize=10,
                ha="center",
                fontweight="bold",
            )
            if "port" in service:
                ax.text(
                    x + width / 2,
                    service_y + 0.1,
                    f"Port: {service['port']}",
                    fontsize=8,
                    ha="center",
                    style="italic",
                )
            service_y -= 0.8


def draw_arrow(  # noqa: PLR0913
    ax: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str = "",
    color: str = "#34495E",
    style: str = "-",
) -> None:
    """Draw an arrow with optional label."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=25,
        linewidth=2.5,
        color=color,
        linestyle=style,
    )
    ax.add_patch(arrow)

    # Add label in the middle of the arrow
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x,
            mid_y + 0.2,
            label,
            fontsize=9,
            ha="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )


def draw_agent(  # noqa: PLR0913
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    name: str,
    icon: str,
    color: str,
) -> None:
    """Draw an agent box."""
    agent_box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(agent_box)
    ax.text(x + width / 2, y + height / 2, f"{icon}\n{name}", fontsize=9, ha="center")


# Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")

# Color scheme
color_docker = "#2496ED"
color_mcp_container = "#BBDEFB"
color_chatbot_container = "#FFF9C4"
color_network = "#C8E6C9"
color_volume = "#FFE0B2"
color_agent = "#D4E6F1"
color_browser = "#E8F4F8"

# Title
ax.text(
    10,
    13.5,
    "Docker MCP Deployment Architecture",
    fontsize=22,
    fontweight="bold",
    ha="center",
)

# Docker network background
network_box = Rectangle(
    (0.5, 1.0),
    19,
    11,
    facecolor=color_network,
    alpha=0.2,
    edgecolor="green",
    linewidth=3,
    linestyle="--",
)
ax.add_patch(network_box)
ax.text(
    1.5,
    11.2,
    "⚡ Docker Network: engineer-assistant",
    fontsize=11,
    fontweight="bold",
    color="green",
)

# ============================================================================
# LEFT SIDE: Prusa MCP Server Container
# ============================================================================

# Prusa MCP Server Container
draw_container(
    ax,
    x=1.0,
    y=5.5,
    width=5.5,
    height=5.5,
    label="⚙ prusa-mcp-server",
    color=color_mcp_container,
    services=[
        {"name": "FastMCP Server", "port": "8000 (internal)"},
        {"name": "SSE Endpoint", "port": "/sse"},
    ],
)

# Port mapping indicator
ax.text(
    3.75,
    5.2,
    "Port Mapping: 8765 → 8000",
    fontsize=9,
    ha="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
)

# MCP Server components
components_y = 8.5
components = [
    "[HTTP/SSE] Transport",
    "[TOOLS] MCP Tools Handler",
    "[API] Prusa Connect API",
    "[BROWSER] Playwright Browser",
]

for i, comp in enumerate(components):
    ax.text(1.5, components_y - i * 0.5, comp, fontsize=9)

# Volume mount indicator for prusa-mcp
volume_box1 = FancyBboxPatch(
    (1.3, 6.0),
    2.0,
    0.6,
    boxstyle="round,pad=0.05",
    edgecolor="orange",
    facecolor=color_volume,
    linewidth=2,
    linestyle="--",
)
ax.add_patch(volume_box1)
ax.text(2.3, 6.3, "⊕ ~/Desktop/prusa-mcp", fontsize=8, ha="center")

# Volume mount indicator for data
volume_box2 = FancyBboxPatch(
    (4.0, 6.0),
    2.0,
    0.6,
    boxstyle="round,pad=0.05",
    edgecolor="orange",
    facecolor=color_volume,
    linewidth=2,
    linestyle="--",
)
ax.add_patch(volume_box2)
ax.text(5.0, 6.3, "⊕ ./data/connect_state", fontsize=8, ha="center")

# ============================================================================
# RIGHT SIDE: Main Chatbot Container
# ============================================================================

# Main Chatbot Container
draw_container(
    ax,
    x=8.0,
    y=5.5,
    width=11,
    height=5.5,
    label="⚙ engineer-assistant-chatbot",
    color=color_chatbot_container,
    services=[
        {"name": "Streamlit Web UI", "port": "8501"},
    ],
)

# Port mapping indicator
ax.text(
    14,
    5.2,
    "Port Mapping: 8501 → 8501",
    fontsize=9,
    ha="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
)

# Supervisor agent box (inside chatbot container)
supervisor_box = FancyBboxPatch(
    (8.25, 7.3),
    4.5,
    0.9,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="#FFE5B4",
    linewidth=2,
)
ax.add_patch(supervisor_box)
ax.text(
    10.5,
    7.9,
    "★ Supervisor",
    fontsize=11,
    ha="center",
    fontweight="bold",
)
ax.text(
    10.5,
    7.55,
    "Routes to specialized agents",
    fontsize=8,
    ha="center",
    style="italic",
)

# Specialized Agents (inside chatbot container)
agents: list[dict[str, str | float]] = [
    {"name": "Engineering", "icon": "⚙", "x": 13.0, "y": 8.0},
    {"name": "Search", "icon": "◉", "x": 14.5, "y": 8.0},
    {"name": "RAG", "icon": "◈", "x": 16.0, "y": 8.0},
    {"name": "ArXiv", "icon": "◐", "y": 8.0, "x": 17.5},
    {"name": "Prusa", "icon": "◆", "x": 13.0, "y": 6.9},
    {"name": "HPC", "icon": "●", "x": 14.5, "y": 6.9},
    {"name": "CLI", "icon": "▶", "x": 16.0, "y": 6.9},
]

for agent in agents:
    draw_agent(
        ax,
        float(agent["x"]),
        float(agent["y"]),
        1.2,
        0.6,
        str(agent["name"]),
        str(agent["icon"]),
        color_agent,
    )


# Volume mount indicator for data
volume_box3 = FancyBboxPatch(
    (8.5, 6.0),
    2.0,
    0.6,
    boxstyle="round,pad=0.05",
    edgecolor="orange",
    facecolor=color_volume,
    linewidth=2,
    linestyle="--",
)
ax.add_patch(volume_box3)
ax.text(9.5, 6.3, "⊕ ./data/conversations", fontsize=8, ha="center")

# ============================================================================
# COMMUNICATION FLOWS
# ============================================================================

# Main HTTP/SSE communication between containers
draw_arrow(
    ax,
    6.5,
    8.2,
    8.0,
    8.2,
    "HTTP/SSE\nlist_tools()\ncall_tool()",
    "#E74C3C",
    "-",
)

draw_arrow(
    ax,
    8.0,
    7.7,
    6.5,
    7.7,
    "Tool Results\n(JSON)",
    "#27AE60",
    "--",
)

# Prusa Agent specifically uses MCP client
prusa_to_mcp_start_x = 14.1  # Right edge of Prusa agent
prusa_to_mcp_start_y = 7.8  # Center of Prusa agent
mcp_server_receive_x = 6.5  # Right edge of MCP server container
mcp_server_receive_y = 8.5


# ============================================================================
# EXTERNAL CONNECTIONS
# ============================================================================

# User's browser
browser_box = FancyBboxPatch(
    (12.0, 12.2),
    3.5,
    0.7,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_browser,
    linewidth=2,
)
ax.add_patch(browser_box)
ax.text(
    13.75,
    12.55,
    "→ User Browser",
    fontsize=11,
    ha="center",
    fontweight="bold",
)

# Arrow from browser to Streamlit
draw_arrow(ax, 13.75, 12.2, 13.75, 11.0, "→ http://localhost:8501", "#3498DB")

# External API connections
external_y = 4.25
external_apis: list[dict[str, Any]] = [
    {"name": "Prusa Connect", "icon": "☁", "x": 1.0},  # Under MCP server
    {"name": "OpenAI API", "icon": "◈", "x": 8.0},  # AI/ML Service
    {"name": "Tavily API", "icon": "◉", "x": 10.0},  # Search service
    {"name": "Weights & Biases", "icon": "◈", "x": 12.0},  # AI/ML Service
    {"name": "Hugging Face", "icon": "◈", "x": 14.0},  # AI/ML Service
    {"name": "HPC Cluster", "icon": "◆", "x": 16.0},  # Compute Service
]

for api in external_apis:
    api_box = FancyBboxPatch(
        (float(api["x"]), external_y),
        1.8,
        0.6,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor="#F0F0F0",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(api_box)
    ax.text(
        float(api["x"]) + 0.9,
        external_y + 0.3,
        f"{api['icon']} {api['name']}",
        fontsize=8,
        ha="center",
    )

# Connection arrows to external APIs
draw_arrow(
    ax, 2.0, 5.45, 2.0, 4.85, "", "#95A5A6", "--"
)  # MCP to Prusa Connect (vertical)
draw_arrow(ax, 9.0, 5.45, 9.0, 4.85, "", "#95A5A6", "--")  # To OpenAI
draw_arrow(ax, 11.0, 5.45, 11.0, 4.85, "", "#95A5A6", "--")  # To Tavily
draw_arrow(ax, 13.0, 5.45, 13.0, 4.85, "", "#95A5A6", "--")  # To Weights & Biases
draw_arrow(ax, 15.0, 5.45, 15.0, 4.85, "", "#95A5A6", "--")  # To Hugging Face
draw_arrow(ax, 17.0, 5.45, 17.0, 4.85, "", "#95A5A6", "--")  # To HPC

# ============================================================================
# LEGEND
# ============================================================================

legend_x = 1.0
legend_y = 2.5

ax.text(legend_x, legend_y + 1.3, "Legend:", fontsize=11, fontweight="bold")

# Colors legend
legend_items: list[dict[str, str | float]] = [
    {"color": color_mcp_container, "label": "Prusa MCP Server Container", "alpha": 0.3},
    {"color": color_chatbot_container, "label": "Main Chatbot Container", "alpha": 0.3},
    {"color": color_volume, "label": "Volume Mount (Host -> Container)", "alpha": 1.0},
    {"color": "#E74C3C", "label": "HTTP/SSE Communication", "alpha": 1.0},
    {"color": "#95A5A6", "label": "External API Calls", "alpha": 1.0},
]

for i, item in enumerate(legend_items):
    y_pos = legend_y + 0.8 - i * 0.25

    # Draw colored box
    legend_box = Rectangle(
        (legend_x, y_pos - 0.1),
        0.3,
        0.2,
        facecolor=str(item["color"]),
        edgecolor="black",
        linewidth=1,
        alpha=float(item["alpha"]),
    )
    ax.add_patch(legend_box)

    # Draw label
    ax.text(legend_x + 0.4, y_pos, str(item["label"]), fontsize=8, va="center")

# Symbols legend (two columns)
symbols_x = 6.5
symbols_y = 2.5

ax.text(symbols_x, symbols_y + 1.3, "Symbols:", fontsize=11, fontweight="bold")

symbol_items: list[dict[str, str]] = [
    {"symbol": "⚙", "label": "Container/Engineering Agent"},
    {"symbol": "★", "label": "Supervisor Agent"},
    {"symbol": "◉", "label": "Search Agent/Service"},
    {"symbol": "◈", "label": "AI/ML/Document Agent/Service"},
    {"symbol": "◐", "label": "ArXiv Agent (Papers)"},
    {"symbol": "◆", "label": "Prusa/HPC/Compute Agent/Service"},
    {"symbol": "●", "label": "HPC Agent"},
    {"symbol": "▶", "label": "CLI Agent"},
    {"symbol": "☁", "label": "Cloud Service"},
    {"symbol": "⊕", "label": "Volume Mount"},
    {"symbol": "⚡", "label": "Docker Network"},
    {"symbol": "→", "label": "User Connection"},
]

# Display symbols in two columns
items_per_column = (len(symbol_items) + 1) // 2  # Split into two columns
symbol_item: dict[str, str]
for i, symbol_item in enumerate(symbol_items):
    if i < items_per_column:
        # First column
        y_pos = symbols_y + 0.8 - i * 0.25
        x_pos = symbols_x
    else:
        # Second column
        y_pos = symbols_y + 0.8 - (i - items_per_column) * 0.25
        x_pos = symbols_x + 5.5  # Offset for second column

    ax.text(x_pos, y_pos, symbol_item["symbol"], fontsize=12, va="center", ha="center")
    ax.text(x_pos + 0.3, y_pos, symbol_item["label"], fontsize=8, va="center")

# Save the diagram
output_path = "outputs/docker_mcp_architecture.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[OK] Docker MCP architecture diagram saved to: {output_path}")

# Also save as PDF for better quality
output_path_pdf = "outputs/docker_mcp_architecture.pdf"
plt.savefig(output_path_pdf, format="pdf", bbox_inches="tight", facecolor="white")
print(f"[OK] Docker MCP architecture diagram (PDF) saved to: {output_path_pdf}")

plt.show()
