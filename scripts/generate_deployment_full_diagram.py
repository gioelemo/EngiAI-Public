"""
Generate a visual diagram of the Docker MCP deployment architecture.

This diagram shows how the Prusa MCP server runs as a separate container
and communicates with the main chatbot application via HTTP/SSE.
"""

from pathlib import Path
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
    ax.text(
        x + width / 2, y + height / 2 - 0.1, f"{icon}\n{name}", fontsize=9, ha="center"
    )


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
    "Docker Deployment Architecture",
    fontsize=22,
    fontweight="bold",
    ha="center",
)

# Docker network background (expanded to include PostgreSQL)
network_box = Rectangle(
    (0.5, 1.0),
    19,
    11.3,
    facecolor=color_network,
    alpha=0.2,
    edgecolor="green",
    linewidth=3,
    linestyle="--",
)
ax.add_patch(network_box)
ax.text(
    1.0,
    11.5,
    "⚡ Docker Network: engiai",
    fontsize=11,
    fontweight="bold",
    color="green",
)

# ============================================================================
# LEFT SIDE: Prusa MCP Server & PostgreSQL Containers
# ============================================================================

# PostgreSQL Container
draw_container(
    ax,
    x=1.0,
    y=1.5,
    width=5.5,
    height=3.5,
    label="□ postgres",
    color="#E8EAF6",
    services=[
        {"name": "PostgreSQL 15", "port": "5432"},
        {"name": "DB: engiai"},
    ],
)

# Port mapping indicator for PostgreSQL
ax.text(
    3.75,
    1.6,
    "Port Mapping: 5432 → 5432",
    fontsize=9,
    ha="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
)

# Volume mount indicator for PostgreSQL data
volume_box_postgres = FancyBboxPatch(
    (1.25, 2.0),
    5.0,
    0.6,
    boxstyle="round,pad=0.05",
    edgecolor="orange",
    facecolor=color_volume,
    linewidth=2,
    linestyle="--",
)
ax.add_patch(volume_box_postgres)
ax.text(
    3.8,
    2.2,
    "⊕ postgres_data (persistent)\n (Chat & Settings)",
    fontsize=8,
    ha="center",
)

# Prusa MCP Server Container
draw_container(
    ax,
    x=1.0,
    y=5.5,
    width=5.5,
    height=5.5,
    label="□ prusa-mcp-server",
    color=color_mcp_container,
    services=[
        {"name": "FastMCP Server", "port": "8000 (internal)"},
        {"name": "SSE Endpoint", "port": "/sse"},
    ],
)

# Port mapping indicator
ax.text(
    3.75,
    5.6,
    "Port Mapping: 8765 → 8000",
    fontsize=9,
    ha="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
)


# Volume mount indicator for prusa-mcp
volume_box1 = FancyBboxPatch(
    (1.25, 6.0),
    2.0,
    0.6,
    boxstyle="round,pad=0.05",
    edgecolor="orange",
    facecolor=color_volume,
    linewidth=2,
    linestyle="--",
)
ax.add_patch(volume_box1)
ax.text(2.3, 6.2, "⊕ prusa-mcp folder\n (MCP files)", fontsize=8, ha="center")

# Volume mount indicator for data
volume_box2 = FancyBboxPatch(
    (4.25, 6.0),
    2.0,
    0.6,
    boxstyle="round,pad=0.05",
    edgecolor="orange",
    facecolor=color_volume,
    linewidth=2,
    linestyle="--",
)
ax.add_patch(volume_box2)
ax.text(
    5.25, 6.2, "⊕ ./data/prusa_tokens.json\n (OAuth2 refresh token)", fontsize=8, ha="center"
)

# Prusa Connect API box (external service the MCP server connects to)
prusa_connect_box = FancyBboxPatch(
    (1.25, 7.3),
    5,
    0.9,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="#F0F0F0",
    linewidth=1.5,
    linestyle="--",
)
ax.add_patch(prusa_connect_box)
ax.text(3.75, 7.75, "☁ Prusa Connect (External API)", fontsize=9, ha="center")


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
    label="□ engiai-chatbot",
    color=color_chatbot_container,
    services=[
        {"name": "Streamlit Web UI", "port": "8501"},
    ],
)

# Port mapping indicator
ax.text(
    14,
    5.6,
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
    {"name": "ArXiv", "icon": "◐", "x": 16.0, "y": 8.0},
    {"name": "RAG", "icon": "◈", "x": 17.5, "y": 8.0},
    {"name": "Prusa", "icon": "◆", "x": 13.0, "y": 6.9},
    {"name": "HPC", "icon": "●", "x": 14.5, "y": 6.9},
    {"name": "CLI", "icon": "▶", "x": 16.0, "y": 6.9},
]

for agent in agents:
    draw_agent(
        ax,
        float(agent["x"]),
        float(agent["y"]),
        1.0,
        0.5,
        str(agent["name"]),
        str(agent["icon"]),
        color_agent,
    )


# Connection from chatbot to PostgreSQL
draw_arrow(
    ax,
    7.99,
    5.45,
    6.5,
    5.0,
    "PostgreSQL\nCheckpointer\n(port 5432)",
    "#9C27B0",
    "-",
)

# Add dependency note
ax.text(
    7.25,
    4.5,
    "⚡ Health Check:\nChatbot waits for\nPostgreSQL ready",
    fontsize=8,
    ha="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#E8F5E9", "alpha": 0.8},
    style="italic",
)

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
    (10.5, 12.5),
    3.5,
    0.7,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor=color_browser,
    linewidth=2,
)
ax.add_patch(browser_box)
ax.text(
    12.25,
    12.75,
    "→ User Browser",
    fontsize=11,
    ha="center",
    fontweight="bold",
)

# Arrow from browser to Streamlit
draw_arrow(ax, 12.25, 12.45, 12.25, 11.0, "→ http://localhost:8501", "#3498DB")

# Host Service (runs on host machine, outside Docker)
host_service_box = FancyBboxPatch(
    (14.5, 12.5),
    3.5,
    0.7,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="#FFE5CC",
    linewidth=2,
)
ax.add_patch(host_service_box)
ax.text(
    16.25,
    12.75,
    "⚙ Host Service\n(host_service.py)",
    fontsize=10,
    ha="center",
    fontweight="bold",
)

# Arrow from chatbot to Host Service
draw_arrow(
    ax,
    16.25,
    11.0,
    16.25,
    12.45,
    "http://host.docker.internal:9999\nOpen GUI Apps",
    "#FF6B6B",
    "--",
)

# External API connections (positioned below PostgreSQL container)
external_y = 4.25
external_apis: list[dict[str, Any]] = [
    {"name": "OpenAI API", "icon": "◈", "x": 8},  # AI/ML Service
    {"name": "Tavily API", "icon": "◉", "x": 9.75},  # Search service
    {"name": "ArXiv API", "icon": "◐", "x": 11.5},  # Research papers
    {"name": "Weights & Biases", "icon": "◈", "x": 13.25},  # AI/ML Service
    {"name": "Hugging Face", "icon": "◈", "x": 15},  # AI/ML Service
    {"name": "HPC Cluster", "icon": "◆", "x": 16.75},  # Compute Service
]

# MMORE Service (bottom right - multimodal RAG backend)
mmore_service_box = FancyBboxPatch(
    (16.0, 1.5),
    3.0,
    2.2,
    boxstyle="round,pad=0.05",
    edgecolor="black",
    facecolor="#B2DFDB",  # Teal-green for MMORE
    linewidth=2.5,
)
ax.add_patch(mmore_service_box)
ax.text(
    17.5,
    3.3,
    "◈ MMORE Service",
    fontsize=11,
    ha="center",
    fontweight="bold",
)
ax.text(
    17.5,
    2.95,
    "Multimodal RAG",
    fontsize=9,
    ha="center",
    style="italic",
)
ax.text(
    17.5,
    2.65,
    "Port: 8000",
    fontsize=8,
    ha="center",
    color="#00695C",
)
ax.text(
    17.5,
    2.35,
    "PDF • Images • Tables",
    fontsize=7,
    ha="center",
    color="#00695C",
)
ax.text(
    17.5,
    2.05,
    "Vector Store + Indexing",
    fontsize=7,
    ha="center",
    color="#00695C",
)
ax.text(
    17.5,
    1.75,
    "/v1/files • /v1/retrieve",
    fontsize=7,
    ha="center",
    family="monospace",
    color="#004D40",
)

# Connection from RAG Agent to MMORE Service
draw_arrow(
    ax,
    18.25,
    8.0,
    18.25,
    3.75,
    "HTTP REST API\nUpload/Retrieve",
    "#20C997",
    "-",
)

for api in external_apis:
    api_box = FancyBboxPatch(
        (float(api["x"]), external_y),
        1.2,
        0.5,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor="#F0F0F0",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(api_box)
    ax.text(
        float(api["x"]) + 0.6,
        external_y + 0.25,
        f"{api['icon']} {api['name']}",
        fontsize=8,
        ha="center",
    )

# Connection arrows to external APIs
draw_arrow(ax, 8.625, 5.45, 8.625, 4.85, "", "#95A5A6", "--")  # To OpenAI
draw_arrow(ax, 10.325, 5.45, 10.325, 4.85, "", "#95A5A6", "--")  # To Tavily
draw_arrow(ax, 12.075, 5.45, 12.075, 4.85, "", "#95A5A6", "--")  # To ArXiv
draw_arrow(ax, 13.825, 5.45, 13.825, 4.85, "", "#95A5A6", "--")  # To Weights & Biases
draw_arrow(ax, 15.575, 5.45, 15.575, 4.85, "", "#95A5A6", "--")  # To Hugging Face
draw_arrow(ax, 17.325, 5.45, 17.325, 4.85, "", "#95A5A6", "--")  # To HPC

# ============================================================================
# LEGEND (Adjusted position to avoid PostgreSQL container)
# ============================================================================

legend_x = 8.0
legend_y = 2.5

ax.text(legend_x, legend_y + 1.3, "Legend:", fontsize=11, fontweight="bold")

# Colors legend
legend_items: list[dict[str, str | float]] = [
    {"color": color_mcp_container, "label": "Prusa MCP Server Container", "alpha": 0.3},
    {"color": color_chatbot_container, "label": "Main Chatbot Container", "alpha": 0.3},
    {"color": "#E8EAF6", "label": "PostgreSQL Database Container", "alpha": 0.3},
    {"color": "#FFE5CC", "label": "Host Service (on Host Machine)", "alpha": 1.0},
    {"color": color_volume, "label": "Volume Mount (Host → Container)", "alpha": 1.0},
    {"color": "#E74C3C", "label": "HTTP/SSE Communication", "alpha": 1.0},
    {"color": "#9C27B0", "label": "Database Connection", "alpha": 1.0},
    {"color": "#FF6B6B", "label": "Host Service Connection", "alpha": 1.0},
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
symbols_x = 10.75
symbols_y = 2.5

ax.text(symbols_x, symbols_y + 1.3, "Symbols:", fontsize=11, fontweight="bold")

symbol_items: list[dict[str, str]] = [
    {"symbol": "□", "label": "Docker Container"},
    {"symbol": "★", "label": "Supervisor Agent"},
    {"symbol": "⚙", "label": "Engineering Agent"},
    {"symbol": "◉", "label": "Search Agent/Service"},
    {"symbol": "◈", "label": "AI/ML/Document Agent/Service"},
    {"symbol": "◐", "label": "ArXiv Agent (Papers)"},
    {"symbol": "◆", "label": "Prusa/HPC/Compute Agent/Service"},
    {"symbol": "●", "label": "HPC Agent"},
    {"symbol": "▶", "label": "CLI Agent"},
    {"symbol": "☁", "label": "Cloud Service"},
    {"symbol": "◈", "label": "MMORE RAG Service"},
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
        x_pos = symbols_x + 2.5  # Offset for second column

    ax.text(x_pos, y_pos, symbol_item["symbol"], fontsize=12, va="center", ha="center")
    ax.text(x_pos + 0.3, y_pos, symbol_item["label"], fontsize=8, va="center")

# Ensure assets directory exists and save the diagram there
assets_dir = Path("assets")
assets_dir.mkdir(parents=True, exist_ok=True)

output_path = assets_dir / "deployment_full.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[OK] Deployment full diagram saved to: {output_path}")

# Also save as PDF for better quality
output_path_pdf = assets_dir / "deployment_full.pdf"
plt.savefig(output_path_pdf, format="pdf", bbox_inches="tight", facecolor="white")
print(f"[OK] Deployment full diagram (PDF) saved to: {output_path_pdf}")
