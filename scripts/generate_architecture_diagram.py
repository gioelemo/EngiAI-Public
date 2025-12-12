"""
Generate a visual diagram of the multi-agent system architecture.
"""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


class AgentInfo(TypedDict, total=False):
    """Type definition for agent information."""

    name: str
    icon: str
    x: float
    y: float
    tools: list[str]
    desc: str
    planned: bool  # Optional: True if agent is not yet implemented


# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(22, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")

# Color scheme
color_user = "#E8F4F8"
color_supervisor = "#FFE5B4"
color_agent = "#D4E6F1"
color_tool = "#E8DAEF"
color_arrow = "#34495E"
color_storage = "#C8E6C9"  # Green for storage systems

# Tool category colors
tool_colors = {
    "engibench": "#FF6B6B",  # Red - EngiBench tools
    "engiopt": "#51CF66",  # Green - WandB/engiopt tools
    "stl": "#4DABF7",  # Blue - STL export tools
    "code": "#F08C00",  # Orange - Code execution tools (changed from yellow for readability)
    "search": "#9775FA",  # Purple - Search tools
    "hpc": "#FF9F1C",  # Gold/Orange - HPC tools
    "mcp": "#868E96",  # Gray - MCP tools (planned)
    "rag": "#20C997",  # Teal - RAG/document tools
    "cli": "#FD7E14",  # Orange - CLI tools
}

# Tool category mapping
tool_categories = {
    # EngiBench tools (unified)
    "create_problem": "engibench",
    "optimize_design": "engibench",
    "simulate_design": "engibench",
    "render_design": "engibench",
    "check_constraints": "engibench",
    "get_problem_info": "engibench",
    "get_problem_details": "engibench",
    "get_dataset_info": "engibench",
    # WandB/engiopt tools
    "download_wandb_model": "engiopt",
    "list_available_algorithms": "engiopt",
    "load_wandb_model": "engiopt",
    "sample_designs_from_model": "engiopt",
    # STL export tools
    "convert_design_to_stl": "stl",
    # Code execution tools (use CLI-based executor instead)
    # Search tools
    "create_search_tool": "search",
    # HPC tools
    "test_hpc_connection": "hpc",
    "submit_slurm_job": "hpc",
    "get_slurm_job_status": "hpc",
    "cancel_slurm_job": "hpc",
    "download_job_outputs": "hpc",
    # MCP tools (planned) - none currently exported; keep mapping minimal
    # RAG tools
    "search_documents": "rag",
    "add_document": "rag",
    "list_documents": "rag",
    "clear_document_memory": "rag",
    # CLI tools
    "execute_cli_command": "cli",
    "list_directory_contents": "cli",
    "open_gui_application": "cli",
    # Add mappings for Prusa / printer-related tools (used by Prusa Agent)
    "get_printers": "mcp",
    "get_printer_status": "mcp",
    "get_printer_jobs": "mcp",
    "get_printer_files": "mcp",
    "get_printer_storages": "mcp",
    "get_printer_events": "mcp",
    "list_print_jobs": "mcp",
    "control_printer": "mcp",
}

printer_tools = [
    "get_printers",
    "get_printer_status",
    "get_printer_jobs",
    "get_printer_files",
    "get_printer_storages",
    "get_printer_events",
    "list_print_jobs",
    "control_printer",
]
for t in printer_tools:
    tool_categories.setdefault(t, "mcp")

# Title
ax.text(
    8,
    11.5,
    "Multi-Agent Engineer Assistant System",
    fontsize=20,
    fontweight="bold",
    ha="center",
)

# User
user_box = FancyBboxPatch(
    (6.5, 9.8),
    3,
    0.8,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_user,
    linewidth=2,
)
ax.add_patch(user_box)
ax.text(8, 10.2, "User", fontsize=14, ha="center", fontweight="bold")

# Supervisor Agent
supervisor_box = FancyBboxPatch(
    (5.5, 7.8),
    5,
    1.2,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_supervisor,
    linewidth=3,
)
ax.add_patch(supervisor_box)
ax.text(8, 8.65, "Supervisor Agent", fontsize=14, ha="center", fontweight="bold")
ax.text(
    8,
    8.25,
    "Routes requests to specialized agents",
    fontsize=9,
    ha="center",
    style="italic",
)

# Arrow from user to supervisor
arrow1 = FancyArrowPatch(
    (8, 9.8),
    (8, 9.0),
    arrowstyle="->",
    mutation_scale=30,
    linewidth=2,
    color=color_arrow,
)
ax.add_patch(arrow1)
ax.text(8.3, 9.4, "Query", fontsize=9)

# Arrow from supervisor back to user
arrow2 = FancyArrowPatch(
    (7.7, 9.0),
    (7.7, 9.8),
    arrowstyle="->",
    mutation_scale=30,
    linewidth=2,
    color=color_arrow,
    linestyle="--",
)
ax.add_patch(arrow2)
ax.text(7.0, 9.4, "Response", fontsize=9)

# Agent position thresholds for arrow routing
LEFT_AGENT_THRESHOLD = 5.0  # Agent x-position below this is considered "left"
RIGHT_AGENT_THRESHOLD = 9.0  # Agent x-position above this is considered "right"

# Specialized Agents
agents: list[AgentInfo] = [
    {
        "name": "RAG Agent",
        "icon": "",
        "x": 0.5,
        "y": 5.0,
        "tools": [
            "search_documents",
            "add_document",
            "list_documents",
            "clear_document_memory",
        ],
        "desc": "Document Q&A, PDF analysis, knowledge base",
    },
    {
        "name": "Engineering Agent",
        "icon": "",
        "x": 3.3,
        "y": 2.0,
        "tools": [
            # EngiBench tools (red, unified)
            "create_problem",
            "optimize_design",
            "simulate_design",
            "render_design",
            "check_constraints",
            "get_problem_info",
            "get_problem_details",
            "get_dataset_info",
            # WandB/engiopt tools (green)
            "download_wandb_model",
            "list_available_algorithms",
            "load_wandb_model",
            "sample_designs_from_model",
            # STL export tools (blue)
            "convert_design_to_stl",
        ],
        "desc": "Structural optimization, EngiBench, ML models",
    },
    {
        "name": "HPC Agent",
        "icon": "",
        "x": 6.3,
        "y": 2.0,
        "tools": [
            "test_hpc_connection",
            "submit_slurm_job",
            "get_slurm_job_status",
            "cancel_slurm_job",
            "download_job_outputs",
        ],
        "desc": "SLURM job management, euler.ethz.ch cluster",
    },
    {
        "name": "Prusa Agent",
        "icon": "",
        "x": 9.3,
        "y": 2.0,
        "tools": ["get_printer_status", "list_print_jobs", "control_printer"],
        "desc": "Prusa Connect 3D printer management",
    },
    {
        "name": "CLI Agent",
        "icon": "",
        "x": 12.3,
        "y": 2.0,
        "tools": [
            "execute_cli_command",
            "list_directory_contents",
        ],
        "desc": "Local command execution, PrusaSlicer",
    },
    {
        "name": "Search Agent",
        "icon": "",
        "x": 12.8,
        "y": 5.0,
        "tools": ["create_search_tool"],
        "desc": "Web research, information gathering",
    },
]

# Draw agents and their connections
for agent in agents:
    # Calculate box height based on number of tools (excluding comments)
    num_tools = len(
        [t for t in agent["tools"] if not (isinstance(t, str) and t.startswith("#"))]
    )
    box_height = 1.2 + (num_tools * 0.18)  # Dynamic height based on tools

    # Agent box (dashed if planned)
    is_planned = agent.get("planned", False)
    agent_box = FancyBboxPatch(
        (agent["x"], agent["y"]),
        2.5,
        box_height,
        boxstyle="round,pad=0.1",
        edgecolor="gray" if is_planned else "black",
        facecolor="#F8F9FA" if is_planned else color_agent,
        linewidth=2,
        linestyle="--" if is_planned else "-",
    )
    ax.add_patch(agent_box)

    # Agent name
    agent_name = agent["name"]
    if agent["icon"]:
        agent_name = f"{agent['icon']} {agent_name}"

    # Check if name has "(Currently Disabled)" suffix
    if "(Currently Disabled)" in agent_name:
        # Split the name and the status
        base_name = agent_name.replace(" (Currently Disabled)", "")
        ax.text(
            agent["x"] + 1.25,
            agent["y"] + box_height - 0.15,
            base_name,
            fontsize=11,
            ha="center",
            fontweight="bold",
        )
        ax.text(
            agent["x"] + 1.25,
            agent["y"] + box_height - 0.35,
            "(Currently Disabled)",
            fontsize=7,
            ha="center",
            style="italic",
            color="gray",
        )
    else:
        ax.text(
            agent["x"] + 1.25,
            agent["y"] + box_height - 0.2,
            agent_name,
            fontsize=11,
            ha="center",
            fontweight="bold",
        )

    # Agent description
    desc_y_offset = 0.55 if "(Currently Disabled)" in agent["name"] else 0.5
    ax.text(
        agent["x"] + 1.25,
        agent["y"] + box_height - desc_y_offset,
        agent["desc"],
        fontsize=8,
        ha="center",
        style="italic",
        wrap=True,
    )

    # Tools section
    ax.text(
        agent["x"] + 1.25,
        agent["y"] + box_height - 0.75,
        "Tools:",
        fontsize=9,
        ha="center",
        fontweight="bold",
    )

    # List all tools
    tools_y_start = agent["y"] + box_height - 1.0
    for i, tool in enumerate(agent["tools"]):
        # Skip comment lines
        if isinstance(tool, str) and tool.startswith("#"):
            continue

        # Get tool color based on category
        tool_category = tool_categories.get(tool, "tool")
        tool_color = tool_colors.get(tool_category, "black")

        ax.text(
            agent["x"] + 0.1,
            tools_y_start - (i * 0.18),
            f"• {tool}",
            fontsize=7,
            ha="left",
            family="monospace",
            color=tool_color,
            fontweight="bold",
        )

    # Arrow from supervisor to agent
    # Supervisor box: x=5.5-10.5, y=7.8-9.0
    # Calculate proper edge connection points
    supervisor_x = 5.5
    supervisor_width = 5
    supervisor_y = 7.8
    supervisor_bottom_y = 7.8

    agent_box_x = agent["x"]
    agent_box_width = 2.5
    agent_center_x = agent_box_x + agent_box_width / 2
    agent_top_y = agent["y"] + box_height

    # Determine supervisor exit point based on agent position
    SUPERVISOR_CENTER_X = 8.0  # Center x-position of supervisor
    ENGINEERING_AGENT_THRESHOLD = 6.5  # X position threshold for Engineering agent
    HPC_AGENT_THRESHOLD = 9.5  # X position threshold for HPC agent
    PRUSA_AGENT_THRESHOLD = 11.0  # X position threshold for Prusa agent

    # For side agents (RAG and Search), connect to sides
    if agent_center_x < LEFT_AGENT_THRESHOLD:  # Left agents (RAG)
        supervisor_exit_x = supervisor_x  # Left edge
        supervisor_exit_y = supervisor_y + 0.6  # Mid-height on left side
    elif agent_center_x > RIGHT_AGENT_THRESHOLD:  # Right agents (Search)
        supervisor_exit_x = supervisor_x + supervisor_width  # Right edge
        supervisor_exit_y = supervisor_y + 0.6  # Mid-height on right side
    else:  # Center agents (Engineering, HPC, Prusa, CLI)
        # Distribute connection points across bottom of supervisor based on agent position
        if agent_center_x < ENGINEERING_AGENT_THRESHOLD:  # Engineering Agent
            supervisor_exit_x = supervisor_x + supervisor_width * 0.25
        elif agent_center_x < HPC_AGENT_THRESHOLD:  # HPC Agent
            supervisor_exit_x = supervisor_x + supervisor_width * 0.45
        elif agent_center_x < PRUSA_AGENT_THRESHOLD:  # Prusa Agent
            supervisor_exit_x = supervisor_x + supervisor_width * 0.65
        else:  # CLI Agent
            supervisor_exit_x = supervisor_x + supervisor_width * 0.85
        supervisor_exit_y = supervisor_bottom_y

    # Adjust connection style based on agent position
    arrow_linestyle = "--" if is_planned else "-"
    arrow_color = "gray" if is_planned else color_arrow

    # Simple straight arrows (no curves, no L-shape)
    arrow = FancyArrowPatch(
        (supervisor_exit_x, supervisor_exit_y),
        (agent_center_x, agent_top_y),
        arrowstyle="<->",
        mutation_scale=20,
        linewidth=1.5,
        color=arrow_color,
        linestyle=arrow_linestyle,
    )
    ax.add_patch(arrow)

# Storage Systems Section (bottom of diagram)
storage_y = 0.5
storage_height = 1.0

# PostgreSQL Database
postgres_box = FancyBboxPatch(
    (1.0, storage_y),
    5.0,
    storage_height,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_storage,
    linewidth=2,
)
ax.add_patch(postgres_box)
ax.text(
    3.5,
    storage_y + storage_height - 0.2,
    "PostgreSQL Database",
    fontsize=11,
    ha="center",
    fontweight="bold",
)
ax.text(
    3.5,
    storage_y + storage_height - 0.5,
    "Conversation history, messages, settings",
    fontsize=8,
    ha="center",
    style="italic",
)
ax.text(
    3.5,
    storage_y + storage_height - 0.75,
    "Persistent state with PostgresSaver",
    fontsize=7,
    ha="center",
    color="gray",
)

# MMORE RAG Service
mmore_box = FancyBboxPatch(
    (6.5, storage_y),
    5.0,
    storage_height,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_storage,
    linewidth=2,
)
ax.add_patch(mmore_box)
ax.text(
    9.0,
    storage_y + storage_height - 0.2,
    "MMORE RAG Service",
    fontsize=11,
    ha="center",
    fontweight="bold",
)
ax.text(
    9.0,
    storage_y + storage_height - 0.5,
    "Multimodal document retrieval",
    fontsize=8,
    ha="center",
    style="italic",
)
ax.text(
    9.0,
    storage_y + storage_height - 0.75,
    "OpenAI embeddings (text-embedding-3-small)",
    fontsize=7,
    ha="center",
    color="gray",
)

# Add storage section label
ax.text(
    1.0,
    storage_y - 0.3,
    "Persistent Storage Layer",
    fontsize=10,
    fontweight="bold",
    color="#2E7D32",
)

# Add legend for tool colors in top left corner
legend_x = 0.5
legend_y = 11.0
ax.text(
    legend_x,
    legend_y + 0.3,
    "Tool Categories:",
    fontsize=9,
    fontweight="bold",
    ha="left",
)

legend_items = [
    ("EngiBench", "engibench"),
    ("WandB/ML", "engiopt"),
    ("STL Export", "stl"),
    ("HPC/SLURM", "hpc"),
    ("RAG/Docs", "rag"),
    ("CLI Tools", "cli"),
    ("Search", "search"),
]

for i, (label, category) in enumerate(legend_items):
    color = tool_colors[category]
    y_pos = legend_y - (i * 0.25)
    # Draw colored box
    legend_box = FancyBboxPatch(
        (legend_x, y_pos - 0.08),
        0.15,
        0.15,
        boxstyle="round,pad=0.02",
        edgecolor="black",
        facecolor=color,
        linewidth=1,
    )
    ax.add_patch(legend_box)
    # Label
    ax.text(
        legend_x + 0.25,
        y_pos,
        label,
        fontsize=8,
        ha="left",
        va="center",
    )

plt.tight_layout()
assets_dir = Path("assets")
assets_dir.mkdir(parents=True, exist_ok=True)

out_path = assets_dir / "agent_architecture.png"
plt.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print(f"✅ Architecture diagram saved to: {out_path}")
plt.close()

# Create a second diagram showing the workflow
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")

# Title
ax2.text(
    5,
    9.5,
    'Request Flow Example: "What is 1+1?"',
    fontsize=18,
    fontweight="bold",
    ha="center",
)

# Step 1: User request
step1_box = FancyBboxPatch(
    (3, 8),
    4,
    0.6,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_user,
    linewidth=2,
)
ax2.add_patch(step1_box)
ax2.text(5, 8.3, '1. User: "What is 1+1?"', fontsize=11, ha="center", fontweight="bold")

# Arrow down
arrow = FancyArrowPatch(
    (5, 8), (5, 7.3), arrowstyle="->", mutation_scale=25, linewidth=2, color=color_arrow
)
ax2.add_patch(arrow)

# Step 2: Supervisor analyzes
step2_box = FancyBboxPatch(
    (2.5, 6.5),
    5,
    0.6,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_supervisor,
    linewidth=2,
)
ax2.add_patch(step2_box)
ax2.text(
    5,
    6.8,
    '2. Supervisor analyzes: "calculation task"',
    fontsize=11,
    ha="center",
    fontweight="bold",
)

# Arrow down
arrow = FancyArrowPatch(
    (5, 6.5),
    (5, 5.8),
    arrowstyle="->",
    mutation_scale=25,
    linewidth=2,
    color=color_arrow,
)
ax2.add_patch(arrow)

# Step 3: Route to Code Execution Agent
step3_box = FancyBboxPatch(
    (2, 5),
    6,
    0.6,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor="#D5F4E6",
    linewidth=2,
)
ax2.add_patch(step3_box)
ax2.text(
    5,
    5.3,
    "3. Routes to: Code Execution Agent",
    fontsize=11,
    ha="center",
    fontweight="bold",
)

# Arrow down
arrow = FancyArrowPatch(
    (5, 5), (5, 4.3), arrowstyle="->", mutation_scale=25, linewidth=2, color=color_arrow
)
ax2.add_patch(arrow)

# Step 4: Agent executes tool
step4_box = FancyBboxPatch(
    (1.5, 3.5),
    7,
    0.6,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_tool,
    linewidth=2,
)
ax2.add_patch(step4_box)
ax2.text(
    5,
    3.8,
    '4. Calls tool: execute_cli_command("python -c "print(1+1)"")',
    fontsize=11,
    ha="center",
    fontweight="bold",
)

# Arrow down
arrow = FancyArrowPatch(
    (5, 3.5),
    (5, 2.8),
    arrowstyle="->",
    mutation_scale=25,
    linewidth=2,
    color=color_arrow,
)
ax2.add_patch(arrow)

# Step 5: Tool returns result
step5_box = FancyBboxPatch(
    (2.5, 2),
    5,
    0.6,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor="#E8DAEF",
    linewidth=2,
)
ax2.add_patch(step5_box)
ax2.text(5, 2.3, '5. Tool returns: "2"', fontsize=11, ha="center", fontweight="bold")

# Arrow down
arrow = FancyArrowPatch(
    (5, 2), (5, 1.3), arrowstyle="->", mutation_scale=25, linewidth=2, color=color_arrow
)
ax2.add_patch(arrow)

# Step 6: Response to user
step6_box = FancyBboxPatch(
    (2, 0.5),
    6,
    0.6,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor="#D5F4E6",
    linewidth=2,
)
ax2.add_patch(step6_box)
ax2.text(
    5,
    0.8,
    '6. Agent responds: "The answer is 2"',
    fontsize=11,
    ha="center",
    fontweight="bold",
)

plt.tight_layout()
out_path2 = assets_dir / "workflow_example.png"
plt.savefig(
    out_path2,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print(f"✅ Workflow diagram saved to: {out_path2}")
plt.close()

print("\n📊 Both diagrams generated successfully!")
print(f"   1. {out_path} - System architecture overview")
print(f"   2. {out_path2} - Example request flow")
