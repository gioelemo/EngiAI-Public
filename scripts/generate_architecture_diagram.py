"""
Generate a visual diagram of the multi-agent system architecture.
"""

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
fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis("off")

# Color scheme
color_user = "#E8F4F8"
color_supervisor = "#FFE5B4"
color_agent = "#D4E6F1"
color_tool = "#E8DAEF"
color_arrow = "#34495E"

# Tool category colors
tool_colors = {
    "engibench": "#FF6B6B",  # Red - EngiBench tools
    "engiopt": "#51CF66",  # Green - WandB/engiopt tools
    "stl": "#4DABF7",  # Blue - STL export tools
    "code": "#F08C00",  # Orange - Code execution tools (changed from yellow for readability)
    "search": "#9775FA",  # Purple - Search tools
    "hpc": "#FF9F1C",  # Gold/Orange - HPC tools
    "mcp": "#868E96",  # Gray - MCP tools (planned)
}

# Tool category mapping
tool_categories = {
    # EngiBench tools
    "create_beam_problem": "engibench",
    "optimize_beam_design": "engibench",
    "simulate_beam_design": "engibench",
    "render_beam_design": "engibench",
    "check_beam_constraints": "engibench",
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
    # Code execution tools
    "execute_python_code": "code",
    "execute_python_expression": "code",
    # Search tools
    "TavilySearch": "search",
    # HPC tools
    "test_hpc_connection": "hpc",
    "submit_slurm_job": "hpc",
    "get_slurm_job_status": "hpc",
    "cancel_slurm_job": "hpc",
    "download_job_outputs": "hpc",
    # MCP tools (planned)
    "print_document": "mcp",
    "export_pdf": "mcp",
    "format_report": "mcp",
}

# Title
ax.text(
    7,
    11.5,
    "Multi-Agent Engineer Assistant System",
    fontsize=20,
    fontweight="bold",
    ha="center",
)

# User
user_box = FancyBboxPatch(
    (5.5, 9.8),
    3,
    0.8,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_user,
    linewidth=2,
)
ax.add_patch(user_box)
ax.text(7, 10.2, "User", fontsize=14, ha="center", fontweight="bold")

# Supervisor Agent
supervisor_box = FancyBboxPatch(
    (4.5, 7.8),
    5,
    1.2,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_supervisor,
    linewidth=3,
)
ax.add_patch(supervisor_box)
ax.text(7, 8.65, "Supervisor Agent", fontsize=14, ha="center", fontweight="bold")
ax.text(
    7,
    8.25,
    "Routes requests to specialized agents",
    fontsize=9,
    ha="center",
    style="italic",
)

# Arrow from user to supervisor
arrow1 = FancyArrowPatch(
    (7, 9.8),
    (7, 9.0),
    arrowstyle="->",
    mutation_scale=30,
    linewidth=2,
    color=color_arrow,
)
ax.add_patch(arrow1)
ax.text(7.3, 9.4, "Query", fontsize=9)

# Arrow from supervisor back to user
arrow2 = FancyArrowPatch(
    (6.7, 9.0),
    (6.7, 9.8),
    arrowstyle="->",
    mutation_scale=30,
    linewidth=2,
    color=color_arrow,
    linestyle="--",
)
ax.add_patch(arrow2)
ax.text(6.0, 9.4, "Response", fontsize=9)

# Agent position thresholds for arrow routing
LEFT_AGENT_THRESHOLD = 5.0  # Agent x-position below this is considered "left"
RIGHT_AGENT_THRESHOLD = 9.0  # Agent x-position above this is considered "right"

# Specialized Agents
agents: list[AgentInfo] = [
    {
        "name": "Code Execution Agent (Currently Disabled)",
        "icon": "",
        "x": 0.2,
        "y": 5.0,
        "tools": ["execute_python_code", "execute_python_expression"],
        "desc": "Python REPL, calculations",
    },
    {
        "name": "Engineering Agent",
        "icon": "",
        "x": 3.0,
        "y": 2.0,
        "tools": [
            # EngiBench tools (red)
            "create_beam_problem",
            "optimize_beam_design",
            "simulate_beam_design",
            "render_beam_design",
            "check_beam_constraints",
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
        "x": 5.8,
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
        "name": "Printer Agent (MCP)",
        "icon": "",
        "x": 8.8,
        "y": 2.0,
        "tools": ["printer status", "slice stl", "print file"],
        "desc": "Document generation and printing (planned)",
        "planned": True,
    },
    {
        "name": "Search Agent",
        "icon": "",
        "x": 11.3,
        "y": 5.0,
        "tools": ["TavilySearch"],
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
    # Supervisor box: x=4.5-9.5, y=7.8-9.0
    # Calculate proper edge connection points
    supervisor_x = 4.5
    supervisor_width = 5
    supervisor_y = 7.8
    supervisor_bottom_y = 7.8

    agent_box_x = agent["x"]
    agent_box_width = 2.5
    agent_center_x = agent_box_x + agent_box_width / 2
    agent_top_y = agent["y"] + box_height

    # Determine supervisor exit point based on agent position
    SUPERVISOR_CENTER_X = 7.0  # Center x-position of supervisor
    ENGINEERING_AGENT_THRESHOLD = 5.5  # X position threshold for Engineering agent
    HPC_AGENT_THRESHOLD = 8.5  # X position threshold for HPC agent

    # For side agents (Code Execution and Search), use angled arrows
    if agent_center_x < LEFT_AGENT_THRESHOLD:  # Left agents (Code Execution)
        supervisor_exit_x = supervisor_x + 0.3
        supervisor_exit_y = supervisor_y + 0.5
    elif agent_center_x > RIGHT_AGENT_THRESHOLD:  # Right agents (Search)
        supervisor_exit_x = supervisor_x + supervisor_width - 0.3
        supervisor_exit_y = supervisor_y + 0.5
    else:  # Center agents (Engineering, HPC, Printer)
        # Distribute connection points across bottom of supervisor based on agent position
        # Engineering: ~4.25, HPC: ~7.05, Printer: ~10.05
        if agent_center_x < ENGINEERING_AGENT_THRESHOLD:  # Engineering Agent
            supervisor_exit_x = supervisor_x + supervisor_width * 0.2
        elif agent_center_x < HPC_AGENT_THRESHOLD:  # HPC Agent
            supervisor_exit_x = supervisor_x + supervisor_width * 0.5
        else:  # Printer Agent
            supervisor_exit_x = supervisor_x + supervisor_width * 0.8
        supervisor_exit_y = supervisor_bottom_y

    # Adjust connection style based on agent position
    arrow_linestyle = "--" if is_planned else "-"
    arrow_color = "gray" if is_planned else color_arrow

    if agent_center_x < LEFT_AGENT_THRESHOLD or agent_center_x > RIGHT_AGENT_THRESHOLD:
        # Angled arrows for side agents
        arrow = FancyArrowPatch(
            (supervisor_exit_x, supervisor_exit_y),
            (agent_center_x, agent_top_y),
            arrowstyle="<->",
            mutation_scale=20,
            linewidth=1.5,
            color=arrow_color,
            connectionstyle="arc3,rad=0.2",
            linestyle=arrow_linestyle,
        )
    else:
        # Straight arrow for center agent
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

# Add legend for tool colors in bottom left corner
legend_x = 0.5
legend_y = 3.5
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
    ("Code Exec", "code"),
    ("Search", "search"),
    ("MCP (planned)", "mcp"),
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
plt.savefig(
    "outputs/agent_architecture.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print("✅ Architecture diagram saved to: outputs/agent_architecture.png")
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
    '4. Calls tool: execute_python_expression("1+1")',
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
plt.savefig(
    "outputs/workflow_example.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print("✅ Workflow diagram saved to: outputs/workflow_example.png")
plt.close()

print("\n📊 Both diagrams generated successfully!")
print("   1. outputs/agent_architecture.png - System architecture overview")
print("   2. outputs/workflow_example.png - Example request flow")
