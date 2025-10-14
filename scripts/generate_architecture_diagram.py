"""
Generate a visual diagram of the multi-agent system architecture.
"""

from typing import TypedDict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


class AgentInfo(TypedDict):
    """Type definition for agent information."""

    name: str
    icon: str
    x: float
    y: float
    tools: list[str]
    desc: str


# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(2, 10)
ax.axis("off")

# Color scheme
color_user = "#E8F4F8"
color_supervisor = "#FFE5B4"
color_agent = "#D4E6F1"
color_tool = "#E8DAEF"
color_arrow = "#34495E"

# Title
ax.text(
    5,
    9.5,
    "Multi-Agent Engineer Assistant System",
    fontsize=20,
    fontweight="bold",
    ha="center",
)

# User
user_box = FancyBboxPatch(
    (4, 8.5),
    2,
    0.6,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_user,
    linewidth=2,
)
ax.add_patch(user_box)
ax.text(5, 8.8, "User", fontsize=12, ha="center", fontweight="bold")

# Supervisor Agent
supervisor_box = FancyBboxPatch(
    (3.5, 6.8),
    3,
    1,
    boxstyle="round,pad=0.1",
    edgecolor="black",
    facecolor=color_supervisor,
    linewidth=3,
)
ax.add_patch(supervisor_box)
ax.text(5, 7.5, "Supervisor Agent", fontsize=14, ha="center", fontweight="bold")
ax.text(
    5,
    7.15,
    "Routes requests to specialized agents",
    fontsize=9,
    ha="center",
    style="italic",
)

# Arrow from user to supervisor
arrow1 = FancyArrowPatch(
    (5, 8.5),
    (5, 7.8),
    arrowstyle="->",
    mutation_scale=30,
    linewidth=2,
    color=color_arrow,
)
ax.add_patch(arrow1)
ax.text(5.3, 8.15, "Query", fontsize=9)

# Arrow from supervisor back to user
arrow2 = FancyArrowPatch(
    (4.8, 7.8),
    (4.8, 8.5),
    arrowstyle="->",
    mutation_scale=30,
    linewidth=2,
    color=color_arrow,
    linestyle="--",
)
ax.add_patch(arrow2)
ax.text(4.2, 8.15, "Response", fontsize=9)

# Specialized Agents
agents: list[AgentInfo] = [
    {
        "name": "Code Execution Agent",
        "icon": "",
        "x": 0.5,
        "y": 3.5,
        "tools": ["execute_python_code", "execute_python_expression"],
        "desc": "Python REPL, calculations",
    },
    {
        "name": "Engineering Agent",
        "icon": "",
        "x": 3.75,
        "y": 2.8,
        "tools": [
            "create_beam_problem",
            "optimize_beam_design",
            "simulate_beam_design",
            "render_beam_design",
            "convert_design_to_stl",
            "check_beam_constraints",
            "get_problem_info",
            "get_problem_details",
            "get_dataset_info",
        ],
        "desc": "Structural optimization, EngiBench",
    },
    {
        "name": "Search Agent",
        "icon": "",
        "x": 7,
        "y": 3.5,
        "tools": ["TavilySearch"],
        "desc": "Web research, information gathering",
    },
]

# Draw agents and their connections
for agent in agents:
    # Calculate box height based on number of tools
    num_tools = len(agent["tools"])
    box_height = 1.2 + (num_tools * 0.18)  # Dynamic height based on tools

    # Agent box
    agent_box = FancyBboxPatch(
        (agent["x"], agent["y"]),
        2.5,
        box_height,
        boxstyle="round,pad=0.1",
        edgecolor="black",
        facecolor=color_agent,
        linewidth=2,
    )
    ax.add_patch(agent_box)

    # Agent name
    agent_name = agent["name"]
    if agent["icon"]:
        agent_name = f"{agent['icon']} {agent_name}"
    ax.text(
        agent["x"] + 1.25,
        agent["y"] + box_height - 0.2,
        agent_name,
        fontsize=11,
        ha="center",
        fontweight="bold",
    )

    # Agent description
    ax.text(
        agent["x"] + 1.25,
        agent["y"] + box_height - 0.5,
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
        ax.text(
            agent["x"] + 0.1,
            tools_y_start - (i * 0.18),
            f"• {tool}",
            fontsize=7,
            ha="left",
            family="monospace",
        )

    # Arrow from supervisor to agent
    # Supervisor box: x=3.5-6.5, y=6.8-7.8
    # Calculate proper edge connection points
    supervisor_x = 3.5
    supervisor_width = 3
    supervisor_y = 6.8

    agent_box_x = agent["x"]
    agent_box_width = 2.5
    agent_center_x = agent_box_x + agent_box_width / 2
    agent_top_y = agent["y"] + box_height

    # Determine supervisor exit point based on agent position
    SUPERVISOR_CENTER_X = 5.0  # Center x-position of supervisor
    if agent_center_x < SUPERVISOR_CENTER_X:  # Left agent (Code Execution)
        supervisor_exit_x = supervisor_x + supervisor_width * 0.3
    elif agent_center_x > SUPERVISOR_CENTER_X:  # Right agent (Search)
        supervisor_exit_x = supervisor_x + supervisor_width * 0.7
    else:  # Center agent (Engineering)
        supervisor_exit_x = supervisor_x + supervisor_width * 0.5

    arrow = FancyArrowPatch(
        (supervisor_exit_x, supervisor_y),
        (agent_center_x, agent_top_y),
        arrowstyle="<->",
        mutation_scale=20,
        linewidth=1.5,
        color=color_arrow,
    )
    ax.add_patch(arrow)

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
