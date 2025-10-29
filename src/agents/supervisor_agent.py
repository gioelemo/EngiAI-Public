"""
Supervisor agent that coordinates multiple specialized agents.

This agent uses a hierarchical approach - it delegates tasks to specialized
sub-agents (Engineering, Search, etc.) rather than having all tools directly.
"""

from typing import Annotated, cast

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import config
from src.agents.cli_agent import CLIAgent
from src.agents.code_execution_agent import CodeExecutionAgent
from src.agents.engineering_agent import EngineeringAgent
from src.agents.hpc_agent import HPCAgent
from src.agents.search_agent import SearchAgent
from src.models.state import MessagesState


class SupervisorState(TypedDict):
    """State for the supervisor agent."""

    messages: Annotated[list, add_messages]
    next: str


class SupervisorAgent:
    """
    Supervisor agent that coordinates specialized agents.

    Uses a hierarchical delegation model where the supervisor routes
    tasks to appropriate specialized agents based on the request.
    """

    def __init__(self, model_name: str | None = None):
        """Initialize the supervisor agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        self.model_name = model_name or config.llm_model
        self.llm = init_chat_model(self.model_name)

        # Initialize specialized sub-agents
        self.enable_code_execution = config.enable_code_execution_agent
        if self.enable_code_execution:
            self.code_execution_agent = CodeExecutionAgent(model_name=self.model_name)
        self.engineering_agent = EngineeringAgent(model_name=self.model_name)
        self.hpc_agent = HPCAgent(model_name=self.model_name)
        self.search_agent = SearchAgent(model_name=self.model_name)
        # Disable CLI agent's internal confirmation - supervisor handles interrupts at its level
        self.cli_agent = CLIAgent(
            model_name=self.model_name, require_confirmation=False
        )

        # Build the supervisor graph
        self.graph = self._build_graph()

    def _build_routing_prompt(self) -> str:
        """Build the routing system prompt based on available agents."""
        available_agents = []
        agent_names = []

        if self.enable_code_execution:
            available_agents.append(
                "- code_execution_agent: Python code execution, calculations, data analysis"
            )
            agent_names.append("'code_execution_agent'")

        available_agents.append(
            "- engineering_agent: Structural optimization, beam design, topology optimization, STL conversion, downloading/using pre-trained models from WandB, generative models (GANs, Diffusion)"
        )
        available_agents.append(
            "- hpc_agent: HPC cluster job management, SLURM job submission, job monitoring, output retrieval"
        )
        available_agents.append("- search_agent: Web research and finding information")
        available_agents.append(
            "- cli_agent: Execute local CLI commands (PrusaSlicer, mesh processing, file conversion, any command-line tool)"
        )
        agent_names.extend(
            ["'engineering_agent'", "'hpc_agent'", "'search_agent'", "'cli_agent'"]
        )

        return (
            "You are a supervisor that routes tasks to specialized agents.\n\n"
            + "Available agents:\n"
            + "\n".join(available_agents)
            + "\n\n"
            + "IMPORTANT ROUTING RULES:\n"
            + "- Any mention of 'wandb', 'models', 'pretrained', 'download model', 'GAN', 'diffusion', "
            + "'beam', 'beam2d', 'optimization', 'design', 'algorithm', 'checkpoint', 'training', 'train', 'generate script', 'generate SLURM' → use engineering_agent\n"
            + "- HPC cluster job management ONLY: submit job, job submission, job status, check job, monitor job, cancel job, download output, 'euler' cluster operations → use hpc_agent\n"
            + "- Web search, research, finding information → use search_agent\n"
            + "- CLI commands, slicing STL files, PrusaSlicer, mesh processing, file conversion, running local command-line tools → use cli_agent\n"
            + (
                "- Python code execution, calculations → use code_execution_agent\n"
                if self.enable_code_execution
                else ""
            )
            + "\n"
            + "Respond with ONLY ONE WORD: "
            + ", ".join(agent_names[:-1])
            + (" or " if len(agent_names) > 1 else "")
            + agent_names[-1]
            + ".\n"
            + "No explanations, no other text, just the agent name."
        )

    def _supervisor_node(self, state: SupervisorState):
        """Supervisor decides which agent should act next."""
        # Only route if we haven't routed yet (no next value set)
        if state.get("next") and state["next"] != "":
            # Already routed, finish
            return {"next": "FINISH"}

        messages = [
            {"role": "system", "content": self._build_routing_prompt()},
            *state["messages"],
        ]
        response = self.llm.invoke(messages)

        # Extract routing decision from response
        content = str(response.content).strip().lower()

        # Determine next agent - check engineering first (more specific keywords)
        next_agent = "FINISH"
        if self.enable_code_execution and (
            "code_execution" in content or "code" in content
        ):
            next_agent = "code_execution_agent"
        elif (
            "engineering" in content
            or "training" in content
            or "generate" in content
            or "wandb" in content
        ):
            # Engineering handles: training scripts, model generation, design tasks, WandB operations
            next_agent = "engineering_agent"
        elif (
            "hpc" in content
            or "submit" in content
            or "status" in content
            or "cancel" in content
            or "monitor" in content
            or "download output" in content
        ):
            # HPC handles: job submission, monitoring, cancellation, output retrieval
            next_agent = "hpc_agent"
        elif "search" in content:
            next_agent = "search_agent"
        elif "cli" in content or "command" in content:
            # CLI handles: local command-line tool execution
            next_agent = "cli_agent"

        return {"next": next_agent}

    def _code_execution_node(self, state: SupervisorState):
        """Delegate to code execution agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.code_execution_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "code_execution"}},
        )
        return {"messages": [result["messages"][-1]], "next": "FINISH"}

    def _engineering_node(self, state: SupervisorState):
        """Delegate to engineering agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.engineering_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "engineering"}},
        )
        return {"messages": [result["messages"][-1]], "next": "FINISH"}

    def _hpc_node(self, state: SupervisorState):
        """Delegate to HPC agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.hpc_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "hpc"}},
        )
        return {"messages": [result["messages"][-1]], "next": "FINISH"}

    def _search_node(self, state: SupervisorState):
        """Delegate to search agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.search_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "search"}},
        )
        return {"messages": [result["messages"][-1]], "next": "FINISH"}

    def _cli_node(self, state: SupervisorState):
        """Delegate to CLI agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.cli_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "cli"}},
        )
        return {"messages": [result["messages"][-1]], "next": "FINISH"}

    def _build_graph(self):
        """Build the supervisor workflow graph with agent routing."""

        # Build the graph
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        if self.enable_code_execution:
            workflow.add_node("code_execution_agent", self._code_execution_node)
        workflow.add_node("engineering_agent", self._engineering_node)
        workflow.add_node("hpc_agent", self._hpc_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("cli_agent", self._cli_node)

        # Add edges - start with supervisor
        workflow.add_edge(START, "supervisor")

        # Build conditional routing dictionary
        routing_dict = {
            "engineering_agent": "engineering_agent",
            "hpc_agent": "hpc_agent",
            "search_agent": "search_agent",
            "cli_agent": "cli_agent",
            "FINISH": END,
        }
        if self.enable_code_execution:
            routing_dict["code_execution_agent"] = "code_execution_agent"

        # Conditional routing based on supervisor decision
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            routing_dict,
        )

        # Agents go directly to END (no looping back to supervisor)
        if self.enable_code_execution:
            workflow.add_edge("code_execution_agent", END)
        workflow.add_edge("engineering_agent", END)
        workflow.add_edge("hpc_agent", END)
        workflow.add_edge("search_agent", END)
        workflow.add_edge("cli_agent", END)

        # Compile with memory
        memory = InMemorySaver()
        # Interrupt before CLI agent so we can check its planned commands
        return workflow.compile(checkpointer=memory, interrupt_before=["cli_agent"])

    def invoke(self, state, config):
        """Invoke the supervisor agent.

        Args:
            state: Current conversation state with messages, or None to resume from interrupt
            config: Configuration including thread_id

        Returns:
            Updated state with agent responses
        """
        # If state is None, we're resuming from an interrupt - pass None to graph
        if state is None:
            result = self.graph.invoke(None, config)
            return {"messages": result["messages"]}

        # Convert MessagesState to SupervisorState if needed
        if "next" not in state:
            supervisor_state = {"messages": state["messages"], "next": ""}
        else:
            supervisor_state = state

        result = self.graph.invoke(supervisor_state, config)

        # Return in MessagesState format for compatibility
        return {"messages": result["messages"]}
