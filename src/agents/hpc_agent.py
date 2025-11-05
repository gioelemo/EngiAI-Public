"""
HPC cluster agent for job submission and monitoring.

This agent specializes in managing SLURM jobs on HPC clusters using SSH connections.
It handles job submission, status monitoring, job cancellation, and output retrieval.
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from config import config
from src.checkpoint import get_checkpointer
from src.models.state import MessagesState
from src.tools.hpc import (
    cancel_slurm_job,
    download_job_outputs,
    get_slurm_job_status,
    submit_slurm_job,
    test_hpc_connection,
)
from src.tools.job_monitor import (
    check_job_status_change,
    get_active_jobs_summary,
    monitor_job_until_complete,
)
from src.utils.prompts import HPC_AGENT_SYSTEM_PROMPT


class HPCAgent:
    """Agent specialized in HPC cluster job management."""

    def __init__(self, model_name: str | None = None):
        """Initialize the HPC agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        self.model_name = model_name or config.llm_model
        self.llm = init_chat_model(self.model_name)

        # Set up HPC tools
        self.tools = [
            test_hpc_connection,
            submit_slurm_job,
            get_slurm_job_status,
            cancel_slurm_job,
            download_job_outputs,
            monitor_job_until_complete,
            check_job_status_change,
            get_active_jobs_summary,
        ]

        # Create a mapping of tool names to tools
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create the agent graph
        self.graph = self._create_graph()
        self.app = self.graph.compile(
            checkpointer=get_checkpointer(),
        )

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph graph for the HPC agent."""
        agent_builder = StateGraph(MessagesState)

        # Add nodes
        agent_builder.add_node("agent", self._agent_node)
        agent_builder.add_node("tools", self._tools_node)

        # Add edges
        agent_builder.add_edge(START, "agent")
        agent_builder.add_conditional_edges(
            "agent", self._should_continue, ["tools", END]
        )
        agent_builder.add_edge("tools", "agent")

        return agent_builder

    def _agent_node(self, state: MessagesState) -> MessagesState:
        """Agent node that calls the LLM."""
        messages: list = [SystemMessage(content=HPC_AGENT_SYSTEM_PROMPT)] + state[
            "messages"
        ]
        return {"messages": [self.llm_with_tools.invoke(messages)]}

    def _tools_node(self, state: MessagesState) -> dict:
        """Tools node that executes tool calls."""
        result = []
        last_message = state["messages"][-1]

        # Only AIMessage has tool_calls
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool = self.tools_by_name[tool_call["name"]]
                observation = tool.invoke(tool_call["args"])
                result.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
                )

        return {"messages": result}

    def _should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
        """Determine if the agent should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, continue to tools node
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        # Otherwise, end
        return "__end__"

    def invoke(
        self,
        state: MessagesState,
        config: dict | None = None,
    ) -> MessagesState:
        """Invoke the agent with a given state.

        Args:
            state: Input conversation state
            config: Optional configuration (e.g., thread_id for conversation tracking)

        Returns:
            Updated conversation state
        """
        return self.app.invoke(state, config)  # type: ignore[arg-type, return-value]
