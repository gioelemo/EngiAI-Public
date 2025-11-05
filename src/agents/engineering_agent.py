"""
Engineering design agent with EngiBench integration.

This agent specializes in structural optimization and engineering design
problems using the EngiBench library.
"""

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from config import config
from src.checkpoint import get_checkpointer
from src.models.state import MessagesState
from src.tools.engibench import (
    check_beam_constraints,
    create_beam_problem,
    get_dataset_info,
    get_problem_details,
    get_problem_info,
    optimize_beam_design,
    render_beam_design,
    simulate_beam_design,
)
from src.tools.engiopt import (
    download_wandb_model,
    generate_training_command,
    list_available_algorithms,
    load_wandb_model,
    sample_designs_from_model,
)
from src.tools.stl_export import convert_design_to_stl
from src.utils.prompts import ENGINEERING_AGENT_SYSTEM_PROMPT


class EngineeringAgent:
    """Agent specialized in engineering design and structural optimization."""

    def __init__(self, model_name: str | None = None):
        """Initialize the engineering agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        self.model_name = model_name or config.llm_model
        self.llm = init_chat_model(self.model_name)

        # Set up engineering tools
        self.tools = [
            # EngiBench tools
            check_beam_constraints,
            create_beam_problem,
            simulate_beam_design,
            optimize_beam_design,
            render_beam_design,
            get_problem_info,
            get_problem_details,
            get_dataset_info,
            # Export tools
            convert_design_to_stl,
            # WandB model download tools
            download_wandb_model,
            list_available_algorithms,
            load_wandb_model,
            sample_designs_from_model,
            # Model training command generator
            generate_training_command,
        ]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the agent graph
        self.agent = self._build_agent()

    def _llm_call(self, state: MessagesState) -> dict:
        """LLM decides whether to call a tool or not.

        Args:
            state: Current conversation state

        Returns:
            Updated state with LLM response
        """
        messages: list[AnyMessage] = [
            SystemMessage(content=ENGINEERING_AGENT_SYSTEM_PROMPT)
        ] + state["messages"]

        return {
            "messages": [self.llm_with_tools.invoke(messages)],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def _tool_node(self, state: MessagesState) -> dict:
        """Execute tool calls from the LLM.

        Args:
            state: Current conversation state

        Returns:
            Updated state with tool results
        """
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

    def _should_continue(self, state: MessagesState) -> Literal["tool_node", "__end__"]:
        """Decide whether to continue to tool execution or end.

        Args:
            state: Current conversation state

        Returns:
            Next node to execute ("tool_node" or "__end__")
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return "__end__"

    def _build_agent(self) -> Any:
        """Build the agent workflow graph.

        Returns:
            Compiled agent graph
        """
        # Build workflow
        agent_builder = StateGraph(MessagesState)

        # Add nodes
        agent_builder.add_node("llm_call", self._llm_call)
        agent_builder.add_node("tool_node", self._tool_node)

        # Add edges to connect nodes
        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_conditional_edges(
            "llm_call", self._should_continue, ["tool_node", END]
        )
        agent_builder.add_edge("tool_node", "llm_call")

        # Compile with persistent checkpointer for conversation memory
        checkpointer = get_checkpointer()
        return agent_builder.compile(checkpointer=checkpointer)

    def invoke(self, state: MessagesState, config: dict | None = None) -> MessagesState:
        """Invoke the agent with a given state.

        Args:
            state: Input conversation state
            config: Optional configuration (e.g., thread_id for conversation tracking)

        Returns:
            Updated conversation state
        """
        return self.agent.invoke(state, config)  # type: ignore[return-value]
