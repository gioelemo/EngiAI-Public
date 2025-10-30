"""
CLI agent for executing local command-line tools.

This agent specializes in running command-line applications like PrusaSlicer,
mesh processing tools, file converters, and other CLI utilities.

Uses LangGraph's interrupt_before mechanism for human-in-the-loop confirmation.
"""

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from config import config
from src.models.state import MessagesState
from src.tools.cli import (
    check_cli_tool_available,
    execute_cli_command,
    get_prusa_slicer_path,
    list_directory_contents,
    open_gui_application,
)
from src.utils.prompts import CLI_AGENT_SYSTEM_PROMPT


class CLIAgent:
    """Agent specialized in executing local command-line tools."""

    def __init__(
        self, model_name: str | None = None, require_confirmation: bool = True
    ):
        """Initialize the CLI agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            require_confirmation: Whether to require user confirmation before executing commands (default: True)
        """
        self.model_name = model_name or config.llm_model
        self.llm = init_chat_model(self.model_name)
        self.require_confirmation = require_confirmation

        # Set up CLI tools
        self.tools = [
            execute_cli_command,
            check_cli_tool_available,
            list_directory_contents,
            open_gui_application,
            get_prusa_slicer_path,
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
            SystemMessage(content=CLI_AGENT_SYSTEM_PROMPT)
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

                # Execute the tool directly - interrupt_before handles confirmation
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

        # Compile with checkpointer for conversation memory
        checkpointer = InMemorySaver()

        # Use interrupt_before for human-in-the-loop confirmation
        # When interrupt_before is set, the graph will pause before executing tool_node
        # and return control to the caller, allowing for confirmation
        if self.require_confirmation:
            return agent_builder.compile(
                checkpointer=checkpointer,
                interrupt_before=[
                    "tool_node"
                ],  # Pause before executing ANY tool for confirmation
            )
        else:
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
