"""
CLI agent for executing local command-line tools.

This agent specializes in running command-line applications like PrusaSlicer,
mesh processing tools, file converters, and other CLI utilities.
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
    list_directory_contents,
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
        # If we have a pending command, skip LLM and go directly to tool execution
        if state.get("pending_command"):
            # The user's response is the last message, route to tool_node
            # We don't need to call the LLM, just pass through
            return {}

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

        # Check if we have a pending command that needs user confirmation response
        if state.get("pending_command"):
            pending = state["pending_command"]

            # Check the last user message for confirmation
            user_message = None
            for msg in reversed(state["messages"]):
                if (
                    hasattr(msg, "type")
                    and msg.type == "human"
                    and isinstance(msg.content, str)
                ):
                    # msg.content can be str or list, ensure we handle it properly
                    user_message = msg.content.lower().strip()
                    break

            # Execute or cancel based on user response
            if user_message and user_message in [
                "yes",
                "y",
                "confirm",
                "ok",
                "proceed",
            ]:
                # User confirmed, execute the command
                tool = self.tools_by_name["execute_cli_command"]
                observation = tool.invoke(pending["args"])
                message_content = f"✅ Command executed:\n\n{observation}"
            else:
                # User rejected or gave invalid response
                message_content = (
                    f"❌ Command execution cancelled: `{pending['command']}`"
                )

            # Return as AIMessage since we're responding to user, not a tool call
            return {
                "messages": [AIMessage(content=message_content)],
                "pending_command": None,
            }

        # Only AIMessage has tool_calls
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool = self.tools_by_name[tool_call["name"]]

                # If it's the execute_cli_command tool and confirmation is required, ask for it
                if (
                    tool_call["name"] == "execute_cli_command"
                    and self.require_confirmation
                ):
                    command = tool_call["args"].get("command", "")
                    working_dir = tool_call["args"].get(
                        "working_dir", "current directory"
                    )

                    # Return a ToolMessage saying we need confirmation, then let LLM ask user
                    confirmation_needed = f"CONFIRMATION_REQUIRED|Command: {command}|WorkingDir: {working_dir}"

                    result.append(
                        ToolMessage(
                            content=confirmation_needed, tool_call_id=tool_call["id"]
                        )
                    )
                    # Store pending command for when user responds
                    return {
                        "messages": result,
                        "pending_command": {
                            "command": command,
                            "args": tool_call["args"],
                            "tool_call_id": tool_call["id"],
                        },
                    }
                else:
                    # Execute other tools without confirmation
                    observation = tool.invoke(tool_call["args"])
                    result.append(
                        ToolMessage(
                            content=str(observation), tool_call_id=tool_call["id"]
                        )
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

    def _route_after_llm(self, state: MessagesState) -> Literal["tool_node", "__end__"]:
        """Route after LLM call - handle both normal tool calls and pending confirmations.

        Args:
            state: Current conversation state

        Returns:
            Next node to execute
        """
        # If we have a pending command and user just responded, go to tool_node
        if state.get("pending_command"):
            return "tool_node"

        # Otherwise use normal routing
        return self._should_continue(state)

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
            "llm_call", self._route_after_llm, ["tool_node", END]
        )
        agent_builder.add_edge("tool_node", "llm_call")

        # Compile with checkpointer for conversation memory
        checkpointer = InMemorySaver()
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
