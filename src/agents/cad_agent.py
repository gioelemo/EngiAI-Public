"""
CAD agent specialized in 3D model conversion and file format handling.

This agent handles tasks related to 3D modeling, STL conversion,
and CAD file operations.
"""

from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph

from config import config
from src.models.state import MessagesState
from src.tools.engibench import convert_design_to_stl
from src.utils.prompts import CAD_AGENT_SYSTEM_PROMPT


class CADAgent:
    """Agent specialized in CAD operations and 3D model conversion."""

    def __init__(self, model_name: str | None = None):
        """Initialize the CAD agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        self.model_name = model_name or config.llm_model
        self.llm = init_chat_model(self.model_name)

        # Set up CAD tools
        self.tools = [
            convert_design_to_stl,
        ]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the agent graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the agent workflow graph."""
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("llm", self._llm_call)
        workflow.add_node("tools", self._tool_node)

        # Add edges
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges("llm", self._should_continue)
        workflow.add_edge("tools", "llm")

        # Compile with memory
        memory = InMemorySaver()
        return workflow.compile(checkpointer=memory)

    def _llm_call(self, state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Call the LLM with system prompt and tools."""
        system_message = SystemMessage(content=CAD_AGENT_SYSTEM_PROMPT)
        messages = [system_message, *state["messages"]]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _tool_node(self, state: MessagesState) -> dict[str, list[ToolMessage]]:
        """Execute the requested tools."""
        outputs = []
        last_message = state["messages"][-1]
        # Type guard: only AIMessage has tool_calls
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                tool = self.tools_by_name[tool_call["name"]]
                result = tool.invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"messages": outputs}

    def _should_continue(self, state: MessagesState) -> Literal["tools", "__end__"]:
        """Determine whether to continue with tools or end."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "__end__"

    def invoke(self, state: MessagesState, config: dict[str, Any]) -> MessagesState:
        """Invoke the agent with the given state.

        Args:
            state: Current conversation state
            config: Configuration including thread_id

        Returns:
            Updated state with agent response
        """
        return self.graph.invoke(state, config)
