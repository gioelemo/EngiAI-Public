"""
Prusa 3D Printer agent with MCP server integration.

This agent specializes in interacting with Prusa Connect to manage
3D printers, monitor print jobs, and control printer operations.
"""

import asyncio
import concurrent.futures
import contextlib
import logging
import os
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, create_model

from config import config
from prusa_mcp_server.client import PrusaMCPClient
from src.checkpoint import get_checkpointer
from src.models.state import MessagesState
from src.utils.prompts import PRUSA_AGENT_SYSTEM_PROMPT

# Try to import nest_asyncio for better async compatibility
try:
    import nest_asyncio  # type: ignore[import-not-found]

    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False


class PrusaAgent:
    """Agent specialized in 3D printer management via Prusa Connect."""

    def __init__(
        self,
        model_name: str | None = None,
        skip_mcp: bool = False,
        temperature: float | None = None,
    ):
        """Initialize the Prusa agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            skip_mcp: Skip MCP server initialization (useful for testing)
            temperature: Model temperature (defaults to config.llm_temperature)
        """
        self.model_name = model_name or config.llm_model
        self.temperature = (
            temperature if temperature is not None else config.llm_temperature
        )
        self.llm = init_chat_model(self.model_name, temperature=self.temperature)

        # Check if MCP should be skipped (for testing or when not available)
        skip_mcp = skip_mcp or os.getenv("SKIP_MCP", "false").lower() == "true"

        if skip_mcp:
            # Initialize with empty tools for testing
            self.tools = []
            self.tools_by_name = {}
            self.llm_with_tools = self.llm
            self.mcp_client = None
        else:
            # Set up HTTP connection to external Prusa MCP server
            mcp_server_url = os.getenv("PRUSA_MCP_URL", "http://localhost:8765")
            self.mcp_client = PrusaMCPClient(server_url=mcp_server_url)

            # Initialize tools synchronously
            self.tools = self._load_tools_sync()
            self.tools_by_name = {tool.name: tool for tool in self.tools}
            self.llm_with_tools = self.llm.bind_tools(self.tools)  # type: ignore[assignment]

        # Build the agent graph
        self.agent = self._build_agent()

    def _call_mcp_tool_sync(self, tool_name: str, **kwargs):
        """Call an MCP tool synchronously using the HTTP client."""
        logger = logging.getLogger(__name__)

        if not self.mcp_client:
            return "MCP client not initialized"

        logger.info(f"Calling MCP tool '{tool_name}' with kwargs: {kwargs}")
        result = self.mcp_client.call_tool_sync(tool_name, **kwargs)
        logger.info(
            f"MCP tool '{tool_name}' returned: {result[:200] if result else '(empty)'}"
        )
        return result

    def _load_tools_sync(self):  # noqa: PLR0912, PLR0915
        """Load MCP tools synchronously using the HTTP client."""
        if not self.mcp_client:
            return []

        async def _load():
            """Load tools but DON'T disconnect - keep connection alive for tool calls."""
            try:
                tools = await self.mcp_client.list_tools()
            except Exception:
                # Ensure we disconnect even if there's an error
                with contextlib.suppress(Exception):
                    await self.mcp_client.disconnect()
                raise
            else:
                # DON'T disconnect here - we need the connection for subsequent tool calls
                # The connection will be reused for actual tool invocations
                return tools

        # Get tool list - try to use existing event loop or create new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = None
        except RuntimeError:
            loop = None

        if loop is None:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            mcp_tools = loop.run_until_complete(_load())
        elif loop.is_running():
            # We're in an async context, need to handle this differently
            if NEST_ASYNCIO_AVAILABLE:
                nest_asyncio.apply(loop)
                mcp_tools = loop.run_until_complete(_load())
            else:
                # Fallback: run in new thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(_load()))
                    mcp_tools = future.result()
        else:
            # Loop exists but not running
            mcp_tools = loop.run_until_complete(_load())

        tools = []
        for mcp_tool in mcp_tools:
            # Create a wrapper function for this specific tool
            def make_tool_func(tool_name):
                def call_tool(**kwargs):
                    print(
                        f"DEBUG: call_tool wrapper called for '{tool_name}' with kwargs: {kwargs}"
                    )
                    return self._call_mcp_tool_sync(tool_name, **kwargs)

                return call_tool

            # Create Pydantic model for arguments if schema is available
            args_schema = None
            if mcp_tool.inputSchema and "properties" in mcp_tool.inputSchema:
                fields = {}
                properties = mcp_tool.inputSchema["properties"]
                required = mcp_tool.inputSchema.get("required", [])

                for prop_name, prop_schema in properties.items():
                    field_type = str  # Default to string
                    description = prop_schema.get("description", "")

                    # Map JSON schema types to Python types
                    if prop_schema.get("type") == "integer":
                        field_type = int
                    elif prop_schema.get("type") == "number":
                        field_type = float
                    elif prop_schema.get("type") == "boolean":
                        field_type = bool

                    # Create field with proper typing
                    if prop_name in required:
                        fields[prop_name] = (
                            field_type,
                            Field(..., description=description),
                        )
                    else:
                        default = prop_schema.get("default", None)
                        fields[prop_name] = (
                            field_type,
                            Field(default=default, description=description),
                        )

                # Create Pydantic model dynamically
                if fields:
                    args_schema = create_model(
                        f"{mcp_tool.name}_args", **fields, __base__=BaseModel
                    )

            # Create LangChain tool
            lc_tool = StructuredTool.from_function(
                func=make_tool_func(mcp_tool.name),
                name=mcp_tool.name,
                description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                args_schema=args_schema,
            )
            tools.append(lc_tool)

        return tools

    def _llm_call(self, state: MessagesState) -> dict:
        """LLM decides whether to call a tool or not.

        Args:
            state: Current conversation state

        Returns:
            Updated state with LLM response
        """
        messages: list[AnyMessage] = [
            SystemMessage(content=PRUSA_AGENT_SYSTEM_PROMPT)
        ] + state["messages"]

        # Count LLM calls, but reset on new user messages
        # Check if the last message is from user (new turn)
        current_calls = state.get("llm_calls", 0)
        if state["messages"] and hasattr(state["messages"][-1], "type"):
            last_msg_type = getattr(state["messages"][-1], "type", None)
            # Reset counter on new human message
            if last_msg_type == "human":
                current_calls = 0

        return {
            "messages": [self.llm_with_tools.invoke(messages)],
            "llm_calls": current_calls + 1,
        }

    def _tool_node(self, state: MessagesState) -> dict:
        """Execute tool calls from the LLM.

        Args:
            state: Current conversation state

        Returns:
            Updated state with tool results
        """
        last_message = state["messages"][-1]

        # Type guard: only AIMessage has tool_calls
        if not isinstance(last_message, AIMessage):
            return {"messages": []}

        tool_calls = last_message.tool_calls

        # Execute each tool call
        outputs: list[ToolMessage] = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Get the tool and invoke it
            tool = self.tools_by_name.get(tool_name)
            if not tool:
                outputs.append(
                    ToolMessage(
                        content=f"Error: Tool '{tool_name}' not found",
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            try:
                logger = logging.getLogger(__name__)
                print(
                    f"DEBUG _tool_node: Invoking tool '{tool_name}' with args: {tool_args}"
                )
                logger.info(f"Invoking tool '{tool_name}' with args: {tool_args}")
                result = tool.invoke(tool_args)
                print(
                    f"DEBUG _tool_node: Tool '{tool_name}' result: {result[:100] if result else '(empty)'}"
                )
                logger.info(
                    f"Tool '{tool_name}' result: {result[:200] if result else '(empty)'}"
                )
                outputs.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception(f"Error invoking tool '{tool_name}'")
                outputs.append(
                    ToolMessage(
                        content=f"Error executing {tool_name}: {e!s}",
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )

        return {"messages": outputs}

    def _should_continue(
        self, state: MessagesState
    ) -> Literal["tools", "summarize", "end"]:
        """Decide whether to continue the conversation.

        Args:
            state: Current conversation state

        Returns:
            Next node to execute: 'tools', 'summarize', or 'end'
        """
        last_message = state["messages"][-1]

        # If LLM made a tool call, execute it
        # Only AIMessage has tool_calls attribute
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        # Check if we've hit the maximum number of LLM calls in this turn
        # This prevents infinite loops while allowing reasonable tool use
        max_calls = 20  # Increased limit to allow more tool interactions
        if state.get("llm_calls", 0) >= max_calls:
            return "summarize"

        # Otherwise, we're done with this turn
        return "end"

    def _summarize_node(self, state: MessagesState) -> dict:
        """Summarize the conversation when hitting max iterations.

        Args:
            state: Current conversation state

        Returns:
            Updated state with summary message
        """
        summary_prompt = """The conversation has reached the maximum number of iterations.
Please provide a brief summary of what was accomplished and any next steps."""

        messages: list[AnyMessage] = state["messages"] + [
            SystemMessage(content=summary_prompt)
        ]

        response = self.llm.invoke(messages)

        return {"messages": [AIMessage(content=response.content)]}

    def _build_agent(self):
        """Build the agent workflow graph.

        Returns:
            Compiled agent graph
        """
        # Build the graph
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("agent", self._llm_call)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("summarize", self._summarize_node)

        # Set entry point
        workflow.add_edge(START, "agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "summarize": "summarize",
                "end": END,
            },
        )

        # After tool execution, go back to agent
        workflow.add_edge("tools", "agent")
        workflow.add_edge("summarize", END)

        # Compile with persistent checkpointer
        checkpointer = get_checkpointer()
        return workflow.compile(checkpointer=checkpointer)

    def invoke(self, state, config):
        """Invoke the Prusa agent.

        Args:
            state: Current conversation state
            config: Configuration including thread_id

        Returns:
            Updated state with agent responses
        """
        return self.agent.invoke(state, config)
