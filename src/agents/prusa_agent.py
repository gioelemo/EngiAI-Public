"""
Prusa 3D Printer agent with MCP server integration.

This agent specializes in interacting with Prusa Connect to manage
3D printers, monitor print jobs, and control printer operations.
"""

import asyncio
import os
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, create_model

from config import config
from src.models.state import MessagesState
from src.utils.prompts import PRUSA_AGENT_SYSTEM_PROMPT


class PrusaAgent:
    """Agent specialized in 3D printer management via Prusa Connect."""

    def __init__(self, model_name: str | None = None):
        """Initialize the Prusa agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        self.model_name = model_name or config.llm_model
        self.llm = init_chat_model(self.model_name)

        # Set up MCP connection to Prusa server
        prusa_mcp_path = os.getenv(
            "PRUSA_MCP_PATH", str(Path.home() / "Desktop" / "prusa-mcp")
        )
        uv_path = os.getenv("UV_PATH", str(Path.home() / ".local" / "bin" / "uv"))

        # MCP server parameters
        self.server_params = StdioServerParameters(
            command=uv_path,
            args=[
                "--directory",
                prusa_mcp_path,
                "run",
                "src/prusa-mcp.py",
            ],
        )

        # Initialize tools synchronously
        self.tools = self._load_tools_sync()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the agent graph
        self.agent = self._build_agent()

    def _call_mcp_tool_sync(self, tool_name: str, **kwargs):
        """Call an MCP tool synchronously by creating a new session."""

        async def _call():
            async with (
                stdio_client(self.server_params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(tool_name, kwargs)
                if result.content:
                    return "\n".join(
                        [
                            c.text if hasattr(c, "text") else str(c)
                            for c in result.content
                        ]
                    )
                return "Tool executed successfully"

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_call())
        finally:
            loop.close()

    def _load_tools_sync(self):
        """Load MCP tools synchronously."""

        async def _load():
            async with (
                stdio_client(self.server_params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                response = await session.list_tools()
                return response.tools

        # Get tool list
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mcp_tools = loop.run_until_complete(_load())
        finally:
            loop.close()

        tools = []
        for mcp_tool in mcp_tools:
            # Create a wrapper function for this specific tool
            def make_tool_func(tool_name):
                def call_tool(**kwargs):
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
                result = tool.invoke(tool_args)
                outputs.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
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

        # Compile with memory
        memory = InMemorySaver()
        return workflow.compile(checkpointer=memory)

    def invoke(self, state, config):
        """Invoke the Prusa agent.

        Args:
            state: Current conversation state
            config: Configuration including thread_id

        Returns:
            Updated state with agent responses
        """
        return self.agent.invoke(state, config)
