"""
Base agent class for LangGraph-based agents.

Provides common functionality for all agents including:
- LLM initialization
- Tool management
- Graph building with standard structure
- Common node implementations (_llm_call, _tool_node, _should_continue)
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from config import config
from src.checkpoint import get_checkpointer
from src.models.state import MessagesState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for LangGraph agents with common functionality."""

    def __init__(
        self,
        model_name: str | None = None,
        tools: list | None = None,
        require_confirmation: bool = False,
        temperature: float | None = None,
        seed: int | None = None,
    ):
        """Initialize the base agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            tools: List of LangChain tools for this agent
            require_confirmation: Whether to require user confirmation before tool execution
            temperature: Model temperature (defaults to config.llm_temperature)
            seed: Random seed for model (defaults to config.llm_seed, None for non-deterministic)
        """
        self.model_name = model_name or config.llm_model
        self.temperature = (
            temperature if temperature is not None else config.llm_temperature
        )
        self.seed = seed if seed is not None else config.llm_seed
        logger.debug(
            f"Initializing {self.__class__.__name__} with temperature={self.temperature}, seed={self.seed}"
        )

        # Build kwargs for model initialization
        model_kwargs: dict = {"temperature": self.temperature}

        # Add seed if provided
        if self.seed is not None:
            model_kwargs["seed"] = self.seed

        # Add Ollama-specific configuration if using an Ollama model
        if self.model_name.startswith("ollama:"):
            model_kwargs["num_ctx"] = config.ollama_num_ctx
            logger.debug(
                f"Ollama model detected, using num_ctx={config.ollama_num_ctx}"
            )

        self.llm = init_chat_model(self.model_name, **model_kwargs)
        self.require_confirmation = require_confirmation

        # Initialize tools
        self.tools = tools or self._create_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the agent graph
        self.agent = self._build_agent()

    @abstractmethod
    def _create_tools(self) -> list:
        """Create the list of tools for this agent.

        Returns:
            List of LangChain tools

        Note:
            Subclasses must implement this method to define their specific tools.
        """

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent.

        Returns:
            System prompt string

        Note:
            Subclasses must implement this method to define their system prompt.
        """

    def _llm_call(self, state: MessagesState) -> dict:
        """LLM decides whether to call a tool or not.

        Args:
            state: Current conversation state

        Returns:
            Updated state with LLM response
        """
        messages: list[AnyMessage] = [
            SystemMessage(content=self._get_system_prompt())
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
                try:
                    tool = self.tools_by_name[tool_call["name"]]
                    observation = tool.invoke(tool_call["args"])
                    result.append(
                        ToolMessage(
                            content=str(observation),
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                        )
                    )
                except Exception as e:
                    # CRITICAL: Always return a ToolMessage, even for errors
                    # Otherwise OpenAI API will fail with "tool_call_id did not have response"
                    error_content = (
                        f"❌ Error executing tool '{tool_call['name']}': {e!s}"
                    )
                    result.append(
                        ToolMessage(
                            content=error_content,
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
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

    def _after_tools(self, state: MessagesState) -> Literal["llm_call", "__end__"]:
        """Route after tool execution.

        Stops the graph immediately if clarification was requested, so the LLM
        cannot call further tools or generate additional output after asking the
        user for input.

        Args:
            state: Current conversation state

        Returns:
            Next node to execute ("llm_call" or "__end__")
        """
        messages = state["messages"]
        # Walk backwards through the most recent ToolMessages from this turn
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                break
            if message.name == "ask_human_for_clarification":
                # Only stop if the tool actually succeeded; on error let the
                # LLM recover (e.g. retry with corrected arguments).
                try:
                    payload = json.loads(message.content)
                    if payload.get("success") is True:
                        return "__end__"
                except (json.JSONDecodeError, AttributeError):
                    pass
        return "llm_call"

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
        agent_builder.add_conditional_edges(
            "tool_node", self._after_tools, ["llm_call", END]
        )

        # Compile with persistent checkpointer for conversation memory
        checkpointer = get_checkpointer()

        # Use interrupt_before for human-in-the-loop confirmation if required
        if self.require_confirmation:
            return agent_builder.compile(
                checkpointer=checkpointer,
                interrupt_before=["tool_node"],
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
