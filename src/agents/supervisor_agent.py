"""
Supervisor agent that coordinates multiple specialized agents.

This agent uses a hierarchical approach - it delegates tasks to specialized
sub-agents (Engineering, Search, etc.) rather than having all tools directly.
"""

import logging
import os
import uuid
from typing import Annotated, Any, Literal, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from config import config
from src.agents.arxiv_agent import ArXivAgent
from src.agents.cli_agent import CLIAgent
from src.agents.engineering_agent import EngineeringAgent
from src.agents.hpc_agent import HPCAgent
from src.agents.prusa_agent import PrusaAgent
from src.agents.rag_agent import RAGAgent
from src.agents.search_agent import SearchAgent
from src.checkpoint import get_checkpointer
from src.models.state import MessagesState
from src.utils.prompts import (
    SUPERVISOR_AGENT_SYSTEM_PROMPT,
    SUPERVISOR_CAPABILITIES_PROMPT,
    _is_eval_mode,
    strip_suggested_prompts,
)

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output for routing decision."""

    agent: Literal[
        "engineering_agent",
        "hpc_agent",
        "search_agent",
        "rag_agent",
        "arxiv_agent",
        "prusa_agent",
        "cli_agent",
        "supervisor_response",
        "FINISH",
    ] = Field(description="The agent to route the task to based on the user's request")
    reasoning: str = Field(
        description="Brief explanation of why this agent was selected"
    )
    task_instruction: str = Field(
        default="",
        description=(
            "Specific sub-task instruction for the selected agent. For multi-step "
            "workflows, scope the instruction to ONLY the next step(s) that this "
            "agent should perform, then stop. E.g. 'Generate the SLURM training "
            "command. Do NOT download models or simulate designs — that will be "
            "handled after HPC training completes.' Leave empty for single-step tasks."
        ),
    )


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

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ):
        """Initialize the supervisor agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            seed: Random seed for model (defaults to config.llm_seed)
        """
        self.model_name = model_name or config.llm_model
        self.temperature = (
            temperature if temperature is not None else config.llm_temperature
        )
        self.seed = seed if seed is not None else config.llm_seed

        # Build model kwargs
        model_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.seed is not None:
            model_kwargs["seed"] = self.seed

        self.llm = init_chat_model(self.model_name, **model_kwargs)
        # Create structured LLM for routing decisions
        self.routing_llm = self.llm.with_structured_output(RouteDecision)

        # Initialize specialized sub-agents
        self.engineering_agent = EngineeringAgent(
            model_name=self.model_name,
            temperature=self.temperature,
            seed=self.seed,
        )
        self.hpc_agent = HPCAgent(
            model_name=self.model_name, temperature=self.temperature, seed=self.seed
        )
        self.search_agent = SearchAgent(
            model_name=self.model_name, temperature=self.temperature, seed=self.seed
        )
        # Initialize RAG agent with MMORE
        self.rag_agent = RAGAgent(
            model_name=self.model_name, temperature=self.temperature, seed=self.seed
        )
        # Initialize ArXiv agent with MMORE
        self.arxiv_agent = ArXivAgent(
            model_name=self.model_name,
            temperature=self.temperature,
            seed=self.seed,
        )
        # PrusaAgent will check SKIP_MCP env var automatically
        self.prusa_agent = PrusaAgent(
            model_name=self.model_name, temperature=self.temperature, seed=self.seed
        )
        # CLI confirmation is handled by supervisor-level interrupt, not internally
        self.cli_agent = CLIAgent(
            model_name=self.model_name,
            require_confirmation=False,
            temperature=self.temperature,
            seed=self.seed,
        )

        # Build the supervisor graph
        self.graph = self._build_graph()

    def _build_routing_prompt(self) -> str:
        """Get the routing system prompt for the supervisor.

        Returns:
            Routing prompt from centralized prompts file
        """
        prompt = SUPERVISOR_AGENT_SYSTEM_PROMPT
        if os.getenv("SKIP_ARXIV", "false").lower() == "true":
            prompt += (
                "\n\nIMPORTANT: ArXiv search is currently unavailable. "
                "Do NOT route to arxiv_agent under any circumstances. "
                "If the user asks about a paper or document AND then wants to perform a design task, "
                "route to engineering_agent (it has both document search and optimization tools). "
                "If the user only wants document Q&A with no follow-up action, route to rag_agent."
            )
        return prompt

    @staticmethod
    def _filter_supervisor_instructions(messages: list) -> list:
        """Remove [SUPERVISOR INSTRUCTION] messages from routing context.

        These scoped sub-task instructions were injected for delegated agents.
        When the supervisor re-evaluates, it should see the original user
        request + agent results, not its own prior sub-task scoping.
        """
        return [
            m
            for m in messages
            if not (
                isinstance(m, HumanMessage)
                and isinstance(m.content, str)
                and m.content.startswith("[SUPERVISOR INSTRUCTION")
            )
        ]

    def _supervisor_node(self, state: SupervisorState):
        """Supervisor decides which agent should act next using LLM-based routing.

        After each agent completes, the supervisor re-evaluates the full message
        history to decide whether to route to another agent or finish.
        """
        # Filter out prior [SUPERVISOR INSTRUCTION] messages so the LLM
        # re-evaluates against the original user request, not scoped sub-tasks.
        filtered = self._filter_supervisor_instructions(state["messages"])
        messages = [
            {"role": "system", "content": self._build_routing_prompt()},
            *filtered,
        ]

        # Use structured output to get routing decision from LLM
        route_decision = cast(RouteDecision, self.routing_llm.invoke(messages))
        next_agent = route_decision.agent

        # Log the routing decision with reasoning
        logger.info(
            f"[SUPERVISOR ROUTING] Agent: '{next_agent}' | Reasoning: {route_decision.reasoning}"
        )

        # Inject scoped task instruction as a HumanMessage so the delegated
        # agent's LLM treats it as a new user directive (not just a prior
        # assistant turn).  This prevents the agent from ignoring the scope
        # and trying to handle the entire workflow with its own tools.
        new_messages: list = []
        if route_decision.task_instruction and next_agent not in (
            "FINISH",
            "supervisor_response",
        ):
            new_messages.append(
                HumanMessage(
                    content=(
                        f"[SUPERVISOR INSTRUCTION — follow this exactly] "
                        f"{route_decision.task_instruction}"
                    )
                )
            )
            logger.info(
                f"[SUPERVISOR] Task instruction for {next_agent}: {route_decision.task_instruction}"
            )

        return {
            "next": next_agent,
            "messages": new_messages,
        }

    def _supervisor_response_node(self, state: SupervisorState):
        """Supervisor responds directly to informational questions."""

        cap_prompt = SUPERVISOR_CAPABILITIES_PROMPT
        if _is_eval_mode():
            cap_prompt = strip_suggested_prompts(cap_prompt)
        messages = [
            {"role": "system", "content": cap_prompt},
            *state["messages"],
        ]
        response = self.llm.invoke(messages)

        return {"messages": [AIMessage(content=response.content)], "next": "FINISH"}

    def _engineering_node(self, state: SupervisorState):
        """Delegate to engineering agent."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        # Use unique thread_id to avoid checkpoint conflicts between invocations
        result = self.engineering_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"engineering_{uuid.uuid4().hex[:8]}"}},
        )
        # Return all new messages to preserve tool call/response chain
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        return {"messages": new_messages, "next": ""}

    def _hpc_node(self, state: SupervisorState):
        """Delegate to HPC agent."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        # Use unique thread_id to avoid checkpoint conflicts between invocations
        result = self.hpc_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"hpc_{uuid.uuid4().hex[:8]}"}},
        )
        # Return all new messages to preserve tool call/response chain
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        return {"messages": new_messages, "next": ""}

    def _search_node(self, state: SupervisorState):
        """Delegate to search agent."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        # Use unique thread_id to avoid checkpoint conflicts between invocations
        result = self.search_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"search_{uuid.uuid4().hex[:8]}"}},
        )
        # Return all new messages to preserve tool call/response chain
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        return {"messages": new_messages, "next": ""}

    def _rag_node(self, state: SupervisorState):
        """Delegate to RAG agent for document Q&A."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        # Use unique thread_id to avoid checkpoint conflicts between invocations
        result = self.rag_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"rag_{uuid.uuid4().hex[:8]}"}},
        )
        # Return all new messages to preserve tool call/response chain
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        return {"messages": new_messages, "next": ""}

    def _arxiv_node(self, state: SupervisorState):
        """Delegate to ArXiv agent for paper search and analysis."""
        if os.getenv("SKIP_ARXIV", "false").lower() == "true":
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "ArXiv search is not available in this context. "
                            "Please use the document search tool (search_documents) instead."
                        )
                    )
                ],
                "next": "",
            }

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        # Use unique thread_id to avoid checkpoint conflicts between invocations
        result = self.arxiv_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"arxiv_{uuid.uuid4().hex[:8]}"}},
        )
        # Return all new messages to preserve tool call/response chain
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        return {"messages": new_messages, "next": ""}

    def _prusa_node(self, state: SupervisorState):
        """Delegate to Prusa agent."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        # Use unique thread_id to avoid checkpoint conflicts between invocations
        result = self.prusa_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"prusa_{uuid.uuid4().hex[:8]}"}},
        )
        # Return all new messages to preserve tool call/response chain
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        return {"messages": new_messages, "next": ""}

    def _cli_node(self, state: SupervisorState):
        """Delegate to CLI agent."""
        logger.info("[SUPERVISOR] _cli_node invoked - delegating to CLI agent")
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        # Use unique thread_id to avoid checkpoint conflicts between invocations
        result = self.cli_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"cli_{uuid.uuid4().hex[:8]}"}},
        )
        logger.info("[SUPERVISOR] CLI agent returned result")
        # Return all new messages to preserve tool call/response chain
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        return {"messages": new_messages, "next": ""}

    def _build_graph(self):
        """Build the supervisor workflow graph with agent routing."""

        # Build the graph
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("supervisor_response", self._supervisor_response_node)
        workflow.add_node("engineering_agent", self._engineering_node)
        workflow.add_node("hpc_agent", self._hpc_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("arxiv_agent", self._arxiv_node)
        workflow.add_node("prusa_agent", self._prusa_node)
        workflow.add_node("cli_agent", self._cli_node)

        # Add edges - start with supervisor
        workflow.add_edge(START, "supervisor")

        # Build conditional routing dictionary
        routing_dict = {
            "supervisor_response": "supervisor_response",
            "engineering_agent": "engineering_agent",
            "hpc_agent": "hpc_agent",
            "search_agent": "search_agent",
            "rag_agent": "rag_agent",
            "arxiv_agent": "arxiv_agent",
            "prusa_agent": "prusa_agent",
            "cli_agent": "cli_agent",
            "FINISH": END,
        }

        # Conditional routing based on supervisor decision
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            routing_dict,
        )

        # Agents loop back to supervisor for multi-step workflow support
        workflow.add_edge("engineering_agent", "supervisor")
        workflow.add_edge("hpc_agent", "supervisor")
        workflow.add_edge("search_agent", "supervisor")
        workflow.add_edge("rag_agent", "supervisor")
        workflow.add_edge("arxiv_agent", "supervisor")
        workflow.add_edge("prusa_agent", "supervisor")
        workflow.add_edge("cli_agent", "supervisor")
        # Only supervisor_response goes directly to END
        workflow.add_edge("supervisor_response", END)

        # Compile with persistent checkpointer
        checkpointer = get_checkpointer()
        # Interrupt before cli_agent for user confirmation of commands
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["cli_agent"],
        )

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
