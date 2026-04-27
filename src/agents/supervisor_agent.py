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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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

# Maximum consecutive re-routings to the same agent before the supervisor
# forces FINISH.  Smaller models often fail to emit FINISH and keep
# re-routing to the same agent (sometimes with spurious tool calls).
# The first delegation is always allowed; the counter tracks re-routings.
# Limit of 1: if an agent just completed with tool results, re-routing to
# the same agent is almost always a mistake (the LLM failed to recognise
# the step was done).
_MAX_CONSECUTIVE_SAME_AGENT_REROUTINGS = 1


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
    more_steps_after: bool = Field(
        default=False,
        description=(
            "True ONLY if the user's request requires a DIFFERENT agent after this "
            "one completes (e.g. rag_agent finds parameters, then engineering_agent "
            "optimises). False for single-agent tasks (the vast majority of requests)."
        ),
    )
    final_response: str = Field(
        default="",
        description=(
            "User-facing reply, written in second person as a direct message to the "
            "user. REQUIRED when agent='FINISH' AND no sub-agent has produced a "
            "reply in this turn — e.g. the user is repeating a request that was "
            "already completed earlier in the conversation, or no action is needed. "
            "In that case, briefly tell the user what was already done (including "
            "any file paths or key results) and offer next steps. Leave empty in "
            "all other cases (when routing to a sub-agent, or when FINISH follows "
            "a sub-agent that already replied)."
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
        rag_read_only: bool = False,
    ):
        """Initialize the supervisor agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            seed: Random seed for model (defaults to config.llm_seed)
            rag_read_only: If True, sub-agents only get read-only RAG tools (search, list).
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
        self.routing_llm = self.llm.with_structured_output(
            RouteDecision, method="function_calling"
        )

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
            model_name=self.model_name,
            temperature=self.temperature,
            seed=self.seed,
            rag_read_only=rag_read_only,
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

        # Loop detection state (reset per invoke)
        self._last_routed_agent: str | None = None
        self._consecutive_same_agent_count: int = 0
        self._last_delegation_had_tools: bool = True
        self._expects_followup: bool = False

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
                "For any paper or document lookup, route to rag_agent first. "
                "After rag_agent returns the information, route to engineering_agent "
                "for any follow-up design or optimization tasks. "
                "Keep task_instruction for rag_agent concise: just state which "
                "parameter values to find and from which papers. Do NOT ask for "
                "full citations, DOIs, authors, or extensive quotes — just the "
                "numeric values needed for the next step."
            )
        if os.getenv("SKIP_SEARCH", "false").lower() == "true":
            prompt += (
                "\n\nIMPORTANT: Web search is currently unavailable. "
                "Do NOT route to search_agent under any circumstances. "
                "Use rag_agent for document lookup and engineering_agent "
                "for design tasks."
            )
        return prompt

    def _supervisor_node(self, state: SupervisorState):
        """Supervisor decides which agent should act next using LLM-based routing.

        After each agent completes, the supervisor re-evaluates the full message
        history to decide whether to route to another agent or finish.
        Prior [SUPERVISOR INSTRUCTION] messages are kept so the routing LLM
        can see which sub-tasks were already delegated and completed.
        """
        messages = [
            {"role": "system", "content": self._build_routing_prompt()},
            *state["messages"],
        ]

        # Use structured output to get routing decision from LLM
        route_decision = cast(RouteDecision, self.routing_llm.invoke(messages))
        next_agent = route_decision.agent

        # Store follow-up flag so _route_after_agent knows whether to
        # loop back to the supervisor or end directly after delegation.
        self._expects_followup = route_decision.more_steps_after

        # Log the routing decision with reasoning
        logger.info(
            f"[SUPERVISOR ROUTING] Agent: '{next_agent}' "
            f"| Reasoning: {route_decision.reasoning} "
            f"| more_steps_after: {route_decision.more_steps_after}"
        )

        # ── Same-agent loop detection ────────────────────────────────
        # If the LLM tries to re-route to the agent that just completed,
        # it likely failed to recognise the step was done.  Rather than
        # forcing FINISH (which would skip remaining steps), re-invoke the
        # routing LLM with a redirect hint so it picks a different agent.
        if next_agent not in ("FINISH", "supervisor_response"):
            if next_agent == self._last_routed_agent:
                self._consecutive_same_agent_count += 1
            else:
                self._consecutive_same_agent_count = 0

            if (
                self._consecutive_same_agent_count
                >= _MAX_CONSECUTIVE_SAME_AGENT_REROUTINGS
            ):
                logger.warning(
                    f"[SUPERVISOR] Loop detected: '{next_agent}' re-routed "
                    f"{self._consecutive_same_agent_count} consecutive times. "
                    "Re-invoking routing with redirect hint."
                )
                # Re-invoke routing with a hint to pick a different agent
                redirect_hint = HumanMessage(
                    content=(
                        f"[SYSTEM] You already delegated to '{next_agent}' and "
                        f"it completed its work (see results above). Do NOT "
                        f"route to '{next_agent}' again. Choose a DIFFERENT "
                        f"agent for the next step, or FINISH if all steps are done."
                    )
                )
                retry_messages = [*messages, redirect_hint]
                retry_decision = cast(
                    RouteDecision, self.routing_llm.invoke(retry_messages)
                )
                next_agent = retry_decision.agent
                self._expects_followup = retry_decision.more_steps_after
                logger.info(
                    f"[SUPERVISOR REDIRECT] Agent: '{next_agent}' "
                    f"| Reasoning: {retry_decision.reasoning} "
                    f"| more_steps_after: {retry_decision.more_steps_after}"
                )
                # If the LLM STILL picks the same agent, force FINISH
                if next_agent == self._last_routed_agent:
                    logger.warning(
                        f"[SUPERVISOR] Redirect failed — still '{next_agent}'. "
                        "Forcing FINISH."
                    )
                    self._consecutive_same_agent_count = 0
                    self._last_routed_agent = None
                    return {
                        "next": "FINISH",
                        "messages": [
                            AIMessage(
                                content=(
                                    "I've completed the task based on the "
                                    "information gathered so far."
                                )
                            )
                        ],
                    }
                # Redirect succeeded — use the new decision going forward
                route_decision = retry_decision
                self._consecutive_same_agent_count = 0

            self._last_routed_agent = next_agent
        else:
            # Routing to FINISH or supervisor_response — reset tracking
            self._last_routed_agent = None
            self._consecutive_same_agent_count = 0

        # ── Inject scoped task instruction ───────────────────────────
        # HumanMessage so the delegated agent's LLM treats it as a new
        # user directive (not just a prior assistant turn).  This
        # prevents the agent from ignoring the scope and trying to
        # handle the entire workflow with its own tools.
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

        if next_agent == "FINISH":
            final_response = (route_decision.final_response or "").strip()
            if final_response:
                new_messages.append(AIMessage(content=final_response))
            else:
                # Safety net: if the routing LLM omitted final_response and
                # no sub-agent has replied this turn (last message is the
                # user's), emit a minimal reply so the UI never renders blank.
                last_message = state["messages"][-1] if state["messages"] else None
                if last_message is None or isinstance(last_message, HumanMessage):
                    new_messages.append(
                        AIMessage(
                            content="Done. Let me know if you'd like anything else."
                        )
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
        result = self.engineering_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"engineering_{uuid.uuid4().hex[:8]}"}},
        )
        return self._extract_agent_result(state, result)

    def _hpc_node(self, state: SupervisorState):
        """Delegate to HPC agent."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.hpc_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"hpc_{uuid.uuid4().hex[:8]}"}},
        )
        return self._extract_agent_result(state, result)

    def _search_node(self, state: SupervisorState):
        """Delegate to search agent."""
        if os.getenv("SKIP_SEARCH", "false").lower() == "true":
            self._last_delegation_had_tools = True
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Web search is not available in this context. "
                            "Please use the document search tool (search_documents) instead."
                        )
                    )
                ],
                "next": "",
            }

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.search_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"search_{uuid.uuid4().hex[:8]}"}},
        )
        return self._extract_agent_result(state, result)

    def _rag_node(self, state: SupervisorState):
        """Delegate to RAG agent for document Q&A."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.rag_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"rag_{uuid.uuid4().hex[:8]}"}},
        )
        return self._extract_agent_result(state, result)

    def _arxiv_node(self, state: SupervisorState):
        """Delegate to ArXiv agent for paper search and analysis."""
        if os.getenv("SKIP_ARXIV", "false").lower() == "true":
            # System bypass, not an idle agent — supervisor should re-route
            # (typically to rag_agent), so mark as "productive".
            self._last_delegation_had_tools = True
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
        result = self.arxiv_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"arxiv_{uuid.uuid4().hex[:8]}"}},
        )
        return self._extract_agent_result(state, result)

    def _prusa_node(self, state: SupervisorState):
        """Delegate to Prusa agent."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.prusa_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"prusa_{uuid.uuid4().hex[:8]}"}},
        )
        return self._extract_agent_result(state, result)

    def _cli_node(self, state: SupervisorState):
        """Delegate to CLI agent."""
        logger.info("[SUPERVISOR] _cli_node invoked - delegating to CLI agent")
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.cli_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"cli_{uuid.uuid4().hex[:8]}"}},
        )
        logger.info("[SUPERVISOR] CLI agent returned result")
        return self._extract_agent_result(state, result)

    def _route_after_agent(self, state: SupervisorState) -> str:  # noqa: ARG002
        """Decide whether to loop back to supervisor or end directly.

        Uses two signals:
        - ``_last_delegation_had_tools``: whether the agent made any tool calls.
        - ``_expects_followup``: set from ``RouteDecision.more_steps_after``
          by the supervisor node before delegation.

        Returns to supervisor only when both conditions hold (agent was
        productive AND the supervisor indicated more agents are needed).
        Otherwise ends directly to avoid wasteful LLM re-evaluation.
        """
        if not self._last_delegation_had_tools:
            logger.info("[SUPERVISOR] Agent produced no tool calls — ending directly.")
            return END
        if self._expects_followup:
            logger.info(
                "[SUPERVISOR] Agent completed with tools — more steps expected, "
                "returning to supervisor."
            )
            return "supervisor"
        logger.info("[SUPERVISOR] Agent completed with tools — task done, ending.")
        return END

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

        # Agents loop back to supervisor only if they used tools (productive).
        # If an agent produced no tool calls it had nothing to do — go
        # directly to END to avoid wasteful supervisor re-evaluation
        # (common with smaller models that fail to emit FINISH).
        agent_nodes = [
            "engineering_agent",
            "hpc_agent",
            "search_agent",
            "rag_agent",
            "arxiv_agent",
            "prusa_agent",
            "cli_agent",
        ]
        after_agent_map = {"supervisor": "supervisor", END: END}
        for agent_name in agent_nodes:
            workflow.add_conditional_edges(
                agent_name, self._route_after_agent, after_agent_map
            )
        # supervisor_response always goes directly to END
        workflow.add_edge("supervisor_response", END)

        # Compile with persistent checkpointer
        checkpointer = get_checkpointer()
        # Interrupt before cli_agent for user confirmation of commands
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["cli_agent"],
        )

    def _extract_agent_result(
        self, state: SupervisorState, result: MessagesState
    ) -> dict:
        """Extract new messages from an agent result and track tool usage.

        Shared by all agent delegation nodes to avoid duplicated logic.
        Sets ``_last_delegation_had_tools`` for idle-loop detection in the
        supervisor node.
        """
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        self._last_delegation_had_tools = any(
            isinstance(m, ToolMessage) for m in new_messages
        )
        return {"messages": new_messages, "next": ""}

    def invoke(self, state, config):
        """Invoke the supervisor agent.

        Args:
            state: Current conversation state with messages, or None to resume from interrupt
            config: Configuration including thread_id

        Returns:
            Updated state with agent responses
        """
        # Reset loop detection state for each top-level invocation
        self._last_routed_agent = None
        self._consecutive_same_agent_count = 0
        self._last_delegation_had_tools = True
        self._expects_followup = False

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
