"""
Supervisor agent that coordinates multiple specialized agents.

This agent uses a hierarchical approach - it delegates tasks to specialized
sub-agents (Engineering, Search, etc.) rather than having all tools directly.
"""

import logging
import uuid
from typing import Annotated, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
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
from src.utils.prompts import SUPERVISOR_AGENT_SYSTEM_PROMPT

# ChromaDB imports removed - now using MMORE via RAGAgent

logger = logging.getLogger(__name__)


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
    ):
        """Initialize the supervisor agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
        """
        self.model_name = model_name or config.llm_model
        self.temperature = (
            temperature if temperature is not None else config.llm_temperature
        )
        self.llm = init_chat_model(self.model_name, temperature=self.temperature)

        # Initialize specialized sub-agents
        self.engineering_agent = EngineeringAgent(
            model_name=self.model_name, temperature=self.temperature
        )
        self.hpc_agent = HPCAgent(
            model_name=self.model_name, temperature=self.temperature
        )
        self.search_agent = SearchAgent(
            model_name=self.model_name, temperature=self.temperature
        )
        # Initialize RAG agent with MMORE
        self.rag_agent = RAGAgent(
            model_name=self.model_name, temperature=self.temperature
        )
        # Initialize ArXiv agent with MMORE
        self.arxiv_agent = ArXivAgent(
            model_name=self.model_name,
            temperature=self.temperature,
        )
        # PrusaAgent will check SKIP_MCP env var automatically
        self.prusa_agent = PrusaAgent(
            model_name=self.model_name, temperature=self.temperature
        )
        # CLI confirmation is handled by supervisor-level interrupt, not internally
        self.cli_agent = CLIAgent(
            model_name=self.model_name,
            require_confirmation=False,
            temperature=self.temperature,
        )

        # Build the supervisor graph
        self.graph = self._build_graph()

    def _build_routing_prompt(self) -> str:
        """Get the routing system prompt for the supervisor.

        Returns:
            Routing prompt from centralized prompts file
        """
        return SUPERVISOR_AGENT_SYSTEM_PROMPT

    def _supervisor_node(self, state: SupervisorState):  # noqa: PLR0912
        """Supervisor decides which agent should act next."""
        # Only route if we haven't routed yet (no next value set)
        if state.get("next") and state["next"] != "":
            # Already routed, finish
            return {
                "next": "FINISH",
                "messages": [],  # Don't return messages - add_messages will handle state
            }

        # Check user message directly for "open" commands - route to CLI immediately
        last_user_message = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                last_user_message = str(msg.content).lower()
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                last_user_message = str(msg.get("content", "")).lower()
                break

        logger.info(
            f"[SUPERVISOR ROUTING DEBUG] Last user message: '{last_user_message}'"
        )
        logger.info(
            f"[SUPERVISOR ROUTING DEBUG] Starts with 'open ': {last_user_message.startswith('open ')}"
        )

        # Direct routing for "open" commands - bypass LLM routing
        if last_user_message.startswith("open "):
            logger.info(
                "[SUPERVISOR ROUTING DEBUG] ROUTING TO CLI_AGENT (direct routing)"
            )
            return {
                "next": "cli_agent",
                "messages": [],  # Don't return messages - add_messages will handle state
            }

        messages = [
            {"role": "system", "content": self._build_routing_prompt()},
            *state["messages"],
        ]
        response = self.llm.invoke(messages)

        # Extract routing decision from response
        content = str(response.content).strip().lower()

        # Determine next agent - check engineering first (more specific keywords)
        next_agent = "supervisor_response"  # Default to supervisor response for safety
        if (
            "engineering" in content
            or "training" in content
            or "generate" in content
            or "wandb" in content
            or "slurm" in content
            or "beam" in content
            or "optimization" in content
            or "design" in content
        ):
            # Engineering handles: training scripts, model generation, design tasks, WandB operations
            next_agent = "engineering_agent"
        elif (
            "hpc" in content
            or "submit" in content
            or "status" in content
            or "cancel" in content
            or "monitor" in content
            or "download output" in content
        ):
            # HPC handles: job submission, monitoring, cancellation, output retrieval
            next_agent = "hpc_agent"
        elif "search" in content:
            next_agent = "search_agent"
        elif "rag" in content or "document" in content:
            # RAG handles: questions about uploaded documents/papers
            next_agent = "rag_agent"
        elif "arxiv" in content or "paper" in content:
            # ArXiv handles: ArXiv paper search, download, and analysis
            next_agent = "arxiv_agent"
        elif (
            "prusa" in content
            or "printer" in content
            or "print" in content
            or "3d printer" in content
        ):
            # Prusa handles: printer management, job monitoring, printer control
            next_agent = "prusa_agent"
        elif "cli" in content or "command" in content or "run" in content:
            # CLI handles: local command-line tool execution
            next_agent = "cli_agent"
        elif "finish" in content or "supervisor" in content:
            # Supervisor will answer directly
            next_agent = "supervisor_response"

        # Preserve the original messages and add the routing decision
        logger.info(f"[SUPERVISOR ROUTING DEBUG] LLM routing decision: '{next_agent}'")
        return {
            "next": next_agent,
            "messages": [],  # Don't return messages - add_messages will handle state
        }

    def _supervisor_response_node(self, state: SupervisorState):
        """Supervisor responds directly to informational questions."""

        # Build a helpful system prompt for answering capability questions
        capabilities_prompt = """You are a helpful assistant that can answer questions about the system's capabilities.

The system has the following capabilities:

**Engineering & Optimization:**
- Structural optimization and topology design
- Beam design problems and simulation
- STL file generation for 3D printing
- Access to pre-trained models from WandB
- Generative models (GANs, Diffusion)



**CLI Command Execution:**
- Execute any local command-line tool
- PrusaSlicer for STL slicing to G-code
- Mesh processing and file conversion tools
- Basic shell commands (pwd, ls, cat, etc.)

**HPC Cluster Management:**
- SLURM job submission and monitoring
- Job status checking and output retrieval
- Remote cluster operations

**Document Intelligence:**
- Upload and analyze research papers (PDFs)
- Question-answering about uploaded documents
- Document summarization and information extraction
- Persistent knowledge base for your papers

**ArXiv Research:**
- Search ArXiv for academic papers
- Download and analyze ArXiv papers
- Ask questions about downloaded papers using RAG
- Track and manage your research paper collection

**Web Research:**
- Search for engineering information
- Find best practices and papers
- Current state-of-the-art research

**Prusa 3D Printer Management:**
- Monitor printer status and print jobs
- Control printers (pause, resume, stop)
- Manage files and storage

Answer the user's question clearly and concisely about what the system can do.

**CRITICAL: You MUST end EVERY response with 2-4 contextual follow-up suggestions in this format:**

```suggested_prompts
Suggestion 1 text here
---
Suggestion 2 text here
---
Suggestion 3 text here
```

For capability questions, suggest specific actions the user might want to try with the system."""

        messages = [
            {"role": "system", "content": capabilities_prompt},
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
        return {"messages": new_messages, "next": "FINISH"}

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
        return {"messages": new_messages, "next": "FINISH"}

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
        return {"messages": new_messages, "next": "FINISH"}

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
        return {"messages": new_messages, "next": "FINISH"}

    def _arxiv_node(self, state: SupervisorState):
        """Delegate to ArXiv agent for paper search and analysis."""

        agent_state = cast(MessagesState, {"messages": state["messages"]})
        # Use unique thread_id to avoid checkpoint conflicts between invocations
        result = self.arxiv_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": f"arxiv_{uuid.uuid4().hex[:8]}"}},
        )
        # Return all new messages to preserve tool call/response chain
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        return {"messages": new_messages, "next": "FINISH"}

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
        return {"messages": new_messages, "next": "FINISH"}

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
        return {"messages": new_messages, "next": "FINISH"}

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

        # Agents go directly to END (no looping back to supervisor)
        workflow.add_edge("supervisor_response", END)
        workflow.add_edge("engineering_agent", END)
        workflow.add_edge("hpc_agent", END)
        workflow.add_edge("search_agent", END)
        workflow.add_edge("rag_agent", END)
        workflow.add_edge("arxiv_agent", END)
        workflow.add_edge("prusa_agent", END)
        workflow.add_edge("cli_agent", END)

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
