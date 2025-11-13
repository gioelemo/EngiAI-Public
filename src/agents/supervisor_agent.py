"""
Supervisor agent that coordinates multiple specialized agents.

This agent uses a hierarchical approach - it delegates tasks to specialized
sub-agents (Engineering, Search, etc.) rather than having all tools directly.
"""

from typing import Annotated, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage
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

    def __init__(self, model_name: str | None = None, temperature: float | None = None):
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
        self.rag_agent = (
            RAGAgent()
        )  # RAG agent doesn't need model_name, uses ChatOpenAI internally
        self.arxiv_agent = ArXivAgent(
            model_name=self.model_name, temperature=self.temperature
        )
        # PrusaAgent will check SKIP_MCP env var automatically
        self.prusa_agent = PrusaAgent(
            model_name=self.model_name, temperature=self.temperature
        )
        # Disable CLI agent's internal confirmation - supervisor handles interrupts at its level
        self.cli_agent = CLIAgent(
            model_name=self.model_name,
            require_confirmation=False,
            temperature=self.temperature,
        )

        # Build the supervisor graph
        self.graph = self._build_graph()

    def _build_routing_prompt(self) -> str:
        """Build the routing system prompt based on available agents."""
        available_agents = []
        agent_names = []

        available_agents.append(
            "- engineering_agent: Structural optimization, beam design, topology optimization, STL conversion, downloading/using pre-trained models from WandB, generative models (GANs, Diffusion)"
        )
        available_agents.append(
            "- hpc_agent: HPC cluster job management, SLURM job submission, job monitoring, output retrieval"
        )
        available_agents.append("- search_agent: Web research and finding information")
        available_agents.append(
            "- rag_agent: Questions about uploaded documents, papers, PDFs (use this for 'what does the paper say', 'explain this document', 'summarize the research')"
        )
        available_agents.append(
            "- arxiv_agent: ArXiv research paper search, download papers from ArXiv, analyze academic papers with RAG (use this for 'find papers about', 'search ArXiv for', 'download paper', 'analyze this ArXiv paper')"
        )
        available_agents.append(
            "- prusa_agent: Prusa 3D printer management via Prusa Connect (printer status, job monitoring, printer control, file management)"
        )
        available_agents.append(
            "- cli_agent: Execute local CLI commands (PrusaSlicer, mesh processing, file conversion, any command-line tool)"
        )
        agent_names.extend(
            [
                "'engineering_agent'",
                "'hpc_agent'",
                "'search_agent'",
                "'rag_agent'",
                "'arxiv_agent'",
                "'prusa_agent'",
                "'cli_agent'",
            ]
        )

        return (
            "You are a supervisor that routes tasks to specialized agents.\n\n"
            + "Available agents:\n"
            + "\n".join(available_agents)
            + "\n\n"
            + "CRITICAL: Distinguish between PURE CAPABILITY QUESTIONS vs ACTUAL TASK REQUESTS.\n\n"
            + "Use FINISH (answer directly) ONLY for questions about general system capabilities:\n"
            + "- 'what can you do?' → FINISH\n"
            + "- 'what agents do you have?' → FINISH\n"
            + "- 'do you support optimization?' (general inquiry) → FINISH\n\n"
            + "Route to agents for ANY specific task requests, even if phrased as questions:\n"
            + "- 'can you optimize this beam?' → engineering_agent (it's asking you to DO something)\n"
            + "- 'can you generate a SLURM script?' → engineering_agent (it's asking you to DO something)\n"
            + "- 'can you slice this STL?' → cli_agent (it's asking you to DO something)\n"
            + "- 'can you download this model?' → engineering_agent (it's asking you to DO something)\n\n"
            + "IMPORTANT ROUTING RULES:\n"
            + "- Any mention of 'wandb', 'models', 'pretrained', 'download model', 'GAN', 'diffusion', "
            + "'beam', 'beam2d', 'optimization', 'design', 'algorithm', 'checkpoint', 'training', 'train', "
            + "'generate script', 'generate SLURM', 'slurm', 'model training' → use engineering_agent\n"
            + "- HPC cluster job management ONLY: submit job, job submission, job status, check job, monitor job, cancel job, download output, 'euler' cluster operations → use hpc_agent\n"
            + "- Web search, research, finding information ONLINE → use search_agent\n"
            + "- Questions about UPLOADED documents, papers, PDFs: 'what does the paper say', 'explain this document', 'summarize the research', 'what are the findings' → use rag_agent\n"
            + "- ArXiv paper search and analysis: 'find papers on ArXiv', 'search ArXiv for', 'download ArXiv paper', 'analyze paper 1605.08386', 'what papers are available on', 'ArXiv ID', 'arxiv.org' → use arxiv_agent\n"
            + "- Prusa printer management: 'printer status', 'print jobs', 'pause print', 'resume print', 'stop print', 'start print', 'Prusa Connect', 'printer', '3D printer' → use prusa_agent\n"
            + "- EXECUTE CLI commands: 'slice file.stl', 'run PrusaSlicer', 'convert file', 'execute pwd' → use cli_agent\n"
            + "\n"
            + "Respond with ONLY ONE WORD: "
            + ", ".join(agent_names[:-1])
            + (" or " if len(agent_names) > 1 else "")
            + agent_names[-1]
            + " or 'FINISH' if it's a pure capability question.\n"
            + "No explanations, no other text, just the agent name or FINISH."
        )

    def _supervisor_node(self, state: SupervisorState):
        """Supervisor decides which agent should act next."""
        # Only route if we haven't routed yet (no next value set)
        if state.get("next") and state["next"] != "":
            # Already routed, finish
            return {
                "next": "FINISH",
                "messages": state["messages"],
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
        elif "cli" in content or "command" in content:
            # CLI handles: local command-line tool execution
            next_agent = "cli_agent"
        elif "finish" in content or "supervisor" in content:
            # Supervisor will answer directly
            next_agent = "supervisor_response"

        # Preserve the original messages and add the routing decision
        return {
            "next": next_agent,
            "messages": state["messages"],
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

**Code Execution:**
- Python code execution and calculations
- Data analysis and quick evaluations

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
        result = self.engineering_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "engineering"}},
        )
        # Extract only the LAST final AI response, excluding intermediate tool calls/responses
        # Sub-agents may loop through multiple tool calls, we only want the final answer
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        final_messages: list[AnyMessage] = [
            msg
            for msg in new_messages
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ]
        # Keep only the last final message (the actual answer to the user)
        if final_messages:
            return {"messages": [final_messages[-1]], "next": "FINISH"}
        elif new_messages:
            return {"messages": [new_messages[-1]], "next": "FINISH"}
        return {"messages": [], "next": "FINISH"}

    def _hpc_node(self, state: SupervisorState):
        """Delegate to HPC agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.hpc_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "hpc"}},
        )
        # Extract only the LAST final AI response, excluding intermediate tool calls/responses
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        final_messages: list[AnyMessage] = [
            msg
            for msg in new_messages
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ]
        if final_messages:
            return {"messages": [final_messages[-1]], "next": "FINISH"}
        elif new_messages:
            return {"messages": [new_messages[-1]], "next": "FINISH"}
        return {"messages": [], "next": "FINISH"}

    def _search_node(self, state: SupervisorState):
        """Delegate to search agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.search_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "search"}},
        )
        # Extract only the LAST final AI response, excluding intermediate tool calls/responses
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        final_messages: list[AnyMessage] = [
            msg
            for msg in new_messages
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ]
        if final_messages:
            return {"messages": [final_messages[-1]], "next": "FINISH"}
        elif new_messages:
            return {"messages": [new_messages[-1]], "next": "FINISH"}
        return {"messages": [], "next": "FINISH"}

    def _rag_node(self, state: SupervisorState):
        """Delegate to RAG agent for document Q&A."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.rag_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "rag"}},
        )
        # Extract only the LAST final AI response, excluding intermediate tool calls/responses
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        final_messages: list[AnyMessage] = [
            msg
            for msg in new_messages
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ]
        if final_messages:
            return {"messages": [final_messages[-1]], "next": "FINISH"}
        elif new_messages:
            return {"messages": [new_messages[-1]], "next": "FINISH"}
        return {"messages": [], "next": "FINISH"}

    def _arxiv_node(self, state: SupervisorState):
        """Delegate to ArXiv agent for paper search and analysis."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.arxiv_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "arxiv"}},
        )
        # Extract only the LAST final AI response, excluding intermediate tool calls/responses
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        final_messages: list[AnyMessage] = [
            msg
            for msg in new_messages
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ]
        if final_messages:
            return {"messages": [final_messages[-1]], "next": "FINISH"}
        elif new_messages:
            return {"messages": [new_messages[-1]], "next": "FINISH"}
        return {"messages": [], "next": "FINISH"}

    def _prusa_node(self, state: SupervisorState):
        """Delegate to Prusa agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.prusa_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "prusa"}},
        )
        # Extract only the LAST final AI response, excluding intermediate tool calls/responses
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        final_messages: list[AnyMessage] = [
            msg
            for msg in new_messages
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ]
        if final_messages:
            return {"messages": [final_messages[-1]], "next": "FINISH"}
        elif new_messages:
            return {"messages": [new_messages[-1]], "next": "FINISH"}
        return {"messages": [], "next": "FINISH"}

    def _cli_node(self, state: SupervisorState):
        """Delegate to CLI agent."""
        agent_state = cast(MessagesState, {"messages": state["messages"]})
        result = self.cli_agent.invoke(
            agent_state,
            {"configurable": {"thread_id": "cli"}},
        )
        # Extract only the LAST final AI response, excluding intermediate tool calls/responses
        input_len = len(state["messages"])
        new_messages = result["messages"][input_len:]
        final_messages: list[AnyMessage] = [
            msg
            for msg in new_messages
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ]
        if final_messages:
            return {"messages": [final_messages[-1]], "next": "FINISH"}
        elif new_messages:
            return {"messages": [new_messages[-1]], "next": "FINISH"}
        return {"messages": [], "next": "FINISH"}

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
        # Interrupt before CLI agent so we can check its planned commands
        return workflow.compile(
            checkpointer=checkpointer, interrupt_before=["cli_agent"]
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
