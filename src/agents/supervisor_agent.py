"""
Supervisor agent that coordinates multiple specialized agents.

This agent uses a hierarchical approach - it delegates tasks to specialized
sub-agents (Engineering, Search, etc.) rather than having all tools directly.
"""

from typing import Annotated, cast

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import config
from src.agents.engineering_agent import EngineeringAgent
from src.agents.search_agent import SearchAgent
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

    def __init__(self, model_name: str | None = None):
        """Initialize the supervisor agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        self.model_name = model_name or config.llm_model
        self.llm = init_chat_model(self.model_name)

        # Initialize specialized sub-agents
        self.engineering_agent = EngineeringAgent(model_name=self.model_name)
        self.search_agent = SearchAgent(model_name=self.model_name)

        # Build the supervisor graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the supervisor workflow graph with agent routing."""
        # Create routing system prompt
        system_prompt = (
            "You are a supervisor that routes tasks to specialized agents.\n\n"
            + "Available agents:\n"
            + "- engineering_agent: Structural optimization, beam design, topology optimization, STL conversion\n"
            + "- search_agent: Web research and finding information\n\n"
            + "Analyze the user's request and respond with ONLY ONE WORD: "
            + "either 'engineering_agent' or 'search_agent'.\n"
            + "No explanations, no other text, just the agent name."
        )

        def supervisor_node(state: SupervisorState):
            """Supervisor decides which agent should act next."""
            # Only route if we haven't routed yet (no next value set)
            if state.get("next") and state["next"] != "":
                # Already routed, finish
                return {"next": "FINISH"}

            messages = [
                {"role": "system", "content": system_prompt},
                *state["messages"],
            ]
            response = self.llm.invoke(messages)

            # Extract routing decision from response
            content = str(response.content).strip().lower()

            # Determine next agent
            next_agent = "FINISH"
            if "engineering" in content:
                next_agent = "engineering_agent"
            elif "search" in content:
                next_agent = "search_agent"

            return {"next": next_agent}

        def engineering_node(state: SupervisorState):
            """Delegate to engineering agent."""
            # Convert supervisor state to agent state (MessagesState format)
            agent_state = cast(MessagesState, {"messages": state["messages"]})
            result = self.engineering_agent.invoke(
                agent_state,
                {"configurable": {"thread_id": "engineering"}},
            )
            # Return ALL new messages from the engineering agent (including tool calls/results)
            # This shows the full conversation flow to the user
            messages_before = len(state["messages"])
            new_messages = result["messages"][messages_before:]
            # Mark that we've completed routing
            return {"messages": new_messages, "next": "FINISH"}

        def search_node(state: SupervisorState):
            """Delegate to search agent."""
            # Convert supervisor state to agent state (MessagesState format)
            agent_state = cast(MessagesState, {"messages": state["messages"]})
            result = self.search_agent.invoke(
                agent_state,
                {"configurable": {"thread_id": "search"}},
            )
            # Return ALL new messages from the search agent (including tool calls/results)
            messages_before = len(state["messages"])
            new_messages = result["messages"][messages_before:]
            # Mark that we've completed routing
            return {"messages": new_messages, "next": "FINISH"}

        # Build the graph
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("engineering_agent", engineering_node)
        workflow.add_node("search_agent", search_node)

        # Add edges - start with supervisor
        workflow.add_edge(START, "supervisor")

        # Conditional routing based on supervisor decision
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "engineering_agent": "engineering_agent",
                "search_agent": "search_agent",
                "FINISH": END,
            },
        )

        # Agents go directly to END (no looping back to supervisor)
        workflow.add_edge("engineering_agent", END)
        workflow.add_edge("search_agent", END)

        # Compile with memory
        memory = InMemorySaver()
        return workflow.compile(checkpointer=memory)

    def invoke(self, state, config):
        """Invoke the supervisor agent.

        Args:
            state: Current conversation state with messages
            config: Configuration including thread_id

        Returns:
            Updated state with agent responses
        """
        # Convert MessagesState to SupervisorState if needed
        if "next" not in state:
            supervisor_state = {"messages": state["messages"], "next": ""}
        else:
            supervisor_state = state

        result = self.graph.invoke(supervisor_state, config)

        # Return in MessagesState format for compatibility
        return {"messages": result["messages"]}
