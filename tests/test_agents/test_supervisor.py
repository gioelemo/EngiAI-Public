"""
Comprehensive tests for the Supervisor Agent.

Tests cover:
- Initialization and configuration
- LLM-based routing with structured output
- Error handling and edge cases
- State management and conversation flow
- Agent delegation
- Integration workflows
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor_agent import RouteDecision, SupervisorAgent, SupervisorState

# Mock heavy dependencies
sys.modules["engibench"] = MagicMock()
sys.modules["engibench.problems"] = MagicMock()
sys.modules["engibench.problems.beams2d"] = MagicMock()
sys.modules["engibench.problems.beams2d.v0"] = MagicMock()


# ============================================================================
# TEST HELPERS
# ============================================================================


def create_mock_llm_with_structured_output(agent_name, reasoning="Test reasoning"):
    """Helper to create a mock LLM that supports with_structured_output."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()

    # Create a RouteDecision object
    route_decision = RouteDecision(agent=agent_name, reasoning=reasoning)
    mock_structured_llm.invoke.return_value = route_decision

    # Mock with_structured_output to return the structured LLM
    mock_llm.with_structured_output.return_value = mock_structured_llm

    return mock_llm


def create_mock_routing_llm(agent_name, reasoning="Test reasoning"):
    """Helper to create a mock routing LLM that returns RouteDecision."""
    mock_routing_llm = MagicMock()
    route_decision = RouteDecision(agent=agent_name, reasoning=reasoning)
    mock_routing_llm.invoke.return_value = route_decision
    return mock_routing_llm


@pytest.fixture
def _mock_agents():
    """Fixture to mock all agent dependencies."""
    with (
        patch("src.agents.supervisor_agent.EngineeringAgent") as mock_eng,
        patch("src.agents.supervisor_agent.HPCAgent") as mock_hpc,
        patch("src.agents.supervisor_agent.SearchAgent") as mock_search,
        patch("src.agents.supervisor_agent.RAGAgent") as mock_rag,
        patch("src.agents.supervisor_agent.ArXivAgent") as mock_arxiv,
        patch("src.agents.supervisor_agent.PrusaAgent") as mock_prusa,
        patch("src.agents.supervisor_agent.CLIAgent") as mock_cli,
        patch("src.agents.supervisor_agent.init_chat_model") as mock_llm,
        patch("src.agents.supervisor_agent.get_checkpointer") as mock_cp,
    ):
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_cp.return_value = None

        yield {
            "engineering": mock_eng,
            "hpc": mock_hpc,
            "search": mock_search,
            "rag": mock_rag,
            "arxiv": mock_arxiv,
            "prusa": mock_prusa,
            "cli": mock_cli,
        }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


@pytest.mark.unit
def test_supervisor_agent_initialization(_mock_agents):
    """Test SupervisorAgent initialization with all components."""
    agent = SupervisorAgent()

    assert agent.graph is not None
    assert agent.llm is not None
    assert agent.routing_llm is not None
    assert hasattr(agent, "engineering_agent")
    assert hasattr(agent, "hpc_agent")
    assert hasattr(agent, "search_agent")
    assert hasattr(agent, "rag_agent")
    assert hasattr(agent, "arxiv_agent")
    assert hasattr(agent, "prusa_agent")
    assert hasattr(agent, "cli_agent")
    assert hasattr(agent.graph, "invoke")


@pytest.mark.unit
def test_supervisor_agent_custom_model(_mock_agents):
    """Test SupervisorAgent with custom model configuration."""
    agent = SupervisorAgent(model_name="gpt-4", temperature=0.5)

    assert agent.model_name == "gpt-4"
    assert agent.temperature == 0.5


@pytest.mark.unit
def test_supervisor_seed_propagation():
    """Test that supervisor passes seed to all sub-agents."""
    # Create supervisor without mocks so we can test real seed propagation
    with patch("src.agents.base_agent.init_chat_model") as mock_base_init:
        with patch("src.agents.supervisor_agent.init_chat_model") as mock_sup_init:
            # Mock the LLMs
            mock_llm = MagicMock()
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm.with_structured_output = MagicMock(return_value=mock_llm)
            mock_base_init.return_value = mock_llm
            mock_sup_init.return_value = mock_llm

            supervisor = SupervisorAgent(seed=456)

            # Check supervisor has seed
            assert supervisor.seed == 456

            # Check all sub-agents have seed
            assert supervisor.engineering_agent.seed == 456
            assert supervisor.hpc_agent.seed == 456
            assert supervisor.search_agent.seed == 456
            assert supervisor.rag_agent.seed == 456
            assert supervisor.arxiv_agent.seed == 456
            assert supervisor.cli_agent.seed == 456
            assert supervisor.prusa_agent.seed == 456


# ============================================================================
# ROUTING PROMPT TESTS
# ============================================================================


@pytest.mark.unit
def test_build_routing_prompt(_mock_agents):
    """Test routing prompt contains all agent names and routing guidelines."""
    agent = SupervisorAgent()
    prompt = agent._build_routing_prompt()

    # Check all agents are mentioned
    assert "engineering_agent" in prompt
    assert "hpc_agent" in prompt
    assert "search_agent" in prompt
    assert "rag_agent" in prompt
    assert "arxiv_agent" in prompt
    assert "prusa_agent" in prompt
    assert "cli_agent" in prompt
    assert "supervisor_response" in prompt

    # Check routing guidelines are present
    assert "documentation" in prompt.lower() or "how do i" in prompt.lower()


@pytest.mark.unit
def test_routing_prompt_contains_keywords(_mock_agents):
    """Test that routing prompt contains important domain keywords."""
    agent = SupervisorAgent()
    prompt = agent._build_routing_prompt()

    # Engineering keywords
    assert "optimization" in prompt.lower()
    assert "topology" in prompt.lower()

    # HPC keywords
    assert "slurm" in prompt.lower()
    assert "cluster" in prompt.lower()

    # Prusa keywords
    assert "printer" in prompt.lower()


# ============================================================================
# ROUTING TESTS - LLM-based decision making
# ============================================================================


@pytest.mark.unit
def test_supervisor_node_already_routed(_mock_agents):
    """Test supervisor always re-evaluates via LLM (loop-back architecture)."""
    mock_llm = create_mock_llm_with_structured_output(
        "FINISH", "Task already completed"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        state = SupervisorState(
            messages=[HumanMessage(content="test")], next="engineering_agent"
        )
        result = agent._supervisor_node(state)

        # With loop-back, supervisor always invokes LLM to re-evaluate
        assert result["next"] == "FINISH"


@pytest.mark.unit
def test_supervisor_routes_to_engineering_with_realistic_response():
    """Test supervisor routes optimization queries to engineering agent."""
    mock_llm = create_mock_llm_with_structured_output(
        "engineering_agent", "User asked about beam optimization"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [
                HumanMessage(
                    content="Optimize my beam design with 35% volume constraint"
                )
            ],
            "next": "",
        }

        result = agent._supervisor_node(state)

        assert result["next"] == "engineering_agent"
        assert "messages" in result


@pytest.mark.unit
def test_supervisor_routes_to_hpc(_mock_agents):
    """Test routing to HPC agent for cluster operations."""
    agent = SupervisorAgent()
    agent.routing_llm = create_mock_routing_llm("hpc_agent", "HPC job submission")

    state = SupervisorState(
        messages=[HumanMessage(content="Submit job.slurm to cluster")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "hpc_agent"


@pytest.mark.unit
def test_supervisor_routes_to_search_for_research_queries():
    """Test supervisor routes web research queries to search agent."""
    test_cases = [
        "Search for topology optimization papers from 2024",
        "Find research on SIMP method",
        "What are the latest papers on structural optimization?",
    ]

    for query in test_cases:
        mock_llm = create_mock_llm_with_structured_output(
            "search_agent", f"Research query: {query}"
        )

        with patch(
            "src.agents.supervisor_agent.init_chat_model", return_value=mock_llm
        ):
            agent = SupervisorAgent()

            state = {
                "messages": [HumanMessage(content=query)],
                "next": "",
            }

            result = agent._supervisor_node(state)
            assert result["next"] == "search_agent", f"Failed for query: {query}"


@pytest.mark.unit
def test_supervisor_routes_to_rag(_mock_agents):
    """Test routing to RAG agent for documentation queries."""
    agent = SupervisorAgent()
    agent.routing_llm = create_mock_routing_llm("rag_agent", "Documentation query")

    state = SupervisorState(
        messages=[HumanMessage(content="How do I submit a job on Euler?")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "rag_agent"


@pytest.mark.unit
def test_supervisor_routes_to_arxiv(_mock_agents):
    """Test routing to ArXiv agent for paper searches."""
    agent = SupervisorAgent()
    agent.routing_llm = create_mock_routing_llm("arxiv_agent", "ArXiv paper search")

    state = SupervisorState(
        messages=[HumanMessage(content="Find papers about transformers on ArXiv")],
        next="",
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "arxiv_agent"


@pytest.mark.unit
def test_supervisor_routes_to_prusa(_mock_agents):
    """Test routing to Prusa agent for printer management."""
    agent = SupervisorAgent()
    agent.routing_llm = create_mock_routing_llm("prusa_agent", "Printer management")

    state = SupervisorState(
        messages=[HumanMessage(content="Check printer status")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "prusa_agent"


@pytest.mark.unit
def test_supervisor_routes_to_cli_for_system_queries():
    """Test supervisor routes CLI commands and app opening to CLI agent."""
    mock_llm = create_mock_llm_with_structured_output("cli_agent", "CLI operation")

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content="open PrusaSlicer")],
            "next": "",
        }

        result = agent._supervisor_node(state)
        assert result["next"] == "cli_agent"


@pytest.mark.unit
def test_supervisor_responds_directly_for_capability_questions():
    """Test supervisor answers capability/help questions directly."""
    test_cases = [
        "What can you do?",
        "How do I use this system?",
        "Tell me about your capabilities",
    ]

    for query in test_cases:
        mock_llm = create_mock_llm_with_structured_output(
            "supervisor_response", "General capability question"
        )

        with patch(
            "src.agents.supervisor_agent.init_chat_model", return_value=mock_llm
        ):
            agent = SupervisorAgent()

            state = {
                "messages": [HumanMessage(content=query)],
                "next": "",
            }

            result = agent._supervisor_node(state)
            assert result["next"] == "supervisor_response", f"Failed for query: {query}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "agent_name,query",
    [
        ("engineering_agent", "Optimize a cantilever beam"),
        ("search_agent", "Find papers on topology optimization"),
        ("hpc_agent", "Submit job to cluster"),
        ("cli_agent", "List simulation files"),
        ("rag_agent", "How does SLURM work?"),
        ("supervisor_response", "What can you do?"),
    ],
)
def test_supervisor_routing_parametrized(agent_name, query):
    """Parametrized test for different routing scenarios."""
    mock_llm = create_mock_llm_with_structured_output(
        agent_name, f"Route to {agent_name}"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content=query)],
            "next": "",
        }

        result = agent._supervisor_node(state)
        assert result["next"] == agent_name


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.unit
def test_supervisor_handles_invalid_agent_name():
    """Test supervisor handles LLM returning invalid agent name via Pydantic validation."""
    from pydantic import ValidationError

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()

    mock_structured_llm.invoke.side_effect = ValidationError.from_exception_data(
        "ValidationError",
        [
            {
                "type": "literal_error",
                "loc": ("agent",),
                "msg": "Invalid agent name",
                "input": "invalid_agent",
                "ctx": {
                    "expected": "'engineering_agent', 'hpc_agent', 'search_agent', 'rag_agent', 'arxiv_agent', 'prusa_agent', 'cli_agent' or 'supervisor_response'"
                },
            }
        ],
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content="Do something")],
            "next": "",
        }

        with pytest.raises(ValidationError):
            agent._supervisor_node(state)


@pytest.mark.unit
def test_supervisor_handles_empty_message_list():
    """Test supervisor handles empty message list gracefully."""
    mock_llm = create_mock_llm_with_structured_output(
        "supervisor_response", "Empty message"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [],
            "next": "",
        }

        try:
            result = agent._supervisor_node(state)
            assert "next" in result
        except Exception as e:
            pytest.fail(f"Should handle empty messages gracefully, but raised: {e}")


@pytest.mark.unit
def test_supervisor_handles_very_long_message():
    """Test supervisor handles very long input messages."""
    mock_llm = create_mock_llm_with_structured_output(
        "engineering_agent", "Long optimization request"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        long_content = "Optimize my beam " + "with many constraints " * 1000

        state = {
            "messages": [HumanMessage(content=long_content)],
            "next": "",
        }

        result = agent._supervisor_node(state)
        assert result["next"] == "engineering_agent"


# ============================================================================
# STATE MANAGEMENT TESTS
# ============================================================================


@pytest.mark.unit
def test_supervisor_returns_empty_messages_with_routing():
    """Test supervisor returns empty messages (state managed by add_messages reducer)."""
    mock_llm = create_mock_llm_with_structured_output(
        "engineering_agent", "Optimization request"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        initial_messages = [
            HumanMessage(content="First message"),
            AIMessage(content="First response"),
            HumanMessage(content="Optimize my beam"),
        ]

        state = {
            "messages": initial_messages,
            "next": "",
        }

        result = agent._supervisor_node(state)

        # Supervisor returns empty messages (add_messages handles state)
        assert result.get("messages", []) == []
        assert result.get("next") == "engineering_agent"


@pytest.mark.unit
def test_supervisor_handles_multi_turn_conversation():
    """Test supervisor handles context from previous conversation turns."""
    # First turn
    mock_llm1 = create_mock_llm_with_structured_output(
        "engineering_agent", "Optimization request"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm1):
        agent = SupervisorAgent()

        state1 = {
            "messages": [HumanMessage(content="Optimize a beam")],
            "next": "",
        }
        result1 = agent._supervisor_node(state1)
        assert result1["next"] == "engineering_agent"

    # Second turn
    mock_llm2 = create_mock_llm_with_structured_output("search_agent", "Search request")

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm2):
        agent = SupervisorAgent()

        state2 = {
            "messages": [
                HumanMessage(content="Optimize a beam"),
                AIMessage(content="I've optimized it"),
                HumanMessage(content="Now search for similar papers"),
            ],
            "next": "",
        }
        result2 = agent._supervisor_node(state2)
        assert result2["next"] == "search_agent"


# ============================================================================
# AGENT NODE TESTS
# ============================================================================


@pytest.mark.unit
def test_engineering_node(_mock_agents):
    """Test engineering node delegation."""
    agent = SupervisorAgent()
    agent.engineering_agent = MagicMock()
    agent.engineering_agent.invoke.return_value = {
        "messages": [HumanMessage(content="test"), AIMessage(content="result")]
    }

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._engineering_node(state)

    assert "messages" in result
    assert result["next"] == ""


@pytest.mark.unit
def test_hpc_node(_mock_agents):
    """Test HPC node delegation."""
    agent = SupervisorAgent()
    agent.hpc_agent = MagicMock()
    agent.hpc_agent.invoke.return_value = {
        "messages": [HumanMessage(content="test"), AIMessage(content="result")]
    }

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._hpc_node(state)

    assert "messages" in result
    assert result["next"] == ""


@pytest.mark.unit
def test_search_node(_mock_agents):
    """Test search node delegation."""
    agent = SupervisorAgent()
    agent.search_agent = MagicMock()
    agent.search_agent.invoke.return_value = {
        "messages": [HumanMessage(content="test"), AIMessage(content="result")]
    }

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._search_node(state)

    assert "messages" in result
    assert result["next"] == ""


@pytest.mark.unit
def test_supervisor_response_node(_mock_agents):
    """Test supervisor response node for capability questions."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="I can help with...")

    state = SupervisorState(
        messages=[HumanMessage(content="what can you do?")], next=""
    )
    result = agent._supervisor_response_node(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["next"] == "FINISH"


# ============================================================================
# INVOKE & GRAPH TESTS
# ============================================================================


@pytest.mark.unit
def test_invoke_with_messages_state(_mock_agents):
    """Test invoke with MessagesState format."""
    agent = SupervisorAgent()
    agent.graph = MagicMock()
    agent.graph.invoke.return_value = {"messages": [AIMessage(content="response")]}

    state = {"messages": [HumanMessage(content="test")]}
    config = {"configurable": {"thread_id": "test"}}

    result = agent.invoke(state, config)

    assert "messages" in result


@pytest.mark.unit
def test_invoke_with_supervisor_state(_mock_agents):
    """Test invoke with SupervisorState format."""
    agent = SupervisorAgent()
    agent.graph = MagicMock()
    agent.graph.invoke.return_value = {"messages": [AIMessage(content="response")]}

    state = {"messages": [HumanMessage(content="test")], "next": ""}
    config = {"configurable": {"thread_id": "test"}}

    result = agent.invoke(state, config)

    assert "messages" in result


@pytest.mark.unit
def test_invoke_resume_from_interrupt(_mock_agents):
    """Test invoke with None state (resume from interrupt)."""
    agent = SupervisorAgent()
    agent.graph = MagicMock()
    agent.graph.invoke.return_value = {"messages": [AIMessage(content="response")]}

    config = {"configurable": {"thread_id": "test"}}

    result = agent.invoke(None, config)

    agent.graph.invoke.assert_called_once_with(None, config)
    assert "messages" in result


@pytest.mark.unit
def test_build_graph_creates_graph(_mock_agents):
    """Test that _build_graph creates a compiled graph."""
    agent = SupervisorAgent()

    assert agent.graph is not None
    assert hasattr(agent.graph, "invoke")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
def test_supervisor_full_routing_workflow():
    """Test complete workflow from user input to agent selection."""
    mock_llm = create_mock_llm_with_structured_output(
        "engineering_agent", "User wants beam optimization"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        # Mock sub-agents
        agent.engineering_agent = Mock(
            return_value={
                "messages": [AIMessage(content="Engineering work done")],
                "next": "FINISH",
            }
        )

        initial_state = {
            "messages": [
                HumanMessage(content="Optimize my beam with volume constraint")
            ],
            "next": "",
        }

        result = agent._supervisor_node(initial_state)

        assert result["next"] == "engineering_agent"
        assert "messages" in result


@pytest.mark.unit
def test_supervisor_routing_is_deterministic():
    """Test that same input produces same routing decision."""
    mock_llm = create_mock_llm_with_structured_output(
        "engineering_agent", "Optimization request"
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=mock_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content="Optimize my beam")],
            "next": "",
        }

        result1 = agent._supervisor_node(state)
        result2 = agent._supervisor_node(state)

        assert result1["next"] == result2["next"]


# ============================================================================
# MORE_STEPS_AFTER / ROUTE-AFTER-AGENT TESTS
# ============================================================================


@pytest.mark.unit
def test_route_after_agent_ends_when_no_followup(_mock_agents):
    """Single-task: agent used tools + more_steps_after=False → END directly."""
    agent = SupervisorAgent()
    # Simulate: agent used tools, supervisor said no followup
    agent._last_delegation_had_tools = True
    agent._expects_followup = False

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._route_after_agent(state)

    assert result == "__end__"  # langgraph.graph.END == "__end__"


@pytest.mark.unit
def test_route_after_agent_returns_to_supervisor_when_followup(_mock_agents):
    """Multi-step: agent used tools + more_steps_after=True → supervisor."""
    agent = SupervisorAgent()
    agent._last_delegation_had_tools = True
    agent._expects_followup = True

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._route_after_agent(state)

    assert result == "supervisor"


@pytest.mark.unit
def test_route_after_agent_ends_when_no_tools(_mock_agents):
    """Agent produced no tool calls → END regardless of more_steps_after."""
    agent = SupervisorAgent()
    agent._last_delegation_had_tools = False
    agent._expects_followup = True  # even True shouldn't matter

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._route_after_agent(state)

    assert result == "__end__"


@pytest.mark.unit
def test_more_steps_after_defaults_to_false(_mock_agents):
    """RouteDecision.more_steps_after defaults to False → END after tools."""
    agent = SupervisorAgent()

    # Simulate supervisor routing with default more_steps_after
    route_decision = RouteDecision(
        agent="engineering_agent",
        reasoning="Optimization task",
    )
    assert route_decision.more_steps_after is False


@pytest.mark.unit
def test_supervisor_node_sets_expects_followup_from_route_decision(_mock_agents):
    """Supervisor node stores more_steps_after in _expects_followup."""
    agent = SupervisorAgent()

    # Mock routing LLM to return more_steps_after=True (multi-step)
    route_decision = RouteDecision(
        agent="rag_agent",
        reasoning="Need to find params first",
        task_instruction="Find volfrac in the paper",
        more_steps_after=True,
    )
    mock_routing_llm = MagicMock()
    mock_routing_llm.invoke.return_value = route_decision
    agent.routing_llm = mock_routing_llm

    state = SupervisorState(
        messages=[HumanMessage(content="Find params and optimize")], next=""
    )
    agent._supervisor_node(state)

    assert agent._expects_followup is True


@pytest.mark.unit
def test_supervisor_node_sets_expects_followup_false_for_single_task(_mock_agents):
    """Supervisor node stores more_steps_after=False for single tasks."""
    agent = SupervisorAgent()

    route_decision = RouteDecision(
        agent="engineering_agent",
        reasoning="Simple optimization",
        more_steps_after=False,
    )
    mock_routing_llm = MagicMock()
    mock_routing_llm.invoke.return_value = route_decision
    agent.routing_llm = mock_routing_llm

    state = SupervisorState(messages=[HumanMessage(content="Optimize a beam")], next="")
    agent._supervisor_node(state)

    assert agent._expects_followup is False


@pytest.mark.unit
def test_skip_arxiv_routes_back_to_supervisor(_mock_agents):
    """SKIP_ARXIV bypass should route back to supervisor for re-routing."""
    agent = SupervisorAgent()

    with patch.dict("os.environ", {"SKIP_ARXIV": "true"}):
        state = SupervisorState(
            messages=[HumanMessage(content="Search arxiv")], next=""
        )
        agent._arxiv_node(state)

        # SKIP_ARXIV sets _last_delegation_had_tools=True so supervisor
        # can re-route (typically to rag_agent). The expects_followup
        # flag should not prevent this.
        assert agent._last_delegation_had_tools is True
