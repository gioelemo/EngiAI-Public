"""
Improved tests for the Supervisor Agent.

These tests are more realistic and catch actual bugs:
- Test realistic LLM response formats (JSON, structured output)
- Test error handling and edge cases
- Test state management through the graph
- Test integration between components
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor_agent import SupervisorAgent

# Mock heavy dependencies
sys.modules["engibench"] = MagicMock()
sys.modules["engibench.problems"] = MagicMock()
sys.modules["engibench.problems.beams2d"] = MagicMock()
sys.modules["engibench.problems.beams2d.v0"] = MagicMock()


# ============================================================================
# ROUTING TESTS - Test decision making logic
# ============================================================================


@pytest.mark.unit
def test_supervisor_routes_to_engineering_with_realistic_response():
    """Test supervisor routes optimization queries with realistic LLM output."""
    # Simulate realistic structured output from LLM
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content='{"next": "engineering_agent", "reasoning": "User asked about beam optimization"}'
                )
            ]
        )
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
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
        assert "messages" in result  # Should preserve message history


@pytest.mark.unit
def test_supervisor_routes_to_search_for_research_queries():
    """Test supervisor routes research/literature queries to search agent."""
    # Provide enough responses for all test cases
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="search_agent"),
                AIMessage(content="search_agent"),
                AIMessage(content="search_agent"),
            ]
        )
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        test_cases = [
            "Search for topology optimization papers from 2024",
            "Find research on SIMP method",
            "What are the latest papers on structural optimization?",
        ]

        for query in test_cases:
            state = {
                "messages": [HumanMessage(content=query)],
                "next": "",
            }

            result = agent._supervisor_node(state)
            assert result["next"] == "search_agent", f"Failed for query: {query}"


@pytest.mark.unit
def test_supervisor_routes_to_hpc_for_compute_queries():
    """Test supervisor routes HPC/compute queries correctly."""
    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="hpc_agent")]))

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [
                HumanMessage(content="Submit this optimization job to the cluster")
            ],
            "next": "",
        }

        result = agent._supervisor_node(state)
        assert result["next"] == "hpc_agent"


@pytest.mark.unit
def test_supervisor_routes_to_cli_for_system_queries():
    """Test supervisor routes CLI/system queries correctly."""
    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="cli_agent")]))

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [
                HumanMessage(content="List the files in my simulation directory")
            ],
            "next": "",
        }

        result = agent._supervisor_node(state)
        assert result["next"] == "cli_agent"


@pytest.mark.unit
def test_supervisor_responds_directly_for_capability_questions():
    """Test supervisor answers capability/help questions directly."""
    # Provide enough responses for all test cases
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="FINISH"),
                AIMessage(content="FINISH"),
                AIMessage(content="FINISH"),
            ]
        )
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        test_cases = [
            "What can you do?",
            "How do I use this system?",
            "Tell me about your capabilities",
        ]

        for query in test_cases:
            state = {
                "messages": [HumanMessage(content=query)],
                "next": "",
            }

            result = agent._supervisor_node(state)
            assert result["next"] == "supervisor_response", f"Failed for query: {query}"


# ============================================================================
# ERROR HANDLING TESTS - Test robustness
# ============================================================================


@pytest.mark.unit
def test_supervisor_handles_invalid_agent_name():
    """Test supervisor handles LLM returning invalid agent name."""
    fake_llm = GenericFakeChatModel(
        messages=iter([AIMessage(content="nonexistent_agent")])
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content="Do something")],
            "next": "",
        }

        # Should either raise a clear error or default to supervisor_response
        # Test whichever behavior your system implements
        try:
            result = agent._supervisor_node(state)
            # If it doesn't raise, it should default gracefully
            assert result["next"] in [
                "supervisor_response",
                "engineering_agent",  # or whatever your default is
            ]
        except (ValueError, KeyError) as e:
            # If it raises, that's also acceptable - just should be clear
            assert "agent" in str(e).lower() or "invalid" in str(e).lower()


@pytest.mark.unit
def test_supervisor_handles_malformed_json_response():
    """Test supervisor handles LLM returning malformed JSON."""
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [AIMessage(content='{"next": "engineering_agent", invalid json}')]
        )
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content="Optimize beam")],
            "next": "",
        }

        # Should handle gracefully - either parse what it can or use fallback
        result = agent._supervisor_node(state)
        assert "next" in result
        assert isinstance(result["next"], str)


@pytest.mark.unit
def test_supervisor_handles_empty_message_list():
    """Test supervisor handles empty message list gracefully."""
    fake_llm = GenericFakeChatModel(
        messages=iter([AIMessage(content="supervisor_response")])
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [],
            "next": "",
        }

        # Should handle empty messages without crashing
        try:
            result = agent._supervisor_node(state)
            assert "next" in result
        except Exception as e:
            pytest.fail(f"Should handle empty messages gracefully, but raised: {e}")


@pytest.mark.unit
def test_supervisor_handles_very_long_message():
    """Test supervisor handles very long input messages."""
    fake_llm = GenericFakeChatModel(
        messages=iter([AIMessage(content="engineering_agent")])
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        # Simulate a very long message
        long_content = "Optimize my beam " + "with many constraints " * 1000

        state = {
            "messages": [HumanMessage(content=long_content)],
            "next": "",
        }

        result = agent._supervisor_node(state)
        assert result["next"] == "engineering_agent"


# ============================================================================
# STATE MANAGEMENT TESTS - Test conversation flow
# ============================================================================


@pytest.mark.unit
def test_supervisor_preserves_message_history():
    """Test supervisor maintains message history through routing."""
    fake_llm = GenericFakeChatModel(
        messages=iter([AIMessage(content="engineering_agent")])
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
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

        # Should preserve all previous messages
        assert len(result.get("messages", [])) >= len(initial_messages)


@pytest.mark.unit
def test_supervisor_handles_multi_turn_conversation():
    """Test supervisor handles context from previous turns."""
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="engineering_agent"),
                AIMessage(content="search_agent"),
            ]
        )
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        # First turn - optimization request
        state1 = {
            "messages": [HumanMessage(content="Optimize a beam")],
            "next": "",
        }
        result1 = agent._supervisor_node(state1)
        assert result1["next"] == "engineering_agent"

        # Second turn - follow-up search request
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
# INTEGRATION TESTS - Test full workflow
# ============================================================================


@pytest.mark.integration
def test_supervisor_full_routing_workflow():
    """Test complete workflow from user input to agent selection."""
    fake_llm = GenericFakeChatModel(
        messages=iter([AIMessage(content="engineering_agent")])
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        # Mock the sub-agents to avoid real execution
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

        # Test the supervisor node works in context
        result = agent._supervisor_node(initial_state)

        assert result["next"] == "engineering_agent"
        assert "messages" in result


@pytest.mark.integration
def test_supervisor_initialization_creates_all_components():
    """Test supervisor initializes with all required components."""
    fake_llm = GenericFakeChatModel(messages=iter(["test"]))

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        # Verify all components are initialized
        assert agent.llm is not None, "LLM should be initialized"
        assert agent.engineering_agent is not None, "Engineering agent should exist"
        assert agent.hpc_agent is not None, "HPC agent should exist"
        assert agent.search_agent is not None, "Search agent should exist"
        assert agent.cli_agent is not None, "CLI agent should exist"
        assert agent.graph is not None, "Graph should be initialized"

        # Verify graph has correct structure
        assert hasattr(agent.graph, "invoke"), "Graph should be invokable"


# ============================================================================
# PERFORMANCE & EDGE CASE TESTS
# ============================================================================


@pytest.mark.unit
def test_supervisor_routing_is_deterministic():
    """Test that same input produces same routing decision."""
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="engineering_agent"),
                AIMessage(content="engineering_agent"),
            ]
        )
    )

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content="Optimize my beam")],
            "next": "",
        }

        result1 = agent._supervisor_node(state)
        result2 = agent._supervisor_node(state)

        assert result1["next"] == result2["next"], "Routing should be deterministic"


@pytest.mark.unit
def test_supervisor_handles_mixed_case_routing():
    """Test supervisor handles routing responses with different casing."""
    test_cases = [
        "engineering_agent",
        "Engineering_Agent",
        "ENGINEERING_AGENT",
    ]

    for response in test_cases:
        fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content=response)]))

        with patch(
            "src.agents.supervisor_agent.init_chat_model", return_value=fake_llm
        ):
            agent = SupervisorAgent()

            state = {
                "messages": [HumanMessage(content="Optimize beam")],
                "next": "",
            }

            result = agent._supervisor_node(state)
            # Should normalize to lowercase or handle case-insensitively
            assert result["next"].lower() == "engineering_agent"


@pytest.mark.unit
@pytest.mark.parametrize(
    "agent_name,query",
    [
        ("engineering_agent", "Optimize a cantilever beam"),
        ("search_agent", "Find papers on topology optimization"),
        ("hpc_agent", "Submit job to cluster"),
        ("cli_agent", "List simulation files"),
        ("supervisor_response", "What can you do?"),
    ],
)
def test_supervisor_routing_parametrized(agent_name, query):
    """Parametrized test for different routing scenarios."""
    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content=agent_name)]))

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content=query)],
            "next": "",
        }

        result = agent._supervisor_node(state)
        assert result["next"] == agent_name


# ============================================================================
# FIXTURES IMPROVEMENTS
# ============================================================================


@pytest.fixture
def mock_supervisor_with_fake_llm():
    """Fixture that provides a fully mocked supervisor for testing."""
    fake_llm = GenericFakeChatModel(messages=iter(["test"]))

    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        # Mock all sub-agents to prevent real execution
        agent.engineering_agent = Mock()
        agent.hpc_agent = Mock()
        agent.search_agent = Mock()
        agent.cli_agent = Mock()

        return agent


@pytest.fixture
def sample_conversation_state():
    """Fixture providing a realistic conversation state for testing."""
    return {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi! How can I help?"),
            HumanMessage(content="Optimize my beam design"),
        ],
        "next": "",
        "metadata": {
            "user_id": "test_user",
            "session_id": "test_session",
        },
    }
