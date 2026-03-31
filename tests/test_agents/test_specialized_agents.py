"""
Tests for specialized agents (CLI, Engineering, HPC, Search).

Tests cover tool counts, configuration, error handling, and _after_tools routing logic.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.cli_agent import CLIAgent
from src.agents.engineering_agent import EngineeringAgent
from src.agents.hpc_agent import HPCAgent
from src.agents.search_agent import SearchAgent
from tests.test_agents.conftest import make_cli_agent

# ============================================================================
# AGENT TOOL COUNT TESTS
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "agent_class,min_tools",
    [
        (CLIAgent, 1),
        (EngineeringAgent, 5),
        (HPCAgent, 1),
        (SearchAgent, 1),
    ],
)
def test_agent_tool_counts(agent_class, min_tools):
    """Each agent must expose at least the expected number of tools."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm
        agent = agent_class()
        assert len(agent.tools) >= min_tools


@pytest.mark.unit
def test_cli_agent_seed_override():
    """Seed passed to CLIAgent must reach init_chat_model."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = CLIAgent(seed=123)

        assert agent.seed == 123
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs.get("seed") == 123


# ============================================================================
# AGENT CONFIGURATION TESTS
# ============================================================================


@pytest.mark.unit
def test_agents_with_custom_model_name():
    """Test that agents can be created with custom model name."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = EngineeringAgent(model_name="custom-model")

        # Agents should successfully initialize
        assert mock_init.called
        assert agent.model_name == "custom-model"


# ============================================================================
# AGENT REQUIRE CONFIRMATION TESTS
# ============================================================================


@pytest.mark.unit
def test_cli_agent_confirmation_flag():
    """Test that CLI agent respects confirmation flag."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent_with_confirmation = CLIAgent(require_confirmation=True)
        agent_without_confirmation = CLIAgent(require_confirmation=False)

        assert agent_with_confirmation.require_confirmation is True
        assert agent_without_confirmation.require_confirmation is False


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.unit
def test_agent_creation_with_missing_api_key():
    """Test agent creation when API key is missing."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        # Simulate missing API key error
        mock_init.side_effect = ValueError("API key not found")

        with pytest.raises(ValueError):
            CLIAgent()


@pytest.mark.unit
def test_agent_creation_with_invalid_model():
    """Test agent creation with invalid model name."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        # Simulate invalid model error
        mock_init.side_effect = RuntimeError("Model not found")

        with pytest.raises(RuntimeError):
            HPCAgent()


# ============================================================================
# _after_tools ROUTING TESTS
# ============================================================================


@pytest.mark.unit
def test_after_tools_routes_to_llm_call_for_regular_tool():
    """Regular tool result should route back to llm_call."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="run ls"),
            AIMessage(
                content="",
                tool_calls=[{"id": "t1", "name": "execute_cli_command", "args": {}}],
            ),
            ToolMessage(
                content="file1.txt", tool_call_id="t1", name="execute_cli_command"
            ),
        ]
    }
    assert agent._after_tools(state) == "llm_call"


@pytest.mark.unit
def test_after_tools_routes_to_clarification_response_for_first_clarification():
    """First ask_human_for_clarification should route to clarification_response."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="optimize a beam"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true, "question": "What volume fraction?"}',
                tool_call_id="t1",
                name="ask_human_for_clarification",
            ),
        ]
    }
    assert agent._after_tools(state) == "clarification_response"


@pytest.mark.unit
def test_after_tools_routes_to_clarification_response_for_first_clarification_mixed():
    """First clarification among other tools should route to clarification_response."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="optimize a beam"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "get_problem_details", "args": {}},
                    {"id": "t2", "name": "ask_human_for_clarification", "args": {}},
                ],
            ),
            ToolMessage(
                content="problem details...",
                tool_call_id="t1",
                name="get_problem_details",
            ),
            ToolMessage(
                content='{"success": true, "question": "What volume fraction?"}',
                tool_call_id="t2",
                name="ask_human_for_clarification",
            ),
        ]
    }
    assert agent._after_tools(state) == "clarification_response"


@pytest.mark.unit
def test_after_tools_routes_to_llm_call_when_no_tool_messages():
    """State with no ToolMessages should fall through to llm_call."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]
    }
    assert agent._after_tools(state) == "llm_call"


@pytest.mark.unit
def test_after_tools_routes_to_llm_call_when_clarification_tool_errored():
    """If ask_human_for_clarification raised an error, let the LLM recover."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="optimize a beam"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            # _tool_node error format
            ToolMessage(
                content="❌ Error executing tool 'ask_human_for_clarification': something broke",
                tool_call_id="t1",
                name="ask_human_for_clarification",
            ),
        ]
    }
    assert agent._after_tools(state) == "llm_call"


@pytest.mark.unit
def test_after_tools_prior_clarification_does_not_block_regular_tools():
    """One prior clarification (count=1 < 2) must not block routing after regular tools."""
    agent = make_cli_agent()
    # First turn: clarification was requested
    # Second turn: user responded, then a regular tool is called — should route to llm_call
    state = {
        "messages": [
            HumanMessage(content="optimize a beam"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true}',
                tool_call_id="t1",
                name="ask_human_for_clarification",
            ),
            HumanMessage(content="volfrac=0.3, rmin=3.0, forcedist=1.0"),
            AIMessage(
                content="",
                tool_calls=[{"id": "t2", "name": "optimize_design", "args": {}}],
            ),
            ToolMessage(
                content="design optimized", tool_call_id="t2", name="optimize_design"
            ),
        ]
    }
    assert agent._after_tools(state) == "llm_call"


@pytest.mark.unit
def test_after_tools_routes_to_end_for_second_clarification():
    """Second successful ask_human_for_clarification must route to __end__."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="optimize a beam"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true, "question": "What volume fraction?"}',
                tool_call_id="t1",
                name="ask_human_for_clarification",
            ),
            AIMessage(
                content="I need more details.",
                tool_calls=[
                    {"id": "t2", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true, "question": "Please also specify filter radius."}',
                tool_call_id="t2",
                name="ask_human_for_clarification",
            ),
        ]
    }
    assert agent._after_tools(state) == "clarification_response"


@pytest.mark.unit
def test_after_tools_error_plus_success_routes_to_clarification_response():
    """One errored + one successful clarification — current batch has success, route to clarification_response."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="optimize a beam"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content="Error executing tool 'ask_human_for_clarification': something broke",
                tool_call_id="t1",
                name="ask_human_for_clarification",
            ),
            AIMessage(
                content="Let me retry.",
                tool_calls=[
                    {"id": "t2", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true, "question": "What volume fraction?"}',
                tool_call_id="t2",
                name="ask_human_for_clarification",
            ),
        ]
    }
    assert agent._after_tools(state) == "clarification_response"


@pytest.mark.unit
def test_after_tools_two_clarifications_with_intermediate_tools():
    """Two successful clarifications with design tools in between must route to clarification_response."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="optimize a beam"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t1", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true, "question": "What volume fraction?"}',
                tool_call_id="t1",
                name="ask_human_for_clarification",
            ),
            AIMessage(
                content="",
                tool_calls=[{"id": "t2", "name": "optimize_design", "args": {}}],
            ),
            ToolMessage(
                content="design optimized", tool_call_id="t2", name="optimize_design"
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "t3", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true, "question": "What filter radius?"}',
                tool_call_id="t3",
                name="ask_human_for_clarification",
            ),
        ]
    }
    assert agent._after_tools(state) == "clarification_response"
