"""
Tests for specialized agents (CLI, Engineering, HPC, Search).

These tests verify agent creation, tool binding, and basic configuration
without requiring full LangGraph execution.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import config
from src.agents.cli_agent import CLIAgent
from src.agents.engineering_agent import EngineeringAgent
from src.agents.hpc_agent import HPCAgent
from src.agents.search_agent import SearchAgent

# ============================================================================
# CLI AGENT TESTS
# ============================================================================


@pytest.mark.unit
def test_cli_agent_creation():
    """Test that CLI agent can be created."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = CLIAgent()

        assert agent is not None
        assert agent.model_name is not None
        mock_init.assert_called_once()


@pytest.mark.unit
def test_cli_agent_has_tools():
    """Test that CLI agent is bound with correct tools."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = CLIAgent()

        # Verify tools were bound
        mock_llm.bind_tools.assert_called_once()
        assert len(agent.tools) > 0


@pytest.mark.unit
def test_cli_agent_seed_override():
    """Test that CLI agent accepts seed override."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = CLIAgent(seed=123)

        assert agent.seed == 123
        # Verify seed was passed to init_chat_model
        call_kwargs = mock_init.call_args[1]
        assert "seed" in call_kwargs
        assert call_kwargs["seed"] == 123


@pytest.mark.unit
def test_engineering_agent_seed_default():
    """Test that Engineering agent uses config default seed."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = EngineeringAgent()

        # Should use config default (which is None in tests)
        assert agent.seed == config.llm_seed


# ============================================================================
# ENGINEERING AGENT TESTS
# ============================================================================


@pytest.mark.unit
def test_engineering_agent_creation():
    """Test that Engineering agent can be created."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = EngineeringAgent()

        assert agent is not None
        assert agent.model_name is not None
        mock_init.assert_called_once()


@pytest.mark.unit
def test_engineering_agent_has_engibench_tools():
    """Test that Engineering agent has EngiBench tools."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = EngineeringAgent()

        # Verify tools were bound
        mock_llm.bind_tools.assert_called_once()
        # Should have multiple EngiBench tools
        assert len(agent.tools) >= 5


# ============================================================================
# HPC AGENT TESTS
# ============================================================================


@pytest.mark.unit
def test_hpc_agent_creation():
    """Test that HPC agent can be created."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = HPCAgent()

        assert agent is not None
        assert agent.model_name is not None
        mock_init.assert_called_once()


@pytest.mark.unit
def test_hpc_agent_has_slurm_tools():
    """Test that HPC agent has SLURM tools."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = HPCAgent()

        # Verify tools were bound
        mock_llm.bind_tools.assert_called_once()
        # Should have HPC-related tools
        assert len(agent.tools) > 0


# ============================================================================
# SEARCH AGENT TESTS
# ============================================================================


@pytest.mark.unit
def test_search_agent_creation():
    """Test that Search agent can be created."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = SearchAgent()

        assert agent is not None
        assert agent.model_name is not None
        mock_init.assert_called_once()


@pytest.mark.unit
def test_search_agent_has_search_tool():
    """Test that Search agent has search tool."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = SearchAgent()

        # Verify tools were bound
        mock_llm.bind_tools.assert_called_once()
        assert len(agent.tools) > 0


# ============================================================================
# AGENT CONFIGURATION TESTS
# ============================================================================


@pytest.mark.unit
def test_agents_use_config_model():
    """Test that agents use model from config."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        agent = CLIAgent()

        # Verify init_chat_model was called (which uses config)
        mock_init.assert_called_once()
        assert agent.model_name is not None


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


@pytest.mark.unit
def test_engineering_agent_confirmation_flag():
    """Test that Engineering agent can be created."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm

        # EngineeringAgent doesn't have require_confirmation parameter
        agent = EngineeringAgent()
        assert agent is not None
        assert agent.model_name is not None


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


def _make_agent():
    """Return a CLIAgent with a mocked LLM for routing tests."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm
        return CLIAgent()


@pytest.mark.unit
def test_after_tools_routes_to_llm_call_for_regular_tool():
    """Regular tool result should route back to llm_call."""
    agent = _make_agent()
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
def test_after_tools_routes_to_end_for_clarification_tool():
    """ask_human_for_clarification result must route to __end__, not llm_call."""
    agent = _make_agent()
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
    assert agent._after_tools(state) == "__end__"


@pytest.mark.unit
def test_after_tools_routes_to_end_when_clarification_mixed_with_other_tools():
    """If clarification tool is among the last batch of tool calls, still route to __end__."""
    agent = _make_agent()
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
    assert agent._after_tools(state) == "__end__"


@pytest.mark.unit
def test_after_tools_routes_to_llm_call_when_no_tool_messages():
    """State with no ToolMessages should fall through to llm_call."""
    agent = _make_agent()
    state = {
        "messages": [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]
    }
    assert agent._after_tools(state) == "llm_call"


@pytest.mark.unit
def test_after_tools_only_inspects_trailing_tool_messages():
    """Clarification from a previous turn (not trailing) must not affect current routing."""
    agent = _make_agent()
    # First turn: clarification was requested
    # Second turn: a regular tool is called — should route to llm_call
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
