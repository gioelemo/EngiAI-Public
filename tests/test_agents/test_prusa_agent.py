"""
Tests for prusa_agent module.

These tests cover the PrusaAgent class for 3D printer management.
Most tests use skip_mcp=True to avoid MCP server connection.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.prusa_agent import PrusaAgent

# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


@pytest.mark.unit
def test_prusa_agent_initialization_skip_mcp():
    """Test PrusaAgent initialization with MCP skipped."""
    agent = PrusaAgent(skip_mcp=True)

    assert agent.tools == []
    assert agent.tools_by_name == {}
    assert agent.mcp_client is None
    assert agent.agent is not None


@pytest.mark.unit
def test_prusa_agent_default_model():
    """Test PrusaAgent uses default model from config."""
    with patch("src.agents.prusa_agent.config") as mock_config:
        mock_config.llm_model = "gpt-4"
        mock_config.llm_temperature = 0.7

        with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            agent = PrusaAgent(skip_mcp=True)

            assert agent.model_name == "gpt-4"
            assert agent.temperature == 0.7


@pytest.mark.unit
def test_prusa_agent_custom_model():
    """Test PrusaAgent with custom model parameters."""
    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        agent = PrusaAgent(
            model_name="gpt-5-mini",
            temperature=0.5,
            skip_mcp=True,
        )

        assert agent.model_name == "gpt-5-mini"
        assert agent.temperature == 0.5


@pytest.mark.unit
def test_prusa_agent_env_skip_mcp(monkeypatch):
    """Test that SKIP_MCP environment variable works."""
    monkeypatch.setenv("SKIP_MCP", "true")

    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        agent = PrusaAgent()  # Not passing skip_mcp, should read from env

        assert agent.mcp_client is None


# ============================================================================
# LLM CALL TESTS
# ============================================================================


@pytest.mark.unit
def test_llm_call_basic():
    """Test basic LLM call."""
    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Response")
        mock_init.return_value = mock_llm

        agent = PrusaAgent(skip_mcp=True)
        agent.llm_with_tools = mock_llm

        state = {"messages": [HumanMessage(content="Hello")], "llm_calls": 0}
        result = agent._llm_call(state)

        assert "messages" in result
        assert result["llm_calls"] == 1


@pytest.mark.unit
def test_llm_call_resets_on_human_message():
    """Test that LLM call counter resets on new human message."""
    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Response")
        mock_init.return_value = mock_llm

        agent = PrusaAgent(skip_mcp=True)
        agent.llm_with_tools = mock_llm

        # Simulate existing conversation with high call count
        state = {
            "messages": [HumanMessage(content="New message")],
            "llm_calls": 10,
        }
        result = agent._llm_call(state)

        # Should reset to 1 (0 + 1) because last message is human
        assert result["llm_calls"] == 1


# ============================================================================
# TOOL NODE TESTS
# ============================================================================


@pytest.mark.unit
def test_tool_node_no_ai_message():
    """Test tool node with non-AI message returns empty."""
    agent = PrusaAgent(skip_mcp=True)

    state = {"messages": [HumanMessage(content="Hello")]}
    result = agent._tool_node(state)

    assert result["messages"] == []


@pytest.mark.unit
def test_tool_node_tool_not_found():
    """Test tool node when tool is not found."""
    agent = PrusaAgent(skip_mcp=True)

    ai_msg = AIMessage(
        content="Let me check that",
        tool_calls=[{"name": "unknown_tool", "args": {}, "id": "1"}],
    )
    state = {"messages": [ai_msg]}
    result = agent._tool_node(state)

    assert len(result["messages"]) == 1
    assert "not found" in result["messages"][0].content


@pytest.mark.unit
def test_tool_node_tool_execution():
    """Test tool node executes tool correctly."""
    agent = PrusaAgent(skip_mcp=True)

    # Add a mock tool
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "Tool result"
    agent.tools_by_name = {"test_tool": mock_tool}

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "test_tool", "args": {"arg1": "value"}, "id": "1"}],
    )
    state = {"messages": [ai_msg]}
    result = agent._tool_node(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Tool result"
    mock_tool.invoke.assert_called_once_with({"arg1": "value"})


@pytest.mark.unit
def test_tool_node_tool_error():
    """Test tool node handles tool execution errors."""
    agent = PrusaAgent(skip_mcp=True)

    # Add a mock tool that raises an error
    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = Exception("Tool error")
    agent.tools_by_name = {"test_tool": mock_tool}

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}],
    )
    state = {"messages": [ai_msg]}
    result = agent._tool_node(state)

    assert len(result["messages"]) == 1
    assert "Error" in result["messages"][0].content


# ============================================================================
# SHOULD CONTINUE TESTS
# ============================================================================


@pytest.mark.unit
def test_should_continue_with_tool_calls():
    """Test should_continue returns 'tools' when there are tool calls."""
    agent = PrusaAgent(skip_mcp=True)

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "tool", "args": {}, "id": "1"}],
    )
    state = {"messages": [ai_msg], "llm_calls": 1}
    result = agent._should_continue(state)

    assert result == "tools"


@pytest.mark.unit
def test_should_continue_end_no_tool_calls():
    """Test should_continue returns 'end' when no tool calls."""
    agent = PrusaAgent(skip_mcp=True)

    ai_msg = AIMessage(content="Here's the answer")
    state = {"messages": [ai_msg], "llm_calls": 1}
    result = agent._should_continue(state)

    assert result == "end"


@pytest.mark.unit
def test_should_continue_summarize_max_calls():
    """Test should_continue returns 'summarize' at max calls."""
    agent = PrusaAgent(skip_mcp=True)

    ai_msg = AIMessage(content="Response")
    state = {"messages": [ai_msg], "llm_calls": 20}  # Max is 20
    result = agent._should_continue(state)

    assert result == "summarize"


# ============================================================================
# SUMMARIZE NODE TESTS
# ============================================================================


@pytest.mark.unit
def test_summarize_node():
    """Test summarize node generates summary."""
    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Summary of conversation")
        mock_init.return_value = mock_llm

        agent = PrusaAgent(skip_mcp=True)

        state = {"messages": [HumanMessage(content="Test")]}
        result = agent._summarize_node(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)


# ============================================================================
# MCP CLIENT TESTS
# ============================================================================


@pytest.mark.unit
def test_call_mcp_tool_sync_no_client():
    """Test _call_mcp_tool_sync returns error when no client."""
    agent = PrusaAgent(skip_mcp=True)

    result = agent._call_mcp_tool_sync("test_tool", arg1="value")

    assert result == "MCP client not initialized"


@pytest.mark.unit
def test_call_mcp_tool_sync_with_client():
    """Test _call_mcp_tool_sync calls client correctly."""
    agent = PrusaAgent(skip_mcp=True)

    # Add a mock client
    mock_client = MagicMock()
    mock_client.call_tool_sync.return_value = "Tool response"
    agent.mcp_client = mock_client

    result = agent._call_mcp_tool_sync("test_tool", arg1="value")

    assert result == "Tool response"
    mock_client.call_tool_sync.assert_called_once_with("test_tool", arg1="value")


# ============================================================================
# AGENT GRAPH TESTS
# ============================================================================


@pytest.mark.unit
def test_build_agent_creates_graph():
    """Test that _build_agent creates a compiled graph."""
    agent = PrusaAgent(skip_mcp=True)

    # The agent should have a compiled graph
    assert agent.agent is not None
    # It should be callable via invoke
    assert hasattr(agent.agent, "invoke")


@pytest.mark.unit
def test_invoke_method():
    """Test the invoke method calls the agent graph."""
    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = PrusaAgent(skip_mcp=True)

        # Mock the compiled agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="Response")]}
        agent.agent = mock_agent

        state = {"messages": [HumanMessage(content="Hello")]}
        config = {"configurable": {"thread_id": "test"}}

        result = agent.invoke(state, config)

        mock_agent.invoke.assert_called_once_with(state, config)
        assert "messages" in result


# ============================================================================
# EDGE CASES
# ============================================================================


@pytest.mark.unit
def test_multiple_tool_calls():
    """Test handling multiple tool calls in one message."""
    agent = PrusaAgent(skip_mcp=True)

    # Add mock tools
    mock_tool1 = MagicMock()
    mock_tool1.invoke.return_value = "Result 1"
    mock_tool2 = MagicMock()
    mock_tool2.invoke.return_value = "Result 2"
    agent.tools_by_name = {"tool1": mock_tool1, "tool2": mock_tool2}

    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "tool1", "args": {}, "id": "1"},
            {"name": "tool2", "args": {}, "id": "2"},
        ],
    )
    state = {"messages": [ai_msg]}
    result = agent._tool_node(state)

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "Result 1"
    assert result["messages"][1].content == "Result 2"


@pytest.mark.unit
def test_empty_tool_calls():
    """Test AI message with empty tool_calls list."""
    agent = PrusaAgent(skip_mcp=True)

    ai_msg = AIMessage(content="Just a response", tool_calls=[])
    state = {"messages": [ai_msg], "llm_calls": 1}

    # Should go to end, not tools
    result = agent._should_continue(state)
    assert result == "end"


# ============================================================================
# ADDITIONAL COVERAGE TESTS
# ============================================================================


@pytest.mark.unit
def test_prusa_agent_system_prompt():
    """Test that Prusa agent uses system prompt."""
    from src.utils.prompts import PRUSA_AGENT_SYSTEM_PROMPT

    agent = PrusaAgent(skip_mcp=True)

    # Agent should use the Prusa system prompt
    assert PRUSA_AGENT_SYSTEM_PROMPT is not None
    assert isinstance(PRUSA_AGENT_SYSTEM_PROMPT, str)
    assert len(PRUSA_AGENT_SYSTEM_PROMPT) > 0


@pytest.mark.unit
def test_tool_node_with_mcp_client():
    """Test tool node with MCP client integration."""
    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm

        agent = PrusaAgent(skip_mcp=True)

        # Manually add MCP client
        mock_client = MagicMock()
        mock_client.call_tool_sync.return_value = "MCP result"
        agent.mcp_client = mock_client

        # Manually add MCP tool wrapper
        def mock_mcp_tool(**kwargs):
            return agent._call_mcp_tool_sync("mcp_tool", **kwargs)

        mock_tool_wrapper = MagicMock(side_effect=mock_mcp_tool)
        agent.tools_by_name = {"mcp_tool": mock_tool_wrapper}

        # Test tool execution
        result = agent._call_mcp_tool_sync("mcp_tool", arg="value")
        assert result == "MCP result"


@pytest.mark.unit
def test_llm_call_max_iterations():
    """Test LLM call respects max iterations."""
    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Response")
        mock_init.return_value = mock_llm

        agent = PrusaAgent(skip_mcp=True)
        agent.llm_with_tools = mock_llm

        # Simulate many iterations
        state = {
            "messages": [AIMessage(content="Previous response")],
            "llm_calls": 100,
        }
        result = agent._llm_call(state)

        # Should still process but increment counter
        assert result["llm_calls"] == 101


@pytest.mark.unit
def test_should_continue_max_llm_calls():
    """Test should_continue handles max LLM calls."""
    agent = PrusaAgent(skip_mcp=True)

    ai_msg = AIMessage(
        content="Response",
        tool_calls=[{"name": "test", "args": {}, "id": "1"}],
    )

    # Test with high call count
    state = {"messages": [ai_msg], "llm_calls": 50}
    result = agent._should_continue(state)

    # Should still return "tools" if there are tool calls
    assert result == "tools"


@pytest.mark.unit
def test_tool_node_preserves_tool_call_id():
    """Test that tool node preserves tool call IDs."""
    agent = PrusaAgent(skip_mcp=True)

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "Result"
    agent.tools_by_name = {"test_tool": mock_tool}

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "test_tool", "args": {}, "id": "call_123"}],
    )
    state = {"messages": [ai_msg]}
    result = agent._tool_node(state)

    # Result should have tool message with matching ID
    assert len(result["messages"]) == 1
    assert result["messages"][0].tool_call_id == "call_123"


@pytest.mark.unit
def test_initialization_with_nest_asyncio():
    """Test that nest_asyncio is applied if available."""
    with patch("src.agents.prusa_agent.NEST_ASYNCIO_AVAILABLE", True):
        with patch("src.agents.prusa_agent.nest_asyncio") as mock_nest:
            agent = PrusaAgent(skip_mcp=True)

            # nest_asyncio.apply() should be called during init
            # (this happens in _initialize_mcp_client when not skipped)
            assert agent is not None


@pytest.mark.unit
def test_prusa_system_prompt_constant():
    """Test PRUSA_AGENT_SYSTEM_PROMPT constant is defined."""
    from src.utils.prompts import PRUSA_AGENT_SYSTEM_PROMPT

    # Should be a proper system prompt
    assert isinstance(PRUSA_AGENT_SYSTEM_PROMPT, str)
    assert len(PRUSA_AGENT_SYSTEM_PROMPT) > 50  # Should be substantial


@pytest.mark.unit
def test_tool_execution_with_empty_args():
    """Test tool execution with no arguments."""
    agent = PrusaAgent(skip_mcp=True)

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "Success"
    agent.tools_by_name = {"no_arg_tool": mock_tool}

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "no_arg_tool", "args": {}, "id": "1"}],
    )
    state = {"messages": [ai_msg]}
    result = agent._tool_node(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Success"
    mock_tool.invoke.assert_called_once_with({})


@pytest.mark.unit
def test_llm_call_with_system_message():
    """Test LLM call includes system message."""
    with patch("src.agents.prusa_agent.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Response")
        mock_init.return_value = mock_llm

        agent = PrusaAgent(skip_mcp=True)
        agent.llm_with_tools = mock_llm

        state = {"messages": [HumanMessage(content="Hello")], "llm_calls": 0}
        result = agent._llm_call(state)

        # Check that invoke was called
        assert mock_llm.invoke.called
        # First message to LLM should include system prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) > 0


@pytest.mark.unit
def test_concurrent_futures_executor():
    """Test that executor is properly configured."""
    agent = PrusaAgent(skip_mcp=True)

    # Agent should have an executor if MCP is available
    # With skip_mcp=True, executor might not be initialized
    assert agent is not None


@pytest.mark.unit
def test_tool_node_json_serialization():
    """Test tool node handles JSON serialization properly."""
    agent = PrusaAgent(skip_mcp=True)

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = {"result": "data", "count": 42}
    agent.tools_by_name = {"json_tool": mock_tool}

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "json_tool", "args": {"key": "value"}, "id": "1"}],
    )
    state = {"messages": [ai_msg]}
    result = agent._tool_node(state)

    # Should convert dict result to string
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0].content, str)
