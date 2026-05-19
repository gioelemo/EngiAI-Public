"""
Tests for BaseAgent core logic methods.

Covers edge cases and error paths in _tool_node, _should_continue,
and _clarification_response that are not exercised by other test files.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from tests.test_agents.conftest import make_cli_agent

# --- _tool_node ---


@pytest.mark.unit
def test_tool_node_exception_becomes_tool_message():
    """When a tool raises, _tool_node must return a ToolMessage with the error text.

    OpenAI rejects responses that leave a tool_call_id without a corresponding
    ToolMessage, so errors must always be wrapped.
    """
    agent = make_cli_agent()

    # Register a fake tool that always raises
    @tool
    def boom(x: str) -> str:  # noqa: ARG001
        """A tool that explodes."""
        raise ValueError("kaboom")

    agent.tools = [boom]
    agent.tools_by_name = {"boom": boom}

    state = {
        "messages": [
            HumanMessage(content="do something"),
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "boom", "args": {"x": "go"}}],
            ),
        ]
    }
    result = agent._tool_node(state)
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert msgs[0].tool_call_id == "tc1"
    assert "❌ Error executing tool 'boom'" in msgs[0].content
    assert "kaboom" in msgs[0].content


@pytest.mark.unit
def test_tool_node_non_ai_last_message_returns_empty():
    """_tool_node must return empty messages when last message is not an AIMessage."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="hello"),
        ]
    }
    result = agent._tool_node(state)
    assert result == {"messages": []}


# --- _should_continue ---


@pytest.mark.unit
def test_should_continue_hallucination_guard():
    """When agent.tools is empty, tool calls from the LLM must be ignored (→ __end__).

    Some models (e.g. Gemini) hallucinate tool calls even when no tools were bound.
    Routing to tool_node would cause a KeyError and an infinite loop.
    """
    agent = make_cli_agent()
    agent.tools = []  # empty — no tools registered
    agent.tools_by_name = {}

    state = {
        "messages": [
            HumanMessage(content="optimize"),
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "ghost_tool", "args": {}}],
            ),
        ]
    }
    assert agent._should_continue(state) == "__end__"


# --- _clarification_response ---


@pytest.mark.unit
def test_clarification_response_extracts_question():
    """Valid clarification ToolMessage → AIMessage containing the question."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="optimize a beam"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc1", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true, "question": "What volume fraction?"}',
                tool_call_id="tc1",
                name="ask_human_for_clarification",
            ),
        ]
    }
    result = agent._clarification_response(state)
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == "What volume fraction?"


@pytest.mark.unit
def test_clarification_response_missing_question_key_returns_empty():
    """JSON payload without a 'question' key must yield no messages."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="optimize"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc1", "name": "ask_human_for_clarification", "args": {}}
                ],
            ),
            ToolMessage(
                content='{"success": true}',  # no "question" key
                tool_call_id="tc1",
                name="ask_human_for_clarification",
            ),
        ]
    }
    result = agent._clarification_response(state)
    assert result == {"messages": []}


@pytest.mark.unit
def test_clarification_response_no_matching_tool_message_returns_empty():
    """When no ask_human_for_clarification ToolMessage exists, return empty."""
    agent = make_cli_agent()
    state = {
        "messages": [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]
    }
    result = agent._clarification_response(state)
    assert result == {"messages": []}
