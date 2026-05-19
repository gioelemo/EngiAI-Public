"""
Tests for _serialize_messages and _deserialize_messages in src/ui/database.py.

Both are pure Python functions with no database or Streamlit dependency.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.ui.database import _deserialize_messages, _serialize_messages

# --- _serialize_messages ---


@pytest.mark.unit
def test_serialize_empty_list():
    """Empty input → empty output."""
    assert _serialize_messages([]) == []


@pytest.mark.unit
def test_serialize_human_message():
    """HumanMessage → type 'HumanMessage' and content preserved."""
    msgs = [HumanMessage(content="hello")]
    result = _serialize_messages(msgs)
    assert len(result) == 1
    assert result[0]["type"] == "HumanMessage"
    assert result[0]["content"] == "hello"


@pytest.mark.unit
def test_serialize_ai_message():
    """AIMessage → type 'AIMessage' and content preserved."""
    msgs = [AIMessage(content="I can help with that.")]
    result = _serialize_messages(msgs)
    assert result[0]["type"] == "AIMessage"
    assert result[0]["content"] == "I can help with that."


@pytest.mark.unit
def test_serialize_tool_message_includes_tool_call_id_and_name():
    """ToolMessage → tool_call_id and name serialized when present."""
    msgs = [
        ToolMessage(
            content="result text",
            tool_call_id="tc-42",
            name="my_tool",
        )
    ]
    result = _serialize_messages(msgs)
    assert result[0]["tool_call_id"] == "tc-42"
    assert result[0]["name"] == "my_tool"


@pytest.mark.unit
def test_serialize_multiple_messages_preserves_order():
    """Order of messages is preserved after serialization."""
    msgs = [
        HumanMessage(content="first"),
        AIMessage(content="second"),
        HumanMessage(content="third"),
    ]
    result = _serialize_messages(msgs)
    assert [r["content"] for r in result] == ["first", "second", "third"]


# --- _deserialize_messages ---


@pytest.mark.unit
def test_deserialize_empty_list():
    """Empty input → empty output."""
    assert _deserialize_messages([]) == []


@pytest.mark.unit
def test_deserialize_human_message():
    """Dict with type 'HumanMessage' → HumanMessage instance."""
    data = [{"type": "HumanMessage", "content": "hello"}]
    result = _deserialize_messages(data)
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "hello"


@pytest.mark.unit
def test_deserialize_ai_message():
    """Dict with type 'AIMessage' → AIMessage instance."""
    data = [{"type": "AIMessage", "content": "sure"}]
    result = _deserialize_messages(data)
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "sure"


@pytest.mark.unit
def test_deserialize_tool_message():
    """Dict with type 'ToolMessage' → ToolMessage with tool_call_id."""
    data = [{"type": "ToolMessage", "content": "done", "tool_call_id": "tc-99"}]
    result = _deserialize_messages(data)
    assert isinstance(result[0], ToolMessage)
    assert result[0].tool_call_id == "tc-99"


@pytest.mark.unit
def test_deserialize_unknown_type_falls_back_to_human():
    """Unrecognised type → falls back to HumanMessage."""
    data = [{"type": "WeirdMessage", "content": "mystery"}]
    result = _deserialize_messages(data)
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "mystery"


@pytest.mark.unit
def test_serialize_deserialize_round_trip():
    """Round-trip serialization preserves type and content for all message kinds."""
    original = [
        HumanMessage(content="ping"),
        AIMessage(content="pong"),
        ToolMessage(content="tool result", tool_call_id="tc-1"),
    ]
    recovered = _deserialize_messages(_serialize_messages(original))

    assert len(recovered) == 3
    assert isinstance(recovered[0], HumanMessage)
    assert isinstance(recovered[1], AIMessage)
    assert isinstance(recovered[2], ToolMessage)
    assert recovered[0].content == "ping"
    assert recovered[1].content == "pong"
    assert recovered[2].content == "tool result"
