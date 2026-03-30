"""
Tier-2 UI tests: check_streamlit_interrupt from confirmation_handler.

The function reads st.session_state.agent, st.session_state.config,
and st.session_state.agent_state. We patch `src.ui.confirmation_handler.st`
with a lightweight fake and control the snapshot returned by graph.get_state().
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.ui.confirmation_handler import check_streamlit_interrupt

# ============================================================================
# Helpers
# ============================================================================


class FakeSessionState:
    """Minimal st.session_state substitute (attribute + item + .get())."""

    def __init__(self, **kwargs):
        object.__setattr__(self, "_store", dict(kwargs))

    def __getattr__(self, name):
        store = object.__getattribute__(self, "_store")
        try:
            return store[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_store")[name] = value

    def __contains__(self, name):
        return name in object.__getattribute__(self, "_store")

    def get(self, name, default=None):
        return object.__getattribute__(self, "_store").get(name, default)


def _make_snapshot(next_nodes=(), values=None):
    """Return a mock graph snapshot with configurable next and values."""
    snap = MagicMock()
    snap.next = next_nodes
    snap.values = values or {}
    return snap


def _make_st(snapshot, agent_state=None):
    """Return a mock st with agent whose graph returns the given snapshot."""
    st = MagicMock()
    agent = MagicMock()
    agent.graph.get_state.return_value = snapshot
    st.session_state = FakeSessionState(
        agent=agent,
        config={"configurable": {"thread_id": "test"}},
        agent_state=agent_state or {"messages": []},
    )
    return st


# ============================================================================
# check_streamlit_interrupt
# ============================================================================


@pytest.mark.unit
def test_returns_false_when_snapshot_next_is_empty():
    """Graph not interrupted (next is empty) → (False, '')."""
    snapshot = _make_snapshot(next_nodes=())
    st = _make_st(snapshot)

    with patch("src.ui.confirmation_handler.st", st):
        interrupted, request = check_streamlit_interrupt()

    assert interrupted is False
    assert request == ""


@pytest.mark.unit
def test_returns_false_when_next_is_not_cli_agent():
    """Graph interrupted at a non-CLI node → (False, '')."""
    snapshot = _make_snapshot(next_nodes=("hpc_agent",))
    st = _make_st(snapshot)

    with patch("src.ui.confirmation_handler.st", st):
        interrupted, request = check_streamlit_interrupt()

    assert interrupted is False
    assert request == ""


@pytest.mark.unit
def test_returns_auto_resume_when_cli_but_no_command_info():
    """CLI agent next but extract_command_info returns '' → (True, '__AUTO_RESUME__')."""
    snapshot = _make_snapshot(next_nodes=("cli_agent",))
    st = _make_st(snapshot)

    with (
        patch("src.ui.confirmation_handler.st", st),
        patch("src.ui.confirmation_handler.extract_command_info", return_value=""),
    ):
        interrupted, request = check_streamlit_interrupt()

    assert interrupted is True
    assert request == "__AUTO_RESUME__"


@pytest.mark.unit
def test_returns_true_with_user_request_when_command_detected():
    """CLI agent + command detected → (True, last human message content)."""
    snapshot = _make_snapshot(next_nodes=("cli_agent",))
    agent_state = {"messages": [HumanMessage(content="run ls -la")]}
    st = _make_st(snapshot, agent_state=agent_state)

    with (
        patch("src.ui.confirmation_handler.st", st),
        patch(
            "src.ui.confirmation_handler.extract_command_info",
            return_value="**Command:** ls -la",
        ),
    ):
        interrupted, request = check_streamlit_interrupt()

    assert interrupted is True
    assert request == "run ls -la"


@pytest.mark.unit
def test_returns_empty_user_request_when_no_human_messages():
    """CLI agent + command detected but no HumanMessage in state → (True, '')."""
    snapshot = _make_snapshot(next_nodes=("cli_agent",))
    agent_state = {"messages": [AIMessage(content="I will run ls")]}
    st = _make_st(snapshot, agent_state=agent_state)

    with (
        patch("src.ui.confirmation_handler.st", st),
        patch(
            "src.ui.confirmation_handler.extract_command_info",
            return_value="**Command:** ls",
        ),
    ):
        interrupted, request = check_streamlit_interrupt()

    assert interrupted is True
    assert request == ""


@pytest.mark.unit
def test_returns_false_on_exception():
    """Any exception (e.g. missing agent attr) → swallowed, returns (False, '')."""
    st = MagicMock()
    # Make get_state raise to trigger the exception path
    st.session_state.agent.graph.get_state.side_effect = RuntimeError("not ready")

    with patch("src.ui.confirmation_handler.st", st):
        interrupted, request = check_streamlit_interrupt()

    assert interrupted is False
    assert request == ""
