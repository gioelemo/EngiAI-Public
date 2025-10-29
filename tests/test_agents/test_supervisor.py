"""
Tests for the Supervisor Agent.

Uses LangChain's GenericFakeChatModel for proper testing without API calls.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.supervisor_agent import SupervisorAgent


@pytest.mark.unit
def test_supervisor_routes_to_engineering(fake_llm_routing_engineering):
    """Test supervisor routes optimization queries to engineering agent."""
    # Patch where init_chat_model is imported in the supervisor module
    with patch(
        "src.agents.supervisor_agent.init_chat_model",
        return_value=fake_llm_routing_engineering,
    ):
        agent = SupervisorAgent()

        state = {
            "messages": [
                HumanMessage(content="Optimize my beam design with 35% volume")
            ],
            "next": "",
        }

        result = agent._supervisor_node(state)

        assert result["next"] == "engineering_agent"


@pytest.mark.unit
def test_supervisor_routes_to_search(fake_llm_routing_search):
    """Test supervisor routes research queries to search agent."""
    with patch(
        "src.agents.supervisor_agent.init_chat_model",
        return_value=fake_llm_routing_search,
    ):
        agent = SupervisorAgent()

        state = {
            "messages": [
                HumanMessage(content="Search for topology optimization papers")
            ],
            "next": "",
        }

        result = agent._supervisor_node(state)

        assert result["next"] == "search_agent"


@pytest.mark.unit
def test_supervisor_routes_to_finish(fake_llm_routing_finish):
    """Test supervisor answers capability questions directly."""
    with patch(
        "src.agents.supervisor_agent.init_chat_model",
        return_value=fake_llm_routing_finish,
    ):
        agent = SupervisorAgent()

        state = {
            "messages": [HumanMessage(content="What can you do?")],
            "next": "",
        }

        result = agent._supervisor_node(state)

        assert result["next"] == "supervisor_response"


@pytest.mark.unit
def test_supervisor_initialization(fake_llm):
    """Test supervisor agent initializes correctly."""
    with patch("src.agents.supervisor_agent.init_chat_model", return_value=fake_llm):
        agent = SupervisorAgent()

        assert agent.llm is not None
        assert agent.engineering_agent is not None
        assert agent.hpc_agent is not None
        assert agent.search_agent is not None
        assert agent.cli_agent is not None
        assert agent.graph is not None
