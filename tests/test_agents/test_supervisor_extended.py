"""
Extended tests for supervisor_agent module.

These tests cover the SupervisorAgent routing logic and node functions.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor_agent import SupervisorAgent, SupervisorState

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def _mock_agents():
    """Create mock agents to avoid real initialization."""
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
        mock_llm.return_value = MagicMock()
        mock_cp.return_value = MagicMock()

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
    """Test SupervisorAgent initialization."""
    agent = SupervisorAgent()

    assert agent.graph is not None
    assert hasattr(agent, "engineering_agent")
    assert hasattr(agent, "hpc_agent")
    assert hasattr(agent, "search_agent")


@pytest.mark.unit
def test_supervisor_agent_custom_model(_mock_agents):
    """Test SupervisorAgent with custom model."""
    agent = SupervisorAgent(model_name="gpt-4", temperature=0.5)

    assert agent.model_name == "gpt-4"
    assert agent.temperature == 0.5


# ============================================================================
# ROUTING PROMPT TESTS
# ============================================================================


@pytest.mark.unit
def test_build_routing_prompt(_mock_agents):
    """Test routing prompt generation."""
    agent = SupervisorAgent()
    prompt = agent._build_routing_prompt()

    assert "engineering_agent" in prompt
    assert "hpc_agent" in prompt
    assert "search_agent" in prompt
    assert "rag_agent" in prompt
    assert "arxiv_agent" in prompt
    assert "prusa_agent" in prompt
    assert "cli_agent" in prompt
    assert "FINISH" in prompt


@pytest.mark.unit
def test_routing_prompt_contains_keywords(_mock_agents):
    """Test that routing prompt contains important keywords."""
    agent = SupervisorAgent()
    prompt = agent._build_routing_prompt()

    # Engineering keywords
    assert "optimization" in prompt.lower()
    assert "wandb" in prompt.lower()

    # HPC keywords
    assert "slurm" in prompt.lower()
    assert "cluster" in prompt.lower()

    # Prusa keywords
    assert "printer" in prompt.lower()


# ============================================================================
# SUPERVISOR NODE TESTS
# ============================================================================


@pytest.mark.unit
def test_supervisor_node_already_routed(_mock_agents):
    """Test supervisor returns FINISH if already routed."""
    agent = SupervisorAgent()

    state = SupervisorState(
        messages=[HumanMessage(content="test")], next="engineering_agent"
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "FINISH"


@pytest.mark.unit
def test_supervisor_node_direct_cli_routing(_mock_agents):
    """Test direct routing to CLI for 'open' commands."""
    agent = SupervisorAgent()

    state = SupervisorState(
        messages=[HumanMessage(content="open PrusaSlicer")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "cli_agent"


@pytest.mark.unit
def test_supervisor_node_routes_engineering(_mock_agents):
    """Test routing to engineering agent."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="engineering_agent")

    state = SupervisorState(
        messages=[HumanMessage(content="optimize this beam")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "engineering_agent"


@pytest.mark.unit
def test_supervisor_node_routes_hpc(_mock_agents):
    """Test routing to HPC agent."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="hpc_agent submit")

    state = SupervisorState(messages=[HumanMessage(content="submit job")], next="")
    result = agent._supervisor_node(state)

    assert result["next"] == "hpc_agent"


@pytest.mark.unit
def test_supervisor_node_routes_search(_mock_agents):
    """Test routing to search agent."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="search_agent")

    state = SupervisorState(
        messages=[HumanMessage(content="search for best practices")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "search_agent"


@pytest.mark.unit
def test_supervisor_node_routes_rag(_mock_agents):
    """Test routing to RAG agent."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="rag_agent document")

    state = SupervisorState(
        messages=[HumanMessage(content="what does the document say")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "rag_agent"


@pytest.mark.unit
def test_supervisor_node_routes_arxiv(_mock_agents):
    """Test routing to ArXiv agent."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="arxiv_agent paper")

    state = SupervisorState(
        messages=[HumanMessage(content="find papers about transformers")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "arxiv_agent"


@pytest.mark.unit
def test_supervisor_node_routes_prusa(_mock_agents):
    """Test routing to Prusa agent."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="prusa_agent printer")

    state = SupervisorState(
        messages=[HumanMessage(content="check printer status")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "prusa_agent"


@pytest.mark.unit
def test_supervisor_node_routes_cli(_mock_agents):
    """Test routing to CLI agent."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="cli_agent command run")

    state = SupervisorState(messages=[HumanMessage(content="run ls command")], next="")
    result = agent._supervisor_node(state)

    assert result["next"] == "cli_agent"


@pytest.mark.unit
def test_supervisor_node_routes_finish(_mock_agents):
    """Test routing to FINISH for capability questions."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="FINISH")

    state = SupervisorState(
        messages=[HumanMessage(content="what can you do?")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "supervisor_response"


# ============================================================================
# SUPERVISOR RESPONSE NODE TESTS
# ============================================================================


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
    assert result["next"] == "FINISH"


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
    assert result["next"] == "FINISH"


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
    assert result["next"] == "FINISH"


@pytest.mark.unit
def test_rag_node(_mock_agents):
    """Test RAG node delegation."""
    agent = SupervisorAgent()
    agent.rag_agent = MagicMock()
    agent.rag_agent.invoke.return_value = {
        "messages": [HumanMessage(content="test"), AIMessage(content="result")]
    }

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._rag_node(state)

    assert "messages" in result
    assert result["next"] == "FINISH"


@pytest.mark.unit
def test_arxiv_node(_mock_agents):
    """Test ArXiv node delegation."""
    agent = SupervisorAgent()
    agent.arxiv_agent = MagicMock()
    agent.arxiv_agent.invoke.return_value = {
        "messages": [HumanMessage(content="test"), AIMessage(content="result")]
    }

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._arxiv_node(state)

    assert "messages" in result
    assert result["next"] == "FINISH"


@pytest.mark.unit
def test_prusa_node(_mock_agents):
    """Test Prusa node delegation."""
    agent = SupervisorAgent()
    agent.prusa_agent = MagicMock()
    agent.prusa_agent.invoke.return_value = {
        "messages": [HumanMessage(content="test"), AIMessage(content="result")]
    }

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._prusa_node(state)

    assert "messages" in result
    assert result["next"] == "FINISH"


@pytest.mark.unit
def test_cli_node(_mock_agents):
    """Test CLI node delegation."""
    agent = SupervisorAgent()
    agent.cli_agent = MagicMock()
    agent.cli_agent.invoke.return_value = {
        "messages": [HumanMessage(content="test"), AIMessage(content="result")]
    }

    state = SupervisorState(messages=[HumanMessage(content="test")], next="")
    result = agent._cli_node(state)

    assert "messages" in result
    assert result["next"] == "FINISH"


# ============================================================================
# INVOKE TESTS
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


# ============================================================================
# BUILD GRAPH TESTS
# ============================================================================


@pytest.mark.unit
def test_build_graph_creates_graph(_mock_agents):
    """Test that _build_graph creates a compiled graph."""
    agent = SupervisorAgent()

    assert agent.graph is not None
    assert hasattr(agent.graph, "invoke")


# ============================================================================
# ROUTING KEYWORD DETECTION TESTS
# ============================================================================


@pytest.mark.unit
def test_routing_detects_wandb_keyword(_mock_agents):
    """Test that wandb keyword routes to engineering."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="wandb model download")

    state = SupervisorState(
        messages=[HumanMessage(content="download model from wandb")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "engineering_agent"


@pytest.mark.unit
def test_routing_detects_slurm_keyword(_mock_agents):
    """Test that slurm keyword routes to engineering (for generation)."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="generate slurm script")

    state = SupervisorState(
        messages=[HumanMessage(content="generate a slurm script")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "engineering_agent"


@pytest.mark.unit
def test_routing_detects_training_keyword(_mock_agents):
    """Test that training keyword routes to engineering."""
    agent = SupervisorAgent()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = MagicMock(content="training script")

    state = SupervisorState(
        messages=[HumanMessage(content="create training script")], next=""
    )
    result = agent._supervisor_node(state)

    assert result["next"] == "engineering_agent"
