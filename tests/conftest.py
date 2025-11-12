"""
Improved shared fixtures and configuration for pytest tests.

Provides realistic mocking utilities and reusable test components.
"""

import os
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

# Disable LangSmith tracing during tests to avoid rate limits
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ============================================================================
# BASIC LLM FIXTURES
# ============================================================================


@pytest.fixture
def fake_llm():
    """
    Basic fake LLM for general testing.

    Returns a GenericFakeChatModel that yields "test response" by default.
    """
    return GenericFakeChatModel(messages=iter([AIMessage(content="test response")]))


@pytest.fixture
def fake_llm_with_custom_responses():
    """
    Factory fixture for creating fake LLMs with custom response sequences.

    Usage:
        def test_something(fake_llm_with_custom_responses):
            llm = fake_llm_with_custom_responses([
                "First response",
                "Second response"
            ])
            # Use llm in your test
    """

    def _create_llm(responses):
        """Create a fake LLM with specified responses."""
        messages = [
            AIMessage(content=resp) if isinstance(resp, str) else resp
            for resp in responses
        ]
        return GenericFakeChatModel(messages=iter(messages))

    return _create_llm


# ============================================================================
# ROUTING FIXTURES - For supervisor routing tests
# ============================================================================


@pytest.fixture
def fake_llm_routing_engineering():
    """Fake LLM that routes to engineering_agent with realistic output."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="engineering_agent")]))


@pytest.fixture
def fake_llm_routing_search():
    """Fake LLM that routes to search_agent with realistic output."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="search_agent")]))


@pytest.fixture
def fake_llm_routing_hpc():
    """Fake LLM that routes to hpc_agent with realistic output."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="hpc_agent")]))


@pytest.fixture
def fake_llm_routing_cli():
    """Fake LLM that routes to cli_agent with realistic output."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="cli_agent")]))


@pytest.fixture
def fake_llm_routing_finish():
    """Fake LLM that routes to FINISH (supervisor responds directly)."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="FINISH")]))


@pytest.fixture
def fake_llm_json_routing():
    """
    Fake LLM that returns JSON-formatted routing decisions.
    More realistic for systems that use structured output.
    """
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content='{"next": "engineering_agent", "reasoning": "User requested optimization"}'
                )
            ]
        )
    )


# ============================================================================
# STATE FIXTURES - For testing conversation state
# ============================================================================


@pytest.fixture
def empty_state():
    """Empty conversation state for testing initialization."""
    return {
        "messages": [],
        "next": "",
    }


@pytest.fixture
def simple_state():
    """Simple single-message state for basic routing tests."""
    return {
        "messages": [HumanMessage(content="Test message")],
        "next": "",
    }


@pytest.fixture
def multi_turn_state():
    """Multi-turn conversation state for testing context handling."""
    return {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi! How can I help you today?"),
            HumanMessage(content="I need to optimize a beam"),
            AIMessage(content="I can help with that. What are your constraints?"),
            HumanMessage(content="Volume should be 35% of original"),
        ],
        "next": "",
    }


@pytest.fixture
def state_with_metadata():
    """Conversation state with metadata for testing advanced features."""
    return {
        "messages": [HumanMessage(content="Optimize my structure")],
        "next": "",
        "metadata": {
            "user_id": "test_user_123",
            "session_id": "session_abc",
            "timestamp": "2025-01-15T10:30:00Z",
        },
    }


# ============================================================================
# MOCK AGENT FIXTURES - For testing without real sub-agents
# ============================================================================


@pytest.fixture
def mock_engineering_agent():
    """Mock engineering agent that returns success."""
    mock = Mock()
    mock.return_value = {
        "messages": [AIMessage(content="Optimization complete")],
        "next": "FINISH",
    }
    return mock


@pytest.fixture
def mock_search_agent():
    """Mock search agent that returns search results."""
    mock = Mock()
    mock.return_value = {
        "messages": [AIMessage(content="Found 5 relevant papers")],
        "next": "FINISH",
    }
    return mock


@pytest.fixture
def mock_hpc_agent():
    """Mock HPC agent that simulates job submission."""
    mock = Mock()
    mock.return_value = {
        "messages": [AIMessage(content="Job submitted with ID: 12345")],
        "next": "FINISH",
    }
    return mock


@pytest.fixture
def mock_cli_agent():
    """Mock CLI agent that simulates command execution."""
    mock = Mock()
    mock.return_value = {
        "messages": [AIMessage(content="Command executed successfully")],
        "next": "FINISH",
    }
    return mock


# ============================================================================
# ERROR SIMULATION FIXTURES - For testing error handling
# ============================================================================


@pytest.fixture
def fake_llm_invalid_routing():
    """Fake LLM that returns invalid agent name for error testing."""
    return GenericFakeChatModel(
        messages=iter([AIMessage(content="nonexistent_agent_xyz")])
    )


@pytest.fixture
def fake_llm_malformed_json():
    """Fake LLM that returns malformed JSON for error testing."""
    return GenericFakeChatModel(
        messages=iter(
            [AIMessage(content='{"next": "engineering_agent", invalid json!!!')]
        )
    )


@pytest.fixture
def fake_llm_empty_response():
    """Fake LLM that returns empty response for error testing."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="")]))


@pytest.fixture
def fake_llm_with_exception():
    """Fake LLM that raises exception for error testing."""
    mock = Mock()
    mock.invoke.side_effect = Exception("Simulated LLM error")
    return mock


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def sample_optimization_queries():
    """Collection of sample optimization queries for testing."""
    return [
        "Optimize my beam design",
        "Minimize the volume while maintaining stiffness",
        "Run topology optimization with 35% volume constraint",
        "Design an optimal cantilever beam",
        "Perform structural optimization on this model",
    ]


@pytest.fixture
def sample_search_queries():
    """Collection of sample search queries for testing."""
    return [
        "Search for topology optimization papers",
        "Find research on SIMP method",
        "What are recent advances in structural optimization?",
        "Look up papers by Ole Sigmund",
        "Find literature on compliance minimization",
    ]


@pytest.fixture
def sample_hpc_queries():
    """Collection of sample HPC queries for testing."""
    return [
        "Submit this job to the cluster",
        "Check the status of my running jobs",
        "Cancel job 12345",
        "What's the queue status?",
        "Run this simulation on 16 cores",
    ]


@pytest.fixture
def sample_cli_queries():
    """Collection of sample CLI queries for testing."""
    return [
        "List files in the current directory",
        "Show me the contents of results.txt",
        "Create a new directory for simulations",
        "Delete the old log files",
        "Check disk usage",
    ]


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Configure custom pytest markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers",
        "requires_api: marks tests that require real API keys (skip in CI)",
    )
    config.addinivalue_line(
        "markers",
        "smoke: marks tests as smoke tests (quick sanity checks)",
    )


def pytest_collection_modifyitems(items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Auto-mark slow tests based on name patterns
        if any(
            keyword in item.nodeid
            for keyword in ["slow", "long_running", "performance"]
        ):
            item.add_marker(pytest.mark.slow)


# ============================================================================
# SESSION FIXTURES - Setup/teardown for entire test session
# ============================================================================


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data that persists across tests."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment once before all tests."""
    # Could set environment variables, configure logging, etc.

    os.environ["TESTING"] = "true"
    # Skip MCP server connection during tests (PrusaAgent will skip initialization)
    os.environ["SKIP_MCP"] = "true"

    yield

    # Cleanup after all tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "SKIP_MCP" in os.environ:
        del os.environ["SKIP_MCP"]


# ============================================================================
# PARAMETRIZE HELPERS
# ============================================================================

# Common parametrize decorators for reuse across tests
ALL_AGENT_TYPES = pytest.mark.parametrize(
    "agent_name",
    [
        "engineering_agent",
        "search_agent",
        "hpc_agent",
        "cli_agent",
        "supervisor_response",
    ],
)

EDGE_CASE_MESSAGES = pytest.mark.parametrize(
    "message_content",
    [
        "",  # Empty message
        " ",  # Whitespace only
        "a" * 10000,  # Very long message
        "Hello\n\n\n\nWorld",  # Multiple newlines
        "Special chars: !@#$%^&*()",  # Special characters
    ],
)
