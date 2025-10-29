"""
Shared fixtures and configuration for pytest tests.

Uses LangChain's built-in testing utilities like GenericFakeChatModel
for proper LLM mocking without API calls.
"""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


@pytest.fixture
def fake_llm():
    """
    Fixture providing LangChain's GenericFakeChatModel for testing.

    This is the recommended way to mock LLMs in LangChain tests.
    Returns an in-memory stub that doesn't require API calls.

    Usage:
        def test_something(fake_llm):
            fake_llm.messages = ["engineering_agent"]
            result = agent.invoke(state)
    """
    return GenericFakeChatModel(messages=iter(["test response"]))


@pytest.fixture
def fake_llm_routing_engineering():
    """Fake LLM that routes to engineering_agent."""
    return GenericFakeChatModel(messages=iter(["engineering_agent"]))


@pytest.fixture
def fake_llm_routing_search():
    """Fake LLM that routes to search_agent."""
    return GenericFakeChatModel(messages=iter(["search_agent"]))


@pytest.fixture
def fake_llm_routing_hpc():
    """Fake LLM that routes to hpc_agent."""
    return GenericFakeChatModel(messages=iter(["hpc_agent"]))


@pytest.fixture
def fake_llm_routing_cli():
    """Fake LLM that routes to cli_agent."""
    return GenericFakeChatModel(messages=iter(["cli_agent"]))


@pytest.fixture
def fake_llm_routing_finish():
    """Fake LLM that routes to FINISH (supervisor responds directly)."""
    return GenericFakeChatModel(messages=iter(["FINISH"]))


# Pytest configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers",
        "requires_api: marks tests that require real API keys (skip in CI)",
    )
