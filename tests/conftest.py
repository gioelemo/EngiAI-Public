"""
Pytest configuration and shared fixtures for tests.

Provides test environment setup and pytest markers.
"""

import os

import pytest

# Disable LangSmith tracing during tests to avoid rate limits
os.environ["LANGCHAIN_TRACING_V2"] = "false"


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
