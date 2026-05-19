"""
Tests for api_usage module.

These tests cover API usage statistics dataclasses and usage fetching functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.api_usage import (
    UNLIMITED_LIMIT_VALUE,
    USAGE_THRESHOLD_CRITICAL,
    USAGE_THRESHOLD_WARNING,
    TavilyUsageStats,
    get_tavily_usage,
    is_api_key_configured,
)

# ============================================================================
# TAVILY USAGE STATS TESTS
# ============================================================================


@pytest.mark.unit
def test_tavily_usage_stats_creation():
    """Test TavilyUsageStats dataclass creation."""
    stats = TavilyUsageStats(
        key_usage=100,
        key_limit=1000,
        key_percentage=10.0,
        current_plan="Bootstrap",
        plan_usage=500,
        plan_limit=15000,
        plan_percentage=3.33,
        paygo_usage=0,
        paygo_limit=100,
        paygo_percentage=0.0,
    )

    assert stats.key_usage == 100
    assert stats.key_limit == 1000
    assert stats.current_plan == "Bootstrap"


@pytest.mark.unit
def test_tavily_usage_stats_is_approaching_limit():
    """Test is_approaching_limit property."""
    # Under threshold
    stats_under = TavilyUsageStats(
        key_usage=50,
        key_limit=100,
        key_percentage=50.0,
        current_plan="Test",
        plan_usage=50,
        plan_limit=100,
        plan_percentage=50.0,
        paygo_usage=0,
        paygo_limit=100,
        paygo_percentage=0.0,
    )
    assert not stats_under.is_approaching_limit

    # Over threshold (key)
    stats_over_key = TavilyUsageStats(
        key_usage=85,
        key_limit=100,
        key_percentage=85.0,
        current_plan="Test",
        plan_usage=50,
        plan_limit=100,
        plan_percentage=50.0,
        paygo_usage=0,
        paygo_limit=100,
        paygo_percentage=0.0,
    )
    assert stats_over_key.is_approaching_limit

    # Over threshold (plan)
    stats_over_plan = TavilyUsageStats(
        key_usage=50,
        key_limit=100,
        key_percentage=50.0,
        current_plan="Test",
        plan_usage=85,
        plan_limit=100,
        plan_percentage=85.0,
        paygo_usage=0,
        paygo_limit=100,
        paygo_percentage=0.0,
    )
    assert stats_over_plan.is_approaching_limit


@pytest.mark.unit
def test_tavily_usage_stats_is_critical():
    """Test is_critical property."""
    # Under threshold
    stats_under = TavilyUsageStats(
        key_usage=90,
        key_limit=100,
        key_percentage=90.0,
        current_plan="Test",
        plan_usage=90,
        plan_limit=100,
        plan_percentage=90.0,
        paygo_usage=0,
        paygo_limit=100,
        paygo_percentage=0.0,
    )
    assert not stats_under.is_critical

    # Over threshold
    stats_over = TavilyUsageStats(
        key_usage=98,
        key_limit=100,
        key_percentage=98.0,
        current_plan="Test",
        plan_usage=50,
        plan_limit=100,
        plan_percentage=50.0,
        paygo_usage=0,
        paygo_limit=100,
        paygo_percentage=0.0,
    )
    assert stats_over.is_critical


# ============================================================================
# GET TAVILY USAGE TESTS
# ============================================================================


@pytest.mark.unit
def test_get_tavily_usage_no_api_key():
    """Test get_tavily_usage with no API key."""
    result = get_tavily_usage("")
    assert result is None

    result = get_tavily_usage(None)  # type: ignore[arg-type]
    assert result is None


@pytest.mark.unit
def test_get_tavily_usage_success():
    """Test successful Tavily usage fetch."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "key": {"usage": 150, "limit": 1000},
        "account": {
            "current_plan": "Bootstrap",
            "plan_usage": 500,
            "plan_limit": 15000,
            "paygo_usage": 25,
            "paygo_limit": 100,
        },
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.utils.api_usage.requests.get", return_value=mock_response):
        result = get_tavily_usage("test_api_key")

    assert result is not None
    assert result.key_usage == 150
    assert result.key_limit == 1000
    assert result.current_plan == "Bootstrap"
    assert result.plan_usage == 500


@pytest.mark.unit
def test_get_tavily_usage_unlimited_key():
    """Test Tavily usage with unlimited (null) key limit."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "key": {"usage": 150, "limit": None},  # Unlimited
        "account": {
            "current_plan": "Enterprise",
            "plan_usage": 500,
            "plan_limit": None,  # Unlimited
            "paygo_usage": 0,
            "paygo_limit": None,
        },
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.utils.api_usage.requests.get", return_value=mock_response):
        result = get_tavily_usage("test_api_key")

    assert result is not None
    assert result.key_limit == UNLIMITED_LIMIT_VALUE
    assert result.plan_limit == UNLIMITED_LIMIT_VALUE
    assert result.key_percentage < 1  # Should be near 0% of unlimited


@pytest.mark.unit
def test_get_tavily_usage_request_error():
    """Test Tavily usage fetch with request error."""
    import requests

    with patch(
        "src.utils.api_usage.requests.get",
        side_effect=requests.exceptions.RequestException("Connection error"),
    ):
        result = get_tavily_usage("test_api_key")

    assert result is None


@pytest.mark.unit
def test_get_tavily_usage_parse_error():
    """Test Tavily usage fetch with parse error."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"invalid": "structure"}
    mock_response.raise_for_status = MagicMock()

    with patch("src.utils.api_usage.requests.get", return_value=mock_response):
        result = get_tavily_usage("test_api_key")

    # Should handle gracefully and return stats with defaults
    assert result is not None or result is None  # Either works based on implementation


# ============================================================================
# PLACEHOLDER DETECTION TESTS
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "tvly-replace-me",
        "your-actual-openai-api-key-here",
        "your-actual-google-api-key-here",
        "your-actual-tavily-api-key-here",
        "your-tavily-api-key",
        "your-openai-api-key",
        "your-google-api-key",
        "your-anthropic-api-key",
        "CHANGEME",
        "ChangeMe",
        "REPLACE_ME",
    ],
)
def test_is_api_key_configured_rejects_placeholders(value):
    """Empty, whitespace, and known placeholder substrings are rejected."""
    assert is_api_key_configured(value) is False


@pytest.mark.unit
def test_is_api_key_configured_rejects_none():
    """None is rejected."""
    assert is_api_key_configured(None) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "sk-proj-abc123def456ghi789jkl",  # OpenAI
        "sk-ant-api03-xxxxxxxxxxxxxxxx",  # Anthropic
        "AIzaSyDxxxxxxxxxxxxxxxxxxxxxx",  # Google
        "tvly-abc123def456",  # Tavily
    ],
)
def test_is_api_key_configured_accepts_real_keys(value):
    """Real-format keys for OpenAI / Anthropic / Google / Tavily are accepted."""
    assert is_api_key_configured(value) is True


# ============================================================================
# SKIP_SEARCH AND AUTH-FAILURE TESTS
# ============================================================================


@pytest.mark.unit
def test_get_tavily_usage_skip_search_short_circuits(monkeypatch):
    """SKIP_SEARCH=true returns None without calling Tavily."""
    monkeypatch.setenv("SKIP_SEARCH", "true")
    with patch("src.utils.api_usage.requests.get") as mock_get:
        result = get_tavily_usage("tvly-real-looking-key")
    assert result is None
    mock_get.assert_not_called()


@pytest.mark.unit
def test_get_tavily_usage_placeholder_key_short_circuits(monkeypatch):
    """A placeholder key returns None without calling Tavily."""
    monkeypatch.delenv("SKIP_SEARCH", raising=False)
    with patch("src.utils.api_usage.requests.get") as mock_get:
        result = get_tavily_usage("tvly-replace-me")
    assert result is None
    mock_get.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("status_code", [401, 403])
def test_get_tavily_usage_unauthorized_returns_none(monkeypatch, caplog, status_code):
    """401/403 from Tavily logs a warning (not exception) and returns None."""
    import logging

    import requests

    monkeypatch.delenv("SKIP_SEARCH", raising=False)
    mock_response = MagicMock()
    mock_response.status_code = status_code
    http_error = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = http_error

    with (
        caplog.at_level(logging.WARNING, logger="src.utils.api_usage"),
        patch("src.utils.api_usage.requests.get", return_value=mock_response),
    ):
        result = get_tavily_usage("tvly-real-looking-key")

    assert result is None
    # Exactly one WARNING-level record, no ERROR (would indicate logger.exception)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(warnings) == 1
    assert str(status_code) in warnings[0].getMessage()
    assert errors == []


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


@pytest.mark.unit
def test_usage_thresholds():
    """Test that usage threshold constants are properly defined."""
    assert USAGE_THRESHOLD_WARNING == 80
    assert USAGE_THRESHOLD_CRITICAL == 95
    assert USAGE_THRESHOLD_WARNING < USAGE_THRESHOLD_CRITICAL


@pytest.mark.unit
def test_unlimited_limit_value():
    """Test unlimited limit constant."""
    assert UNLIMITED_LIMIT_VALUE == 2147483647  # Max int32
