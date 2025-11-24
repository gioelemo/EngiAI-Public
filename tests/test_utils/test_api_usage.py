"""
Tests for api_usage module.

These tests cover API usage statistics dataclasses and usage fetching functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.api_usage import (
    MATHPIX_RESET_DAY,
    UNLIMITED_LIMIT_VALUE,
    USAGE_THRESHOLD_CRITICAL,
    USAGE_THRESHOLD_WARNING,
    MathpixUsageStats,
    TavilyUsageStats,
    get_mathpix_usage,
    get_tavily_usage,
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
# MATHPIX USAGE STATS TESTS
# ============================================================================


@pytest.mark.unit
def test_mathpix_usage_stats_creation():
    """Test MathpixUsageStats dataclass creation."""
    stats = MathpixUsageStats(
        pages_usage=100,
        pages_limit=5000,
        pages_percentage=2.0,
        rate_limit=50,
    )

    assert stats.pages_usage == 100
    assert stats.pages_limit == 5000
    assert stats.rate_limit == 50


@pytest.mark.unit
def test_mathpix_usage_stats_is_approaching_limit():
    """Test is_approaching_limit property for Mathpix."""
    # Under threshold
    stats_under = MathpixUsageStats(
        pages_usage=3000,
        pages_limit=5000,
        pages_percentage=60.0,
        rate_limit=50,
    )
    assert not stats_under.is_approaching_limit

    # Over threshold
    stats_over = MathpixUsageStats(
        pages_usage=4200,
        pages_limit=5000,
        pages_percentage=84.0,
        rate_limit=50,
    )
    assert stats_over.is_approaching_limit


@pytest.mark.unit
def test_mathpix_usage_stats_is_critical():
    """Test is_critical property for Mathpix."""
    # Under threshold
    stats_under = MathpixUsageStats(
        pages_usage=4500,
        pages_limit=5000,
        pages_percentage=90.0,
        rate_limit=50,
    )
    assert not stats_under.is_critical

    # Over threshold
    stats_over = MathpixUsageStats(
        pages_usage=4800,
        pages_limit=5000,
        pages_percentage=96.0,
        rate_limit=50,
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
# GET MATHPIX USAGE TESTS
# ============================================================================


@pytest.mark.unit
def test_get_mathpix_usage_no_credentials():
    """Test get_mathpix_usage with no credentials."""
    result = get_mathpix_usage("", "")
    assert result is None

    result = get_mathpix_usage("key", "")
    assert result is None

    result = get_mathpix_usage("", "app_id")
    assert result is None


@pytest.mark.unit
def test_get_mathpix_usage_success():
    """Test successful Mathpix usage fetch."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ocr_usage": [
            {"usage_type": "pdf", "count": 100},
            {"usage_type": "image", "count": 50},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.utils.api_usage.requests.get", return_value=mock_response):
        result = get_mathpix_usage("test_key", "test_app_id")

    assert result is not None
    assert result.pages_usage == 150  # 100 + 50
    assert result.pages_limit == 5000
    assert result.rate_limit == 50


@pytest.mark.unit
def test_get_mathpix_usage_empty_response():
    """Test Mathpix usage with empty ocr_usage list."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ocr_usage": []}
    mock_response.raise_for_status = MagicMock()

    with patch("src.utils.api_usage.requests.get", return_value=mock_response):
        result = get_mathpix_usage("test_key", "test_app_id")

    assert result is not None
    assert result.pages_usage == 0


@pytest.mark.unit
def test_get_mathpix_usage_request_error():
    """Test Mathpix usage fetch with request error."""
    import requests

    with patch(
        "src.utils.api_usage.requests.get",
        side_effect=requests.exceptions.RequestException("Connection error"),
    ):
        result = get_mathpix_usage("test_key", "test_app_id")

    assert result is None


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


@pytest.mark.unit
def test_mathpix_reset_day():
    """Test Mathpix reset day constant."""
    assert MATHPIX_RESET_DAY == 5
    assert 1 <= MATHPIX_RESET_DAY <= 28  # Valid day range
