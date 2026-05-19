"""
API usage monitoring utilities.

This module provides functions to check API usage for various external services.
"""

import logging
import os
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# Constants for usage thresholds
USAGE_THRESHOLD_WARNING = 80  # Percentage threshold for warning (approaching limit)
USAGE_THRESHOLD_CRITICAL = 95  # Percentage threshold for critical (near limit)
UNLIMITED_LIMIT_VALUE = 2147483647  # Max int32 value representing unlimited access

# Substrings that mark a value as a placeholder rather than a real API key.
# Matched case-insensitively against the trimmed key. Real keys for OpenAI
# (``sk-...``), Google (``AIza...``), Anthropic (``sk-ant-...``) and Tavily
# (``tvly-<hex>``) do not contain any of these strings.
PLACEHOLDER_SUBSTRINGS: tuple[str, ...] = (
    "replace-me",
    "replace_me",
    "your-actual",
    "your-api-key",
    "your-openai-api-key",
    "your-google-api-key",
    "your-tavily-api-key",
    "your-anthropic-api-key",
    "changeme",
)


def is_api_key_configured(api_key: str | None) -> bool:
    """Return True if ``api_key`` looks like a real value (not empty/placeholder)."""
    if not api_key:
        return False
    normalized = api_key.strip().lower()
    if not normalized:
        return False
    return not any(p in normalized for p in PLACEHOLDER_SUBSTRINGS)


@dataclass
class TavilyUsageStats:
    """Tavily API usage statistics."""

    # Key-level usage
    key_usage: int
    key_limit: int
    key_percentage: float

    # Account-level usage
    current_plan: str
    plan_usage: int
    plan_limit: int
    plan_percentage: float
    paygo_usage: int
    paygo_limit: int
    paygo_percentage: float

    @property
    def is_approaching_limit(self) -> bool:
        """Check if usage is approaching limit (>80%)."""
        return (
            self.key_percentage > USAGE_THRESHOLD_WARNING
            or self.plan_percentage > USAGE_THRESHOLD_WARNING
        )

    @property
    def is_critical(self) -> bool:
        """Check if usage is critical (>95%)."""
        return (
            self.key_percentage > USAGE_THRESHOLD_CRITICAL
            or self.plan_percentage > USAGE_THRESHOLD_CRITICAL
        )


def get_tavily_usage(api_key: str) -> TavilyUsageStats | None:
    """
    Get Tavily API usage statistics.

    Args:
        api_key: Tavily API key

    Returns:
        TavilyUsageStats or None: Usage statistics object or None if request fails

    Example::

        Example response from Tavily API:
        {
          "key": {"usage": 150, "limit": 1000},
          "account": {
            "current_plan": "Bootstrap",
            "plan_usage": 500,
            "plan_limit": 15000,
            "paygo_usage": 25,
            "paygo_limit": 100
          }
        }
    """
    if os.getenv("SKIP_SEARCH", "false").lower() == "true":
        return None

    if not is_api_key_configured(api_key):
        logger.warning("Tavily API key not configured")
        return None

    try:
        response = requests.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()

        # Try to parse JSON, handle empty response case
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            # JSON decode failed - likely empty response
            logger.warning(
                "Tavily API returned empty or invalid JSON response "
                "(likely 202 Accepted without body)"
            )
            return None

        # Extract usage data
        key_data = data.get("key", {})
        account_data = data.get("account", {})

        key_usage = key_data.get("usage", 0)
        # Tavily returns null for unlimited access - treat as unlimited
        key_limit_raw = key_data.get("limit")
        key_limit = (
            key_limit_raw if key_limit_raw is not None else UNLIMITED_LIMIT_VALUE
        )
        key_percentage = (key_usage / key_limit * 100) if key_limit > 0 else 0

        plan_usage = account_data.get("plan_usage", 0)
        plan_limit_raw = account_data.get("plan_limit")
        plan_limit = (
            plan_limit_raw if plan_limit_raw is not None else UNLIMITED_LIMIT_VALUE
        )
        plan_percentage = (plan_usage / plan_limit * 100) if plan_limit > 0 else 0

        paygo_usage = account_data.get("paygo_usage", 0)
        paygo_limit_raw = account_data.get("paygo_limit")
        paygo_limit = paygo_limit_raw if paygo_limit_raw is not None else 0
        paygo_percentage = (paygo_usage / paygo_limit * 100) if paygo_limit > 0 else 0

        return TavilyUsageStats(
            key_usage=key_usage,
            key_limit=key_limit,
            key_percentage=key_percentage,
            current_plan=account_data.get("current_plan", "Unknown"),
            plan_usage=plan_usage,
            plan_limit=plan_limit,
            plan_percentage=plan_percentage,
            paygo_usage=paygo_usage,
            paygo_limit=paygo_limit,
            paygo_percentage=paygo_percentage,
        )

    except requests.exceptions.RequestException as e:
        status = (
            e.response.status_code
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None
            else None
        )
        if status in (401, 403):
            logger.warning(
                "Tavily API rejected the configured key (HTTP %s); "
                "skipping usage check.",
                status,
            )
        else:
            logger.exception("Failed to fetch Tavily usage")
        return None
    except (KeyError, ValueError):
        logger.exception("Failed to parse Tavily usage response")
        return None
