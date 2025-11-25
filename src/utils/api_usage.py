"""
API usage monitoring utilities.

This module provides functions to check API usage for various external services.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import requests

logger = logging.getLogger(__name__)

# Constants for usage thresholds
USAGE_THRESHOLD_WARNING = 80  # Percentage threshold for warning (approaching limit)
USAGE_THRESHOLD_CRITICAL = 95  # Percentage threshold for critical (near limit)
UNLIMITED_LIMIT_VALUE = 2147483647  # Max int32 value representing unlimited access
MATHPIX_RESET_DAY = 5  # Day of month when Mathpix usage resets


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


@dataclass
class MathpixUsageStats:
    """Mathpix API usage statistics."""

    # Monthly usage (pages processed)
    pages_usage: int
    pages_limit: int
    pages_percentage: float

    # Rate limit tracking (requests per minute)
    rate_limit: int  # 50 req/min

    @property
    def is_approaching_limit(self) -> bool:
        """Check if usage is approaching limit (>80%)."""
        return self.pages_percentage > USAGE_THRESHOLD_WARNING

    @property
    def is_critical(self) -> bool:
        """Check if usage is critical (>95%)."""
        return self.pages_percentage > USAGE_THRESHOLD_CRITICAL


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
    if not api_key:
        logger.warning("Tavily API key not configured")
        return None

    try:
        response = requests.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

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

    except requests.exceptions.RequestException:
        logger.exception("Failed to fetch Tavily usage")
        return None
    except (KeyError, ValueError):
        logger.exception("Failed to parse Tavily usage response")
        return None


def get_mathpix_usage(api_key: str, app_id: str) -> MathpixUsageStats | None:
    """
    Get Mathpix API usage statistics.

    Args:
        api_key: Mathpix API key (app_key)
        app_id: Mathpix app ID

    Returns:
        MathpixUsageStats object or None if request fails

    Note:
        Mathpix limits: 50 requests/minute, 5,000 pages/month
        Usage resets on the 5th of each month
    """
    if not api_key or not app_id:
        logger.warning("Mathpix API credentials not configured")
        return None

    try:
        # Mathpix OCR usage endpoint
        # According to API docs, we need to specify date range and grouping parameters
        # Get current month's usage (from the 5th of current month when usage resets)

        # Calculate the start date (5th of current month or 5th of previous month)
        now = datetime.now(UTC)
        if now.day >= MATHPIX_RESET_DAY:
            # Current billing period started on the 5th of this month
            from_date = datetime(
                now.year, now.month, MATHPIX_RESET_DAY, 0, 0, 0, tzinfo=UTC
            )
        elif now.month == 1:
            # We're before the 5th in January, so current period started last month (December)
            from_date = datetime(
                now.year - 1, 12, MATHPIX_RESET_DAY, 0, 0, 0, tzinfo=UTC
            )
        else:
            # We're before the 5th, so current period started last month
            from_date = datetime(
                now.year, now.month - 1, MATHPIX_RESET_DAY, 0, 0, 0, tzinfo=UTC
            )

        # Format dates as ISO strings
        from_date_str = from_date.isoformat()
        to_date_str = now.isoformat()

        # Build request URL with parameters
        params = {
            "from_date": from_date_str,
            "to_date": to_date_str,
            "group_by": "usage_type",
            "timespan": "month",
        }

        logger.debug(f"Fetching Mathpix usage from {from_date_str} to {to_date_str}")

        response = requests.get(
            "https://api.mathpix.com/v3/ocr-usage",
            headers={"app_id": app_id, "app_key": api_key},
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        logger.debug(f"Mathpix API response: {data}")

        # Extract and sum up all request counts from the ocr_usage array
        # The API returns an array of usage objects grouped by usage_type
        ocr_usage_list = data.get("ocr_usage", [])

        if not ocr_usage_list:
            logger.warning(f"Empty ocr_usage list in response: {data}")

        # Sum all counts (pages/requests processed)
        pages_usage = sum(item.get("count", 0) for item in ocr_usage_list)

        pages_limit = 5000  # Default limit for most plans

        # Calculate percentage
        pages_percentage = (pages_usage / pages_limit * 100) if pages_limit > 0 else 0

        logger.info(
            f"Mathpix usage: {pages_usage}/{pages_limit} pages ({pages_percentage:.1f}%)"
        )

        return MathpixUsageStats(
            pages_usage=pages_usage,
            pages_limit=pages_limit,
            pages_percentage=pages_percentage,
            rate_limit=50,  # 50 requests/minute
        )

    except requests.exceptions.RequestException:
        logger.exception("Failed to fetch Mathpix usage")
        return None
    except (KeyError, ValueError):
        logger.exception("Failed to parse Mathpix usage response")
        return None
