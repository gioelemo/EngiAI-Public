"""
Tests for weave_integration module.

These tests cover Weave initialization and availability checking.
Weave automatically instruments LangChain after init() - no decorators needed.
"""

from unittest.mock import patch

import pytest

from src.utils.weave_integration import init_weave, is_weave_enabled

# ============================================================================
# INIT_WEAVE TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_init_weave_disabled(mock_config):
    """Test init_weave returns False when disabled."""
    mock_config.setup_weave_tracing.return_value = False

    result = init_weave()

    assert result is False
    mock_config.setup_weave_tracing.assert_called_once()


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_init_weave_success(mock_config):
    """Test init_weave returns True when successful."""
    mock_config.setup_weave_tracing.return_value = True

    result = init_weave()

    assert result is True
    mock_config.setup_weave_tracing.assert_called_once()


# ============================================================================
# IS_WEAVE_ENABLED TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_is_weave_enabled_false_when_disabled(mock_config):
    """Test is_weave_enabled returns False when Weave is disabled."""
    mock_config.use_weave = False

    result = is_weave_enabled()

    assert result is False


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_is_weave_enabled_false_when_not_installed(mock_config):
    """Test is_weave_enabled returns False when Weave is not installed."""
    mock_config.use_weave = True

    with patch("builtins.__import__", side_effect=ImportError):
        result = is_weave_enabled()

    assert result is False


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_is_weave_enabled_true(mock_config):
    """Test is_weave_enabled returns True when enabled and available."""
    mock_config.use_weave = True

    with patch.dict("sys.modules", {"weave": "mock_weave_module"}):
        result = is_weave_enabled()

    assert result is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@patch("src.utils.weave_integration.config")
def test_full_initialization_workflow(mock_config):
    """Test complete workflow from disabled to enabled."""
    # Test disabled state
    mock_config.use_weave = False
    mock_config.setup_weave_tracing.return_value = False

    assert init_weave() is False
    assert is_weave_enabled() is False

    # Test enabled state
    mock_config.use_weave = True
    mock_config.setup_weave_tracing.return_value = True

    assert init_weave() is True
    assert is_weave_enabled() is True
