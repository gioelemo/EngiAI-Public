"""
Tests for config module.

Tests configuration loading and validation.
"""

import importlib
from unittest.mock import patch

import pytest

import config as config_module


@pytest.mark.unit
def test_config_loads_from_env():
    """Test that config loads from environment variables."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key-123",
            "HPC_HOST_ALIAS": "test-cluster",
            "PRUSA_SLICER_PATH": "/test/path/prusa",
        },
    ):
        # Reload config to pick up mocked env vars
        importlib.reload(config_module)
        # Access the config object from the reloaded module
        config = config_module.config

        assert config.openai_api_key == "test-key-123"
        assert config.hpc_host_alias == "test-cluster"


@pytest.mark.unit
def test_config_has_required_attributes():
    """Test that config object has all required attributes."""
    config = config_module.config

    # Test that required attributes exist (may be empty strings)
    assert hasattr(config, "openai_api_key")
    assert hasattr(config, "hpc_host_alias")
    assert hasattr(config, "llm_model")
    # Config doesn't have llm_temperature, it uses defaults in model init
    assert isinstance(config.llm_model, str)


@pytest.mark.unit
def test_config_llm_defaults():
    """Test that LLM config has sensible defaults."""
    config = config_module.config

    # Model should be a string
    assert isinstance(config.llm_model, str)
    assert len(config.llm_model) > 0

    # HPC host should have a default
    assert isinstance(config.hpc_host_alias, str)
