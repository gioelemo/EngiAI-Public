"""
Tests for config module.

Tests configuration loading and validation.
"""

import importlib
import os
from unittest.mock import patch

import pytest

import config as config_module
from config import Config, get_setting_from_db


@pytest.mark.unit
def test_config_loads_from_env():
    """Test that config loads from environment variables."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key-123",
            "TAVILY_API_KEY": "test-tavily-key",
            "GOOGLE_API_KEY": "test-google-key",
            "HPC_HOST_ALIAS": "test-cluster",
        },
    ):
        # Reload config to pick up mocked env vars
        importlib.reload(config_module)
        # Access the config object from the reloaded module
        config = config_module.config

        assert config.openai_api_key == "test-key-123"
        assert config.tavily_api_key == "test-tavily-key"
        assert config.google_api_key == "test-google-key"
        assert config.hpc_host_alias == "test-cluster"


@pytest.mark.unit
def test_config_has_required_attributes():
    """Test that config object has all required attributes."""
    config = config_module.config

    # Test that required attributes exist (may be empty strings)
    assert hasattr(config, "openai_api_key")
    assert hasattr(config, "tavily_api_key")
    assert hasattr(config, "google_api_key")
    assert hasattr(config, "hpc_host_alias")
    assert hasattr(config, "llm_model")
    assert hasattr(config, "llm_temperature")
    assert isinstance(config.llm_model, str)
    assert isinstance(config.llm_temperature, float)


@pytest.mark.unit
def test_config_llm_defaults():
    """Test that LLM config has sensible defaults."""
    config = config_module.config

    # Model should be a string
    assert isinstance(config.llm_model, str)
    assert len(config.llm_model) > 0

    # Temperature should be a float
    assert isinstance(config.llm_temperature, float)
    assert 0.0 <= config.llm_temperature <= 2.0

    # HPC host should have a default
    assert isinstance(config.hpc_host_alias, str)


@pytest.mark.unit
def test_config_validation_missing_openai_key():
    """Test that config raises ValueError when OPENAI_API_KEY is missing."""
    with patch("config.load_dotenv"):  # Mock load_dotenv to prevent loading .env file
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}, clear=True):
            with pytest.raises(ValueError):
                Config()


@pytest.mark.unit
def test_config_validation_missing_tavily_key():
    """Test that config raises ValueError when TAVILY_API_KEY is missing."""
    with patch("config.load_dotenv"):  # Mock load_dotenv to prevent loading .env file
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            with pytest.raises(ValueError):
                Config()


@pytest.mark.unit
def test_config_validation_missing_google_key():
    """Test that config raises ValueError when GOOGLE_API_KEY is missing."""
    with patch("config.load_dotenv"):  # Mock load_dotenv to prevent loading .env file
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True):
            with pytest.raises(ValueError):
                Config()


@pytest.mark.unit
def test_config_validation_both_keys_missing():
    """Test that config raises ValueError when both required keys are missing."""
    with patch("config.load_dotenv"):  # Mock load_dotenv to prevent loading .env file
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError):
                Config()


@pytest.mark.unit
def test_config_sets_environment_variables():
    """Test that config sets environment variables for compatibility."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-openai",
            "TAVILY_API_KEY": "test-tavily",
            "GOOGLE_API_KEY": "test-google",
        },
        clear=True,
    ):
        config = Config()

        # Verify env vars are set
        assert os.environ["OPENAI_API_KEY"] == "test-openai"
        assert os.environ["TAVILY_API_KEY"] == "test-tavily"
        assert os.environ["GOOGLE_API_KEY"] == "test-google"


@pytest.mark.unit
def test_config_property_slurm_email_user():
    """Test slurm_email_user property with database fallback."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
        },
        clear=True,
    ):
        config = Config()
        email = config.slurm_email_user

        assert isinstance(email, str)
        # Should get from database or use empty string default
        assert email == ""


@pytest.mark.unit
def test_config_property_slurm_project_path():
    """Test slurm_project_path property with database fallback."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
        },
        clear=True,
    ):
        config = Config()
        project_path = config.slurm_project_path

        assert isinstance(project_path, str)
        assert len(project_path) > 0
        # Should use default value from database or hardcoded default
        assert project_path == "$HOME/EngiOpt"


@pytest.mark.unit
def test_config_property_hf_home_remote():
    """Test hf_home_remote property with database fallback."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
        },
        clear=True,
    ):
        config = Config()
        hf_home = config.hf_home_remote

        assert isinstance(hf_home, str)
        assert len(hf_home) > 0
        # Should use default value from database or hardcoded default
        assert hf_home == "$SCRATCH/models"


@pytest.mark.unit
def test_config_property_hf_datasets_cache_remote():
    """Test hf_datasets_cache_remote property with database fallback."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
        },
        clear=True,
    ):
        config = Config()
        datasets_cache = config.hf_datasets_cache_remote

        assert isinstance(datasets_cache, str)
        assert len(datasets_cache) > 0
        # Should use default value from database or hardcoded default
        assert datasets_cache == "$SCRATCH/datasets"


@pytest.mark.unit
def test_config_setup_langsmith_tracing_enabled():
    """Test LangSmith tracing setup when enabled."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
            "LANGCHAIN_TRACING": "true",
            "LANGCHAIN_ENDPOINT": "https://test.api.com",
        },
        clear=True,
    ):
        config = Config()
        config.setup_langsmith_tracing("test-project")

        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_ENDPOINT") == "https://test.api.com"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"


@pytest.mark.unit
def test_config_setup_langsmith_tracing_disabled():
    """Test LangSmith tracing setup when disabled."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
            "LANGCHAIN_TRACING": "false",
        },
        clear=True,
    ):
        config = Config()

        # Clear any existing tracing env vars
        os.environ.pop("LANGCHAIN_TRACING_V2", None)

        config.setup_langsmith_tracing("test-project")

        # Should not set tracing when disabled
        assert os.environ.get("LANGCHAIN_TRACING_V2") != "true"


@pytest.mark.unit
def test_config_temperature_parsing():
    """Test that temperature is correctly parsed as float."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
            "LLM_TEMPERATURE": "0.5",
        },
        clear=True,
    ):
        config = Config()

        assert config.llm_temperature == 0.5
        assert isinstance(config.llm_temperature, float)


@pytest.mark.unit
def test_config_seed_parsing():
    """Test that seed is correctly parsed as integer or None."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
            "LLM_SEED": "42",
        },
        clear=True,
    ):
        config = Config()
        assert config.llm_seed == 42
        assert isinstance(config.llm_seed, int)


@pytest.mark.unit
def test_config_seed_none_when_unset():
    """Test that seed defaults to None when not set."""
    with patch("config.load_dotenv"):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "TAVILY_API_KEY": "test-key",
                "GOOGLE_API_KEY": "test-key",
            },
            clear=True,
        ):
            config = Config()
            assert config.llm_seed is None


@pytest.mark.unit
def test_config_seed_validation_negative():
    """Test that config raises ValueError for negative seed."""
    with patch("config.load_dotenv"):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "TAVILY_API_KEY": "test-key",
                "GOOGLE_API_KEY": "test-key",
                "LLM_SEED": "-1",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="must be a non-negative integer"):
                Config()


@pytest.mark.unit
def test_config_seed_validation_invalid_string():
    """Test that config raises ValueError for invalid seed string."""
    with patch("config.load_dotenv"):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "TAVILY_API_KEY": "test-key",
                "GOOGLE_API_KEY": "test-key",
                "LLM_SEED": "not-a-number",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="must be an integer"):
                Config()


@pytest.mark.unit
def test_get_setting_from_db_fallback():
    """Test get_setting_from_db returns default when database fails."""
    # DatabaseManager is imported inside get_setting_from_db to avoid circular imports
    with patch("src.ui.database.DatabaseManager") as mock_db:
        # Simulate database error
        mock_db.return_value.get_setting.side_effect = Exception("DB error")

        result = get_setting_from_db("test_key", "default_value")

        assert result == "default_value"


@pytest.mark.unit
def test_config_wandb_configuration():
    """Test WandB configuration loading."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-key",
            "TAVILY_API_KEY": "test-key",
            "GOOGLE_API_KEY": "test-key",
            "WANDB_ENTITY": "test-entity",
            "WANDB_REPORT_URL": "https://wandb.ai/test",
        },
        clear=True,
    ):
        config = Config()

        assert config.wandb_entity == "test-entity"
        assert config.wandb_report_url == "https://wandb.ai/test"
        assert isinstance(config.wandb_personal_project, str)
        assert isinstance(config.wandb_official_project, str)
