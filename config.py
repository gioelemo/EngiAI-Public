"""
Configuration module inspired from https://github.com/SoheylM/agentic-eng-design.
Handles loading of environment variables and configuration settings.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Constants for LLM configuration validation
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0


def get_setting_from_db(key: str, default: Any = None) -> Any:
    """
    Get a setting value from the database.

    Args:
        key: Setting key
        default: Default value if setting not found

    Returns:
        Setting value or default
    """
    try:
        # Import here to avoid circular dependency at runtime
        from src.ui.database import DatabaseManager  # noqa: PLC0415

        db = DatabaseManager()
        return db.get_setting(key, default)
    except Exception as e:
        logger.warning(f"Could not retrieve setting '{key}' from database: {e}")
        return default


class Config:
    """Configuration class for the agentic engineering design system."""

    def __init__(self, env_file: str | None = None):
        """
        Initialize the configuration.

        Args:
            env_file: Optional path to .env file. If None, will look for .env in current directory.
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        # Required API keys
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY") or ""
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY") or ""

        # Model configuration
        self.llm_model: str = os.getenv("LLM_MODEL", "openai:gpt-4o")
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

        # Validate temperature is in valid range
        if not MIN_TEMPERATURE <= self.llm_temperature <= MAX_TEMPERATURE:
            raise ValueError(  # noqa: TRY003
                f"LLM_TEMPERATURE must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, "
                f"got {self.llm_temperature}"
            )

        logger.info(
            f"Loaded LLM configuration: model={self.llm_model}, temperature={self.llm_temperature}"
        )

        # HPC/SLURM configuration
        self.hpc_host_alias: str = os.getenv("HPC_HOST_ALIAS", "euler")
        self.euler_hostname: str = os.getenv("EULER_HOSTNAME", "euler.ethz.ch")
        self.euler_username: str = os.getenv("EULER_USERNAME", "")

        # Initialize with env vars - database lookups are deferred to avoid circular imports
        self._slurm_email_user: str | None = None
        self._slurm_venv_path: str | None = None
        self._slurm_project_path: str | None = None
        self._hf_home_remote: str | None = None
        self._hf_datasets_cache_remote: str | None = None

        # Weights & Biases configuration
        self.wandb_entity: str = os.getenv("WANDB_ENTITY", "")
        self.wandb_report_url: str = os.getenv("WANDB_REPORT_URL", "")
        self.wandb_personal_project: str = os.getenv(
            "WANDB_PERSONAL_PROJECT", "gioelemo-ethz/engiopt"
        )
        self.wandb_official_project: str = os.getenv(
            "WANDB_OFFICIAL_PROJECT", "engibench/engiopt"
        )

        # CLI Tools configuration
        self.prusa_slicer_path: str = os.getenv("PRUSA_SLICER_PATH", "PrusaSlicer")

        # Database configuration for persistent checkpointing
        self.database_url: str = os.getenv(
            "DATABASE_URL", "sqlite:///data/conversations.db"
        )

        # LangSmith configuration
        self.langchain_tracing = (
            os.getenv("LANGCHAIN_TRACING", "false").lower() == "true"
        )
        self.langchain_endpoint = os.getenv(
            "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
        )
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT")

        # Validate required configuration
        self._validate_config()

        # Set environment variables for compatibility
        self._set_env_vars()

    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        required_vars = {
            "OPENAI_API_KEY": self.openai_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
        }

        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(  # noqa: TRY003
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Please set them in your .env file."
            )

    def _set_env_vars(self) -> None:
        """Set environment variables for compatibility with existing code."""
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["TAVILY_API_KEY"] = self.tavily_api_key

    @property
    def slurm_email_user(self) -> str:
        """Get SLURM email user from database or use default."""
        if self._slurm_email_user is None:
            self._slurm_email_user = get_setting_from_db("slurm_email_user", "")
        return self._slurm_email_user

    @property
    def slurm_venv_path(self) -> str:
        """Get SLURM venv path from database or use default."""
        if self._slurm_venv_path is None:
            self._slurm_venv_path = get_setting_from_db(
                "slurm_venv_path",
                "~/venvs/engineer_assistant",
            )
        return self._slurm_venv_path

    @property
    def slurm_project_path(self) -> str:
        """Get SLURM project path from database or use default."""
        if self._slurm_project_path is None:
            self._slurm_project_path = get_setting_from_db(
                "slurm_project_path", "$HOME/EngiOpt"
            )
        return self._slurm_project_path

    @property
    def hf_home_remote(self) -> str:
        """Get HuggingFace home remote path from database or use default."""
        if self._hf_home_remote is None:
            self._hf_home_remote = get_setting_from_db(
                "hf_home_remote",
                "$SCRATCH/models",
            )
        return self._hf_home_remote

    @property
    def hf_datasets_cache_remote(self) -> str:
        """Get HuggingFace datasets cache remote path from database or use default."""
        if self._hf_datasets_cache_remote is None:
            self._hf_datasets_cache_remote = get_setting_from_db(
                "hf_datasets_cache_remote",
                "$SCRATCH/datasets",
            )
        return self._hf_datasets_cache_remote

    def setup_langsmith_tracing(self, project_name: str) -> None:
        """
        Set up LangSmith tracing for a specific project.

        Args:
            project_name: Name of the project to trace in LangSmith
        """
        if not self.langchain_tracing:
            return

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
        os.environ["LANGCHAIN_PROJECT"] = project_name


# Create a global config instance
config = Config()
