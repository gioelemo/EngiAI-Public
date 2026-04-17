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
MIN_SEED = 0

# HTTP status codes
HTTP_OK = 200


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

        # MMORE health check cache
        self._mmore_enabled: bool | None = None

        # Required API keys
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY") or ""
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY") or ""
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY") or ""

        # Model configuration
        self.llm_model: str = os.getenv("LLM_MODEL", "openai:gpt-4.1")
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.llm_seed: int | None = self._parse_seed(os.getenv("LLM_SEED"))

        # Ollama-specific configuration
        # num_ctx controls the context window size for Ollama models (default: 32768)
        # Increase this for models that support larger contexts or if hitting context limits
        self.ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "32768"))

        # Validate temperature is in valid range
        if not MIN_TEMPERATURE <= self.llm_temperature <= MAX_TEMPERATURE:
            raise ValueError(
                f"LLM_TEMPERATURE must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, "
                f"got {self.llm_temperature}"
            )

        # Validate seed if provided
        if self.llm_seed is not None and self.llm_seed < MIN_SEED:
            raise ValueError(
                f"LLM_SEED must be a non-negative integer, got {self.llm_seed}"
            )

        logger.info(
            f"Loaded LLM configuration: model={self.llm_model}, "
            f"temperature={self.llm_temperature}, seed={self.llm_seed}"
        )

        # HPC/SLURM configuration
        self.hpc_host_alias: str = os.getenv("HPC_HOST_ALIAS", "")
        self.hpc_hostname: str = os.getenv("HPC_HOSTNAME", "")
        self.hpc_username: str = os.getenv("HPC_USERNAME", "")

        # Initialize with env vars - database lookups are deferred to avoid circular imports
        self._slurm_email_user: str | None = None
        self._slurm_venv_path: str | None = None
        self._slurm_project_path: str | None = None
        self._hf_home_remote: str | None = None
        self._hf_datasets_cache_remote: str | None = None

        # Weights & Biases configuration
        self.wandb_entity: str = os.getenv("WANDB_ENTITY", "")
        self.wandb_report_url: str = os.getenv("WANDB_REPORT_URL", "")
        self.wandb_personal_project: str = os.getenv("WANDB_PERSONAL_PROJECT", "")
        self.wandb_official_project: str = os.getenv("WANDB_OFFICIAL_PROJECT", "")

        # Weave configuration (for LLM tracing and benchmarking)
        self.use_weave: bool = os.getenv("USE_WEAVE", "false").lower() == "true"
        self.use_weave_chatbot: bool = (
            os.getenv("USE_WEAVE_CHATBOT", "false").lower() == "true"
        )
        self.weave_project: str = os.getenv("WEAVE_PROJECT", "")

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

    def _parse_seed(self, seed_str: str | None) -> int | None:
        """Parse seed from environment variable.

        Args:
            seed_str: Seed value from environment

        Returns:
            Integer seed or None if not set

        Raises:
            ValueError: If seed_str is not a valid integer
        """
        if seed_str is None or seed_str.strip() == "":
            return None

        try:
            return int(seed_str)
        except ValueError as e:
            raise ValueError(
                f"LLM_SEED must be an integer or empty, got '{seed_str}'"
            ) from e

    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        if not self.openai_api_key and not self.google_api_key:
            logger.warning(
                "Neither OPENAI_API_KEY nor GOOGLE_API_KEY is set. "
                "At least one LLM provider key is required to run the chatbot. "
                "Please set one of them in your .env file."
            )

        if not self.tavily_api_key:
            logger.warning(
                "TAVILY_API_KEY is not set. Web search functionality will be unavailable."
            )

    def _set_env_vars(self) -> None:
        """Set environment variables for compatibility with existing code."""
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["TAVILY_API_KEY"] = self.tavily_api_key
        os.environ["GOOGLE_API_KEY"] = self.google_api_key

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
                "~/venvs/engiai",
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

    @property
    def mmore_enabled(self) -> bool:
        """Check if MMORE RAG system is enabled and running.

        Returns True only if:
        1. SKIP_MMORE env var is not "true"
        2. MMORE service is actually reachable (health check)

        The result is cached after the first check.
        """
        if self._mmore_enabled is not None:
            return self._mmore_enabled

        # Check if SKIP_MMORE is set to true
        skip_mmore = os.getenv("SKIP_MMORE", "false").lower() == "true"
        if skip_mmore:
            logger.info("MMORE disabled via SKIP_MMORE=true")
            self._mmore_enabled = False
            return False

        # Check if MMORE service is actually running
        try:
            import requests  # noqa: PLC0415

            mmore_url = os.getenv("MMORE_RAG_URL", "http://localhost:8000").rstrip("/")
            response = requests.get(f"{mmore_url}/", timeout=5)
            is_healthy = response.status_code == HTTP_OK
            if is_healthy:
                logger.info(f"MMORE service is running at {mmore_url}")
            else:
                logger.warning(
                    f"MMORE service returned status {response.status_code} at {mmore_url}"
                )
            self._mmore_enabled = is_healthy
        except requests.RequestException as e:
            # Only warn if MMORE was actually needed (not explicitly skipped)
            # This prevents noise when running evaluations without RAG
            if not skip_mmore:
                logger.warning(f"MMORE service not reachable: {e}")
            self._mmore_enabled = False
        except ImportError:
            logger.warning("requests library not installed, cannot check MMORE health")
            self._mmore_enabled = False

        return self._mmore_enabled

    def reset_mmore_cache(self) -> None:
        """Reset the MMORE enabled cache to force a fresh health check.

        Call this after changing SKIP_MMORE env var at runtime.
        """
        self._mmore_enabled = None

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

    def setup_weave_tracing(self) -> bool:
        """
        Set up Weave tracing for LLM calls and benchmarking.

        Automatically instruments LangChain components for tracing.
        See: https://docs.wandb.ai/weave/guides/integrations/langchain

        Returns:
            True if Weave was successfully initialized, False otherwise.
        """
        if not self.use_weave:
            logger.info("Weave tracing is disabled (USE_WEAVE=false)")
            return False

        try:
            import weave  # noqa: PLC0415

            # Initialize Weave with the configured project
            # This automatically instruments LangChain - no decorators needed
            weave.init(self.weave_project)
            logger.info(
                f"Weave tracing initialized for project: {self.weave_project} "
                "(LangChain auto-instrumentation enabled)"
            )
        except ImportError:
            logger.warning("Weave is not installed. Install with: pip install weave")
            return False
        except Exception:
            logger.exception("Failed to initialize Weave tracing")
            return False
        else:
            return True


# Create a global config instance
config = Config()
