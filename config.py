"""
Configuration module inspired from https://github.com/SoheylM/agentic-eng-design.
Handles loading of environment variables and configuration settings.
"""

import os

from dotenv import load_dotenv


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
        self.llm_model: str = os.getenv("LLM_MODEL", "openai:gpt-4.1")

        # HPC/SLURM configuration
        self.hpc_host_alias: str = os.getenv("HPC_HOST_ALIAS", "euler")
        self.euler_hostname: str = os.getenv("EULER_HOSTNAME", "euler.ethz.ch")
        self.euler_username: str = os.getenv("EULER_USERNAME", "")
        self.slurm_email_user: str = os.getenv("SLURM_EMAIL_USER", "")

        # Weights & Biases configuration
        self.wandb_report_url: str = os.getenv("WANDB_REPORT_URL", "")
        self.wandb_personal_project: str = os.getenv(
            "WANDB_PERSONAL_PROJECT", "gioelemo-ethz/engiopt"
        )
        self.wandb_official_project: str = os.getenv(
            "WANDB_OFFICIAL_PROJECT", "engibench/engiopt"
        )

        # CLI Tools configuration
        self.prusa_slicer_path: str = os.getenv("PRUSA_SLICER_PATH", "PrusaSlicer")

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
            raise ValueError

    def _set_env_vars(self) -> None:
        """Set environment variables for compatibility with existing code."""
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["TAVILY_API_KEY"] = self.tavily_api_key

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
