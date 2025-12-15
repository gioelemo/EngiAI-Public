"""
ArXiv Research Agent for searching and analyzing research papers.

This agent combines ArXiv search capabilities with MMORE RAG for paper analysis.
"""

import logging
from typing import TYPE_CHECKING

from src.agents.base_agent import BaseAgent
from src.tools import MMOREClient
from src.tools.arxiv_tools import create_arxiv_tools
from src.utils.prompts import ARXIV_AGENT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from src.ui.database import DatabaseManager

logger = logging.getLogger(__name__)


class ArXivAgent(BaseAgent):
    """Agent for ArXiv paper search and analysis with MMORE RAG integration."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        mmore_url: str | None = None,
        db_manager: "DatabaseManager | None" = None,
    ):
        """Initialize the ArXiv agent with search and MMORE RAG capabilities.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            mmore_url: URL of MMORE service (defaults to MMORE_RAG_URL env var)
            db_manager: Optional database manager (defaults to creating a new one)
        """
        # Initialize MMORE client for paper analysis
        self.mmore_client = MMOREClient(base_url=mmore_url)

        # Set database manager if provided (for testing)
        if db_manager is not None:
            self._db = db_manager

        super().__init__(model_name=model_name, temperature=temperature)

        # Verify MMORE connection
        if self.mmore_client.health_check():
            logger.info("ArXiv Agent initialized with MMORE service")
        else:
            logger.warning("MMORE service not reachable - some features may not work")

    @property
    def db(self):
        """Lazy-load database manager to avoid circular import."""
        if not hasattr(self, "_db"):
            from src.ui.database import DatabaseManager  # noqa: PLC0415

            self._db = DatabaseManager()
        return self._db

    @db.setter
    def db(self, value):
        """Allow setting database manager (useful for testing)."""
        self._db = value

    def _create_tools(self) -> list:
        """Create LangChain tools for the ArXiv agent.

        Returns:
            List of ArXiv tools with MMORE integration
        """
        return create_arxiv_tools(
            mmore_client=self.mmore_client,
            db_manager=self.db,
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the ArXiv agent.

        Returns:
            System prompt string
        """
        return ARXIV_AGENT_SYSTEM_PROMPT
