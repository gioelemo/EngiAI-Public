"""
ArXiv Research Agent for searching and analyzing research papers.

This agent combines ArXiv search capabilities with MMORE RAG for paper analysis.
"""

import logging
import os

from src.agents.base_agent import BaseAgent
from src.tools import MMOREClient
from src.tools.arxiv_tools import create_arxiv_tools
from src.utils.prompts import ARXIV_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ArXivAgent(BaseAgent):
    """Agent for ArXiv paper search and analysis with MMORE RAG integration."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        mmore_url: str | None = None,
    ):
        """Initialize the ArXiv agent with search and MMORE RAG capabilities.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            mmore_url: URL of MMORE service (defaults to MMORE_RAG_URL env var)
        """
        # Check if MMORE should be skipped (e.g., during benchmarks)
        skip_mmore = os.getenv("SKIP_MMORE", "false").lower() == "true"

        if skip_mmore:
            logger.info("SKIP_MMORE=true: ArXiv Agent initialized without MMORE client")
            self.mmore_client: MMOREClient | None = None
        else:
            # Initialize MMORE client for paper analysis
            self.mmore_client = MMOREClient(base_url=mmore_url)

        super().__init__(model_name=model_name, temperature=temperature)

        # Verify MMORE connection if client was initialized
        if not skip_mmore and self.mmore_client:
            if self.mmore_client.health_check():
                logger.info("ArXiv Agent initialized with MMORE service")
            else:
                logger.warning(
                    "MMORE service not reachable - some features may not work"
                )

    def _create_tools(self) -> list:
        """Create LangChain tools for the ArXiv agent.

        Returns:
            List of ArXiv tools with MMORE integration
        """
        return create_arxiv_tools(mmore_client=self.mmore_client)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the ArXiv agent.

        Returns:
            System prompt string
        """
        return ARXIV_AGENT_SYSTEM_PROMPT
