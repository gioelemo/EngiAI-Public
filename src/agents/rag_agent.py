"""
RAG Agent for document-based question answering.

This agent handles queries about uploaded documents using the RAG system.
Now powered by MMORE for advanced multimodal document processing.
"""

import logging
import os
from typing import Literal

from langchain_core.messages import ToolMessage

from src.agents.base_agent import BaseAgent
from src.tools import MMOREClient
from src.tools.rag_tools import create_rag_tools
from src.utils.prompts import build_rag_agent_prompt

logger = logging.getLogger(__name__)

# Maximum number of tool calls before the RAG agent stops searching.
# Prevents infinite search loops when the index is empty or queries return nothing.
_MAX_RAG_TOOL_CALLS = 6


class RAGAgent(BaseAgent):
    """Agent for document-based question answering using MMORE RAG."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        mmore_url: str | None = None,
        seed: int | None = None,
        rag_read_only: bool = False,
    ):
        """Initialize the RAG agent with MMORE client.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            mmore_url: URL of MMORE service (defaults to MMORE_RAG_URL env var)
            seed: Random seed for model (defaults to config.llm_seed)
            rag_read_only: If True, only expose read-only RAG tools (search, list).
        """
        # Store before super().__init__() so _create_tools() can use it
        self.rag_read_only = rag_read_only

        # Check if MMORE should be skipped (e.g., during benchmarks)
        skip_mmore = os.getenv("SKIP_MMORE", "false").lower() == "true"

        if skip_mmore:
            logger.info("SKIP_MMORE=true: RAG Agent initialized without MMORE client")
            self.mmore_client: MMOREClient | None = None
        else:
            # Initialize MMORE client
            self.mmore_client = MMOREClient(base_url=mmore_url)

        super().__init__(model_name=model_name, temperature=temperature, seed=seed)

        # Verify MMORE connection if client was initialized
        if not skip_mmore and self.mmore_client:
            if self.mmore_client.health_check():
                logger.info("RAG Agent initialized with MMORE service")
            else:
                logger.warning(
                    "MMORE service not reachable - some features may not work"
                )

    def _create_tools(self) -> list:
        """Create LangChain tools for the RAG agent."""
        if self.mmore_client is None:
            return []

        return create_rag_tools(self.mmore_client, read_only=self.rag_read_only)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the RAG agent.

        Returns:
            System prompt string
        """
        return build_rag_agent_prompt(read_only=self.rag_read_only)

    def _after_tools(
        self, state
    ) -> Literal["llm_call", "clarification_response", "__end__"]:
        """Route after tool execution, with a cap on total tool calls.

        Prevents infinite search loops when the RAG index returns no results.
        """
        tool_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
        if tool_count >= _MAX_RAG_TOOL_CALLS:
            logger.warning(
                f"RAG agent reached tool call limit ({tool_count}/{_MAX_RAG_TOOL_CALLS}), stopping"
            )
            return "__end__"
        return super()._after_tools(state)
