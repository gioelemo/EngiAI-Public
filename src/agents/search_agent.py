"""
Search-focused agent with web search capabilities.
"""

from src.agents.base_agent import BaseAgent
from src.tools.search import create_search_tool
from src.utils.prompts import SEARCH_AGENT_SYSTEM_PROMPT


class SearchAgent(BaseAgent):
    """Agent specialized in web search and information retrieval."""

    def __init__(self, model_name: str | None = None):
        """Initialize the search agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        super().__init__(model_name=model_name)

    def _create_tools(self) -> list:
        """Create the list of search tools.

        Returns:
            List of LangChain tools for search operations
        """
        return [create_search_tool()]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the search agent.

        Returns:
            System prompt string
        """
        return SEARCH_AGENT_SYSTEM_PROMPT
