"""
Search tools for web search capabilities.
"""

from langchain_tavily import TavilySearch

from config import config


def create_search_tool(max_results: int = 2) -> TavilySearch:
    """Create a Tavily search tool.

    Args:
        max_results: Maximum number of search results to return

    Returns:
        Configured TavilySearch tool
    """
    return TavilySearch(api_key=config.tavily_api_key, max_results=max_results)
