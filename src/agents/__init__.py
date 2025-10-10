"""
Agents module for the engineer assistant.

Contains different specialized agents.
"""

from src.agents.general_agent import GeneralAgent
from src.agents.math_agent import MathAgent
from src.agents.search_agent import SearchAgent

__all__ = ["GeneralAgent", "MathAgent", "SearchAgent"]
