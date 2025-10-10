"""
State definitions for agent workflows.
"""

import operator
from typing import Annotated, NotRequired

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class MessagesState(TypedDict):
    """State for message-based agents.

    Attributes:
        messages: List of conversation messages (accumulated with operator.add)
        llm_calls: Optional counter for number of LLM calls made
    """

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: NotRequired[int]
