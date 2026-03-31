"""Shared fixtures for agent tests."""

from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from src.agents.cli_agent import CLIAgent


class FakeLLMWithTools(BaseChatModel):
    """Fake LLM that supports bind_tools() for testing."""

    def __init__(self, responses: list[AIMessage] | None = None, **kwargs):
        """Initialize with a list of responses."""
        super().__init__(**kwargs)
        self._responses = responses or [AIMessage(content="Default response")]
        self._response_index = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
        else:
            response = self._responses[-1]  # Reuse last response

        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to the model - just return self for testing."""
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "fake-llm-with-tools"


@pytest.fixture
def mock_mmore_client():
    """Mock MMORE client with standard stubs for health, retrieve, upload, delete, list."""
    mock = Mock()
    mock.health_check.return_value = True
    mock.retrieve.return_value = [
        Document(
            page_content="Test content from document",
            metadata={"source": "test.pdf", "chunk_id": "1", "score": 0.95},
        )
    ]
    mock.upload_file.return_value = {"status": "success", "fileId": "test_id"}
    mock.delete_file.return_value = {"status": "success"}
    mock.list_files.return_value = ["test.pdf"]
    return mock


@pytest.fixture
def mock_checkpointer():
    """Lightweight in-memory checkpointer for testing."""
    return MemorySaver()


def make_cli_agent() -> CLIAgent:
    """Return a CLIAgent with a mocked LLM."""
    with patch("src.agents.base_agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_init.return_value = mock_llm
        return CLIAgent()
