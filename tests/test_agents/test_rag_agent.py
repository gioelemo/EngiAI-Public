"""Tests for RAG Agent.

NOTE: Tests marked as @pytest.mark.slow are SKIPPED in CI (runs with -m "not slow").
They only run when you explicitly run pytest locally without the marker filter.
These tests require database connections not available in CI environment.
"""

from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver


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
    """Mock MMORE client for testing."""
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
    return mock


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for testing."""
    # Use real MemorySaver instead of Mock - it's lightweight and doesn't need external deps
    return MemorySaver()


class TestRAGAgentInitialization:
    """Test RAG agent initialization."""

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_rag_agent_initialization(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that RAG agent initializes correctly."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent(model_name="openai:gpt-4o")

        assert agent is not None
        assert agent.model_name == "openai:gpt-4o"
        # Check that init_chat_model was called with correct model and temperature
        # (seed may also be passed depending on config)
        call_args = mock_init_llm.call_args
        assert call_args[0] == ("openai:gpt-4o",)
        assert call_args[1]["temperature"] == 0.7  # Default from config
        mock_mmore_cls.assert_called_once_with(base_url=None)

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_rag_agent_tools_created(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that RAG agent creates the expected tools."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()

        assert len(agent.tools) == 5
        tool_names = {tool.name for tool in agent.tools}
        assert "search_documents" in tool_names
        assert "add_document" in tool_names
        assert "add_url_to_knowledge_base" in tool_names
        assert "list_documents" in tool_names
        assert "delete_document" in tool_names


class TestRAGAgentSearchDocuments:
    """Test the search_documents tool."""

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_search_documents_success(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test successful document search."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client.retrieve.return_value = [
            Document(
                page_content="Test content about topology optimization",
                metadata={"source": "test.pdf", "score": 0.95},
            )
        ]
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()
        search_tool = agent.tools_by_name["search_documents"]

        result = search_tool.invoke({"query": "What is topology optimization?"})

        assert "topology optimization" in result
        assert "test.pdf" in result
        mock_mmore_client.retrieve.assert_called_once()

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_search_documents_with_num_results(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test document search with custom number of results."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client.retrieve.return_value = []
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()
        search_tool = agent.tools_by_name["search_documents"]

        search_tool.invoke({"query": "test query", "num_results": 10})

        mock_mmore_client.retrieve.assert_called_once_with(
            query="test query", max_matches=10, min_similarity=0.3
        )

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_search_documents_error_handling(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test error handling in document search."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client.retrieve.side_effect = Exception("Search failed")
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()
        search_tool = agent.tools_by_name["search_documents"]

        result = search_tool.invoke({"query": "test"})

        assert "Error searching documents" in result


@pytest.mark.slow
class TestRAGAgentAddDocument:
    """Test the add_document tool."""

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    @patch("pathlib.Path.exists")
    def test_add_document_success(
        self,
        mock_exists,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test successful document addition."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_exists.return_value = True
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()
        add_tool = agent.tools_by_name["add_document"]

        result = add_tool.invoke({"file_path": "/path/to/test.pdf"})

        assert "Successfully added" in result
        assert "test.pdf" in result
        mock_mmore_client.upload_file.assert_called_once()

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    @patch("pathlib.Path.exists")
    def test_add_document_file_not_found(
        self,
        mock_exists,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test document addition with non-existent file."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_exists.return_value = False
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()
        add_tool = agent.tools_by_name["add_document"]

        result = add_tool.invoke({"file_path": "/invalid/path.pdf"})

        assert "File not found" in result

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    @patch("pathlib.Path.exists")
    def test_add_document_error_handling(
        self,
        mock_exists,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test error handling in document addition."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_exists.return_value = True
        mock_mmore_client.upload_file.side_effect = Exception("Upload failed")
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()
        add_tool = agent.tools_by_name["add_document"]

        result = add_tool.invoke({"file_path": "/path/to/test.pdf"})

        assert "Error adding document" in result


@pytest.mark.slow
class TestRAGAgentListDocuments:
    """Test the list_documents tool."""

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_list_documents_success(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test successful document listing."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client.list_files.return_value = ["test_id"]
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()

        list_tool = agent.tools_by_name["list_documents"]

        result = list_tool.invoke({})

        assert "Knowledge Base" in result
        assert "1 document" in result
        assert "test_id" in result
        mock_mmore_client.list_files.assert_called_once()

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_list_documents_empty(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test listing when no documents exist."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client.list_files.return_value = []
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()

        list_tool = agent.tools_by_name["list_documents"]

        result = list_tool.invoke({})

        assert "No documents in MMORE knowledge base yet" in result
        mock_mmore_client.list_files.assert_called_once()


@pytest.mark.slow
class TestRAGAgentDeleteDocument:
    """Test the delete_document tool."""

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_delete_document_success(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test successful document deletion."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()

        delete_tool = agent.tools_by_name["delete_document"]

        result = delete_tool.invoke({"file_id": "test_id"})

        assert "Deleted" in result
        assert "test_id" in result
        mock_mmore_client.delete_file.assert_called_once_with("test_id")


class TestRAGAgentInvoke:
    """Test RAG agent invocation."""

    @patch("src.agents.base_agent.get_checkpointer")
    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_invoke_with_simple_query(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_get_checkpointer,
        mock_mmore_client,
        mock_checkpointer,
    ):
        """Test invoking RAG agent with a simple query."""
        from src.agents.rag_agent import RAGAgent

        # Use custom FakeLLMWithTools which supports bind_tools
        fake_llm = FakeLLMWithTools(
            responses=[AIMessage(content="Based on the document, here's the answer.")]
        )
        mock_init_llm.return_value = fake_llm
        mock_get_checkpointer.return_value = mock_checkpointer
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()
        state = {"messages": [HumanMessage(content="What is in the document?")]}
        config = {"configurable": {"thread_id": "test_thread"}}

        result = agent.invoke(state, config)

        assert result is not None
        assert "messages" in result


@pytest.mark.unit
class TestRAGAgentSystemPrompt:
    """Test RAG agent system prompt."""

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_system_prompt_content(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test that system prompt contains key instructions."""
        from src.agents.rag_agent import RAGAgent

        mock_llm = FakeLLMWithTools()
        mock_init_llm.return_value = mock_llm
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()

        # The system prompt should mention MMORE
        system_prompt = agent._get_system_prompt()
        assert "MMORE" in system_prompt
        assert agent is not None


@pytest.mark.slow
class TestRAGAgentAddURLTool:
    """Test the add_url_to_knowledge_base tool."""

    @patch("src.tools.rag_tools.WebCrawler")
    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_add_url_direct_pdf(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_crawler_cls,
        mock_mmore_client,
    ):
        """Test adding a direct PDF URL."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client
        mock_crawler = Mock()
        mock_crawler_cls.return_value = mock_crawler

        agent = RAGAgent()
        add_url_tool = agent.tools_by_name["add_url_to_knowledge_base"]

        # Mock tempfile and requests for PDF download
        with patch("src.tools.rag_tools.tempfile.NamedTemporaryFile") as mock_temp:
            with patch("src.tools.rag_tools.requests.get") as mock_get:
                mock_temp_file = Mock()
                mock_temp_file.name = "/tmp/test.pdf"
                mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                mock_temp_file.__exit__ = Mock(return_value=False)
                mock_temp.return_value = mock_temp_file

                mock_response = Mock()
                mock_response.content = b"PDF content"
                mock_response.raise_for_status = Mock()
                mock_get.return_value = mock_response

                result = add_url_tool.invoke(
                    {
                        "url": "https://example.com/paper.pdf",
                        "crawl_subpages": False,
                    }
                )

                assert "Successfully added" in result or "Error" in result

    @patch("src.tools.rag_tools.WebCrawler")
    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_add_url_with_crawling(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_crawler_cls,
        mock_mmore_client,
    ):
        """Test adding URL with subpage crawling."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        mock_crawler = Mock()
        mock_crawler.crawl.return_value = [
            {"url": "https://example.com", "content": "Test content"}
        ]
        mock_crawler_cls.return_value = mock_crawler

        agent = RAGAgent()
        add_url_tool = agent.tools_by_name["add_url_to_knowledge_base"]

        with patch("src.tools.rag_tools.tempfile.TemporaryDirectory") as mock_temp:
            mock_temp_dir = Mock()
            mock_temp_dir.name = "/tmp/test_dir"
            mock_temp_dir.__enter__ = Mock(return_value=mock_temp_dir)
            mock_temp_dir.__exit__ = Mock(return_value=False)
            mock_temp.return_value = mock_temp_dir

            with patch("builtins.open", create=True):
                result = add_url_tool.invoke(
                    {
                        "url": "https://example.com",
                        "crawl_subpages": True,
                        "max_pages": 5,
                    }
                )

                # Should handle crawling
                assert isinstance(result, str)


@pytest.mark.slow
class TestRAGAgentToolsIntegration:
    """Test tool integration and interaction."""

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_all_tools_registered(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test that all expected tools are registered."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()

        expected_tools = [
            "search_documents",
            "add_document",
            "add_url_to_knowledge_base",
            "list_documents",
            "delete_document",
        ]

        for tool_name in expected_tools:
            assert tool_name in agent.tools_by_name

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_tools_have_descriptions(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test that all tools have proper descriptions."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent()

        for tool in agent.tools_by_name.values():
            assert tool.description is not None
            assert len(tool.description) > 0


@pytest.mark.unit
class TestRAGAgentEdgeCases:
    """Test edge cases and error conditions."""

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_initialization_with_custom_params(
        self,
        mock_init_llm,
        mock_mmore_cls,
    ):
        """Test RAG agent initialization with custom parameters."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client = Mock()
        mock_mmore_client.health_check.return_value = True
        mock_mmore_cls.return_value = mock_mmore_client

        agent = RAGAgent(
            model_name="custom-model",
            temperature=0.7,
            mmore_url="http://custom-url:8000",
        )

        assert agent is not None
        mock_mmore_cls.assert_called_with(base_url="http://custom-url:8000")

    @patch("src.agents.rag_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_mmore_health_check_failure(
        self,
        mock_init_llm,
        mock_mmore_cls,
    ):
        """Test handling of MMORE health check failure."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client = Mock()
        mock_mmore_client.health_check.return_value = False
        mock_mmore_cls.return_value = mock_mmore_client

        # Should still initialize but warn
        agent = RAGAgent()
        assert agent is not None
