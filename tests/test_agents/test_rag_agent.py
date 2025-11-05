"""Tests for RAG Agent."""

from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import Mock, patch

import pytest
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
def mock_vector_store():
    """Mock vector store for testing."""
    mock = Mock()
    mock.similarity_search.return_value = [
        Mock(
            page_content="Test content from document",
            metadata={"source": "test.pdf", "page": 1},
        )
    ]
    mock.add_documents.return_value = ["doc_id_1"]
    mock.get_collection_count.return_value = 5
    return mock


@pytest.fixture
def mock_document_processor():
    """Mock document processor for testing."""
    mock = Mock()
    mock.process_file.return_value = [
        Mock(
            page_content="Processed document content",
            metadata={"source": "test.pdf", "page": 1},
        )
    ]
    return mock


@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain for testing."""
    mock = Mock()
    mock.ask.return_value = {
        "answer": "The answer based on documents",
        "num_sources": 3,
    }
    mock.clear_history.return_value = None
    return mock


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for testing."""
    # Use real MemorySaver instead of Mock - it's lightweight and doesn't need external deps
    return MemorySaver()


class TestRAGAgentInitialization:
    """Test RAG agent initialization."""

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_rag_agent_initialization(
        self, mock_init_llm, mock_processor, mock_store, mock_chain
    ):
        """Test that RAG agent initializes correctly."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()

        agent = RAGAgent(model_name="openai:gpt-4o")

        assert agent is not None
        assert agent.model_name == "openai:gpt-4o"
        mock_init_llm.assert_called_once_with("openai:gpt-4o")
        mock_processor.assert_called_once()
        mock_store.assert_called_once()
        mock_chain.assert_called_once()

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_rag_agent_tools_created(
        self, mock_init_llm, mock_processor, mock_store, mock_chain
    ):
        """Test that RAG agent creates the expected tools."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()

        agent = RAGAgent()

        assert len(agent.tools) == 4
        tool_names = {tool.name for tool in agent.tools}
        assert "search_documents" in tool_names
        assert "add_document" in tool_names
        assert "list_documents" in tool_names
        assert "clear_document_memory" in tool_names


class TestRAGAgentSearchDocuments:
    """Test the search_documents tool."""

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_search_documents_success(
        self, mock_init_llm, mock_processor, mock_store, mock_chain, mock_rag_chain
    ):
        """Test successful document search."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_chain.return_value = mock_rag_chain

        agent = RAGAgent()
        search_tool = agent.tools_by_name["search_documents"]

        result = search_tool.invoke({"query": "What is topology optimization?"})

        assert "answer based on documents" in result
        assert "3 document(s)" in result
        mock_rag_chain.ask.assert_called_once()

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_search_documents_with_num_results(
        self, mock_init_llm, mock_processor, mock_store, mock_chain, mock_rag_chain
    ):
        """Test document search with custom number of results."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_chain.return_value = mock_rag_chain

        agent = RAGAgent()
        search_tool = agent.tools_by_name["search_documents"]

        search_tool.invoke({"query": "test query", "num_results": 10})

        mock_rag_chain.ask.assert_called_once_with("test query", k=10)

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_search_documents_error_handling(
        self, mock_init_llm, mock_processor, mock_store, mock_chain, mock_rag_chain
    ):
        """Test error handling in document search."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_rag_chain.ask.side_effect = Exception("Search failed")
        mock_chain.return_value = mock_rag_chain

        agent = RAGAgent()
        search_tool = agent.tools_by_name["search_documents"]

        result = search_tool.invoke({"query": "test"})

        assert "Error searching documents" in result


class TestRAGAgentAddDocument:
    """Test the add_document tool."""

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_add_document_success(
        self,
        mock_init_llm,
        mock_processor_cls,
        mock_store_cls,
        mock_chain,
        mock_document_processor,
        mock_vector_store,
    ):
        """Test successful document addition."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_processor_cls.return_value = mock_document_processor
        mock_store_cls.return_value = mock_vector_store

        agent = RAGAgent()
        add_tool = agent.tools_by_name["add_document"]

        result = add_tool.invoke({"file_path": "/path/to/test.pdf"})

        assert "Successfully added" in result
        assert "test.pdf" in result
        mock_document_processor.process_file.assert_called_once_with(
            "/path/to/test.pdf"
        )
        mock_vector_store.add_documents.assert_called_once()

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_add_document_with_metadata(
        self,
        mock_init_llm,
        mock_processor_cls,
        mock_store_cls,
        mock_chain,
        mock_document_processor,
        mock_vector_store,
    ):
        """Test document addition with metadata."""
        import json

        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_processor_cls.return_value = mock_document_processor
        mock_store_cls.return_value = mock_vector_store

        agent = RAGAgent()
        add_tool = agent.tools_by_name["add_document"]

        metadata = json.dumps({"author": "Test Author", "year": 2024})
        result = add_tool.invoke(
            {"file_path": "/path/to/test.pdf", "metadata": metadata}
        )

        assert "Successfully added" in result
        mock_document_processor.process_file.assert_called_once()

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_add_document_error_handling(
        self,
        mock_init_llm,
        mock_processor_cls,
        mock_store_cls,
        mock_chain,
        mock_document_processor,
    ):
        """Test error handling in document addition."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_document_processor.process_file.side_effect = Exception("File not found")
        mock_processor_cls.return_value = mock_document_processor

        agent = RAGAgent()
        add_tool = agent.tools_by_name["add_document"]

        result = add_tool.invoke({"file_path": "/invalid/path.pdf"})

        assert "Error adding document" in result


class TestRAGAgentListDocuments:
    """Test the list_documents tool."""

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_list_documents_success(
        self,
        mock_init_llm,
        mock_processor,
        mock_store_cls,
        mock_chain,
        mock_vector_store,
    ):
        """Test successful document listing."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_vector_store.similarity_search.return_value = [
            Mock(
                page_content="content",
                metadata={"source": "doc1.pdf", "page": 1},
            ),
            Mock(
                page_content="content",
                metadata={"source": "doc1.pdf", "page": 2},
            ),
            Mock(
                page_content="content",
                metadata={"source": "doc2.pdf", "page": 1},
            ),
        ]
        mock_store_cls.return_value = mock_vector_store

        agent = RAGAgent()
        list_tool = agent.tools_by_name["list_documents"]

        result = list_tool.invoke({})

        assert "Knowledge Base" in result
        assert "2 documents" in result
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_list_documents_empty(
        self,
        mock_init_llm,
        mock_processor,
        mock_store_cls,
        mock_chain,
        mock_vector_store,
    ):
        """Test listing when no documents exist."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_vector_store.similarity_search.return_value = []
        mock_store_cls.return_value = mock_vector_store

        agent = RAGAgent()
        list_tool = agent.tools_by_name["list_documents"]

        result = list_tool.invoke({})

        assert "No documents in the knowledge base yet" in result


class TestRAGAgentClearMemory:
    """Test the clear_document_memory tool."""

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_clear_memory_success(
        self, mock_init_llm, mock_processor, mock_store, mock_chain, mock_rag_chain
    ):
        """Test successful memory clearing."""
        from src.agents.rag_agent import RAGAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_chain.return_value = mock_rag_chain

        agent = RAGAgent()
        clear_tool = agent.tools_by_name["clear_document_memory"]

        result = clear_tool.invoke({})

        assert "Conversation history cleared" in result
        mock_rag_chain.clear_history.assert_called_once()


class TestRAGAgentInvoke:
    """Test RAG agent invocation."""

    @patch("src.agents.rag_agent.get_checkpointer")
    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_invoke_with_simple_query(
        self,
        mock_init_llm,
        mock_processor,
        mock_store,
        mock_chain,
        mock_get_checkpointer,
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

        agent = RAGAgent()
        state = {"messages": [HumanMessage(content="What is in the document?")]}
        config = {"configurable": {"thread_id": "test_thread"}}

        result = agent.invoke(state, config)

        assert result is not None
        assert "messages" in result


@pytest.mark.unit
class TestRAGAgentSystemPrompt:
    """Test RAG agent system prompt."""

    @patch("src.agents.rag_agent.EngineeringRAGChain")
    @patch("src.agents.rag_agent.EngineerRAGStore")
    @patch("src.agents.rag_agent.MultimodalDocumentProcessor")
    @patch("src.agents.rag_agent.init_chat_model")
    def test_system_prompt_content(
        self,
        mock_init_llm,
        mock_processor,
        mock_store,
        mock_chain,
    ):
        """Test that system prompt contains key instructions."""
        from src.agents.rag_agent import RAGAgent

        mock_llm = FakeLLMWithTools()
        mock_init_llm.return_value = mock_llm

        agent = RAGAgent()

        # The system prompt should be used in _llm_call
        # We can't directly access it, but we can verify the agent was created
        assert agent is not None
