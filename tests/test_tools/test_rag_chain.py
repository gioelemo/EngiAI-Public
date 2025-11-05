"""Tests for EngineeringRAGChain."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_vector_store():
    """Mock vector store for RAG chain testing."""
    mock = Mock()
    mock.similarity_search.return_value = [
        Document(
            page_content="Topology optimization is a method for optimizing material layout.",
            metadata={"source": "paper.pdf", "page": 5, "chunk_id": 0},
        ),
        Document(
            page_content="The SIMP method uses a power law approach.",
            metadata={"source": "paper.pdf", "page": 6, "chunk_id": 1},
        ),
    ]
    mock.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="Content 1",
                metadata={"source": "doc1.pdf", "page": 1, "chunk_id": 0},
            ),
            0.92,
        ),
        (
            Document(
                page_content="Content 2",
                metadata={"source": "doc2.pdf", "page": 2, "chunk_id": 1},
            ),
            0.85,
        ),
    ]
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for RAG chain testing."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage

    # Use FakeMessagesListChatModel which can be reused multiple times
    return FakeMessagesListChatModel(
        responses=[AIMessage(content="This is the generated answer based on context.")]
        * 10  # Support multiple calls
    )


class TestEngineeringRAGChainInitialization:
    """Test RAG chain initialization."""

    @patch("src.tools.rag_chain.init_chat_model")
    def test_initialization_default_params(self, mock_init_llm, mock_vector_store):
        """Test initialization with default parameters."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = Mock()

        chain = EngineeringRAGChain(mock_vector_store)

        assert chain.vectorstore == mock_vector_store
        assert chain.chat_history == []
        mock_init_llm.assert_called_once()

    @patch("src.tools.rag_chain.init_chat_model")
    def test_initialization_custom_params(self, mock_init_llm, mock_vector_store):
        """Test initialization with custom parameters."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = Mock()
        custom_prompt = "Custom prompt: {context}\n{question}"

        chain = EngineeringRAGChain(
            mock_vector_store,
            model_name="openai:gpt-4o",
            system_prompt=custom_prompt,
            temperature=0.5,
        )

        assert chain.model_name == "openai:gpt-4o"
        mock_init_llm.assert_called_once_with("openai:gpt-4o", temperature=0.5)


class TestEngineeringRAGChainAsk:
    """Test the ask method."""

    @patch("src.tools.rag_chain.init_chat_model")
    def test_ask_basic_question(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test asking a basic question."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        result = chain.ask("What is topology optimization?")

        assert result["answer"] == "This is the generated answer based on context."
        assert result["question"] == "What is topology optimization?"
        assert "source_documents" in result
        assert result["num_sources"] == 2
        mock_vector_store.similarity_search.assert_called_once()

    @patch("src.tools.rag_chain.init_chat_model")
    def test_ask_with_custom_k(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test asking with custom number of documents to retrieve."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        chain.ask("Test question", k=10)

        mock_vector_store.similarity_search.assert_called_once_with(
            "Test question", k=10, filter_dict=None
        )

    @patch("src.tools.rag_chain.init_chat_model")
    def test_ask_with_filter(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test asking with metadata filter."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        filter_dict = {"source": "specific.pdf"}
        chain.ask("Test question", filter_dict=filter_dict)

        mock_vector_store.similarity_search.assert_called_once_with(
            "Test question", k=5, filter_dict=filter_dict
        )

    @patch("src.tools.rag_chain.init_chat_model")
    def test_ask_without_sources(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test asking without including sources in response."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        result = chain.ask("Test question", include_sources=False)

        assert "answer" in result
        assert "source_documents" not in result
        assert "num_sources" not in result


class TestEngineeringRAGChainFormatting:
    """Test document formatting methods."""

    @patch("src.tools.rag_chain.init_chat_model")
    def test_format_docs_with_documents(self, mock_init_llm, mock_vector_store):
        """Test formatting documents for context."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = Mock()

        chain = EngineeringRAGChain(mock_vector_store)
        docs = [
            Document(
                page_content="Content 1",
                metadata={"source": "doc1.pdf", "page": 1, "chunk_id": 0},
            ),
            Document(
                page_content="Content 2",
                metadata={"source": "doc2.pdf", "page": 2, "chunk_id": 1},
            ),
        ]

        formatted = chain._format_docs(docs)

        assert "[Document 1]" in formatted
        assert "[Document 2]" in formatted
        assert "Content 1" in formatted
        assert "Content 2" in formatted
        assert "doc1.pdf" in formatted
        assert "doc2.pdf" in formatted

    @patch("src.tools.rag_chain.init_chat_model")
    def test_format_docs_empty(self, mock_init_llm, mock_vector_store):
        """Test formatting empty document list."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = Mock()

        chain = EngineeringRAGChain(mock_vector_store)
        formatted = chain._format_docs([])

        assert formatted == "No relevant documents found."

    @patch("src.tools.rag_chain.init_chat_model")
    def test_format_chat_history_empty(self, mock_init_llm, mock_vector_store):
        """Test formatting empty chat history."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = Mock()

        chain = EngineeringRAGChain(mock_vector_store)
        formatted = chain._format_chat_history()

        assert formatted == "No previous conversation."

    @patch("src.tools.rag_chain.init_chat_model")
    def test_format_chat_history_with_messages(self, mock_init_llm, mock_vector_store):
        """Test formatting chat history with messages."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = Mock()

        chain = EngineeringRAGChain(mock_vector_store)
        chain.chat_history = [
            {
                "question": "What is SIMP?",
                "answer": "SIMP is a topology optimization method.",
            },
            {
                "question": "How does it work?",
                "answer": "It uses a power law approach.",
            },
        ]

        formatted = chain._format_chat_history()

        assert "User: What is SIMP?" in formatted
        assert "Assistant: SIMP is a topology optimization method." in formatted
        assert "User: How does it work?" in formatted

    @patch("src.tools.rag_chain.init_chat_model")
    def test_format_chat_history_limits_to_last_5(
        self, mock_init_llm, mock_vector_store
    ):
        """Test that chat history is limited to last 5 exchanges."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = Mock()

        chain = EngineeringRAGChain(mock_vector_store)
        # Add 10 history items
        chain.chat_history = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)
        ]

        formatted = chain._format_chat_history()

        # Should only include last 5
        assert "Q5" in formatted
        assert "Q9" in formatted
        assert "Q0" not in formatted  # First one should not be included


class TestEngineeringRAGChainHistory:
    """Test chat history management."""

    @patch("src.tools.rag_chain.init_chat_model")
    def test_history_saved_after_ask(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test that chat history is saved after asking."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        assert len(chain.chat_history) == 0

        chain.ask("Test question")

        assert len(chain.chat_history) == 1
        assert chain.chat_history[0]["question"] == "Test question"
        assert "answer" in chain.chat_history[0]

    @patch("src.tools.rag_chain.init_chat_model")
    def test_clear_history(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test clearing chat history."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        chain.ask("Question 1")
        chain.ask("Question 2")
        assert len(chain.chat_history) == 2

        chain.clear_history()

        assert len(chain.chat_history) == 0

    @patch("src.tools.rag_chain.init_chat_model")
    def test_get_history(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test getting chat history."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        chain.ask("Test question")

        history = chain.get_history()

        assert len(history) == 1
        assert isinstance(history, list)
        # Ensure it's a copy, not the original
        history.clear()
        assert len(chain.chat_history) == 1


class TestEngineeringRAGChainAskWithScores:
    """Test asking with relevance scores."""

    @patch("src.tools.rag_chain.init_chat_model")
    def test_ask_with_scores(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test asking with scores returns scores."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        result = chain.ask_with_scores("Test question")

        assert "scores" in result
        assert "docs_with_scores" in result
        assert len(result["scores"]) == 2
        mock_vector_store.similarity_search_with_score.assert_called_once()


class TestEngineeringRAGChainStreamAnswer:
    """Test streaming answer generation."""

    @patch("src.tools.rag_chain.init_chat_model")
    def test_stream_answer(self, mock_init_llm, mock_vector_store):
        """Test streaming answer generation."""
        from langchain_core.language_models.fake_chat_models import (
            FakeMessagesListChatModel,
        )
        from langchain_core.messages import AIMessage

        from src.tools.rag_chain import (
            EngineeringRAGChain,
        )

        # Use fake chat model for streaming
        fake_llm = FakeMessagesListChatModel(
            responses=[AIMessage(content="This is streamed answer")]
        )
        mock_init_llm.return_value = fake_llm

        chain = EngineeringRAGChain(mock_vector_store)
        result = list(chain.stream_answer("Test question"))

        # Result should be a list of strings (content from chunks)
        assert len(result) > 0
        # Should save to history
        assert len(chain.chat_history) == 1


@pytest.mark.unit
class TestEngineeringRAGChainEdgeCases:
    """Test edge cases and error scenarios."""

    @patch("src.tools.rag_chain.init_chat_model")
    def test_ask_with_empty_question(self, mock_init_llm, mock_vector_store, mock_llm):
        """Test asking with empty question."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        result = chain.ask("")

        assert "answer" in result

    @patch("src.tools.rag_chain.init_chat_model")
    def test_ask_with_no_relevant_docs(self, mock_init_llm, mock_llm):
        """Test asking when no relevant documents are found."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = []
        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        result = chain.ask("Test question")

        assert result["num_sources"] == 0

    @patch("src.tools.rag_chain.init_chat_model")
    def test_ask_with_very_long_question(
        self, mock_init_llm, mock_vector_store, mock_llm
    ):
        """Test asking with very long question."""
        from src.tools.rag_chain import EngineeringRAGChain

        mock_init_llm.return_value = mock_llm

        chain = EngineeringRAGChain(mock_vector_store)
        long_question = "What is " + "topology optimization " * 1000

        result = chain.ask(long_question)

        assert "answer" in result
