"""Tests for EngineerRAGStore (ChromaDB vector store)."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_chroma():
    """Mock Chroma vector store."""
    mock = Mock()
    mock.add_documents.return_value = ["id1", "id2", "id3"]
    mock.similarity_search.return_value = [
        Document(
            page_content="Test content",
            metadata={"source": "test.pdf", "page": 1},
        )
    ]
    mock.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="Test content",
                metadata={"source": "test.pdf", "page": 1},
            ),
            0.85,
        )
    ]
    mock.delete.return_value = None
    mock.delete_collection.return_value = None
    mock._collection = Mock(count=Mock(return_value=10))

    # Create proper mock for collections
    mock_collection = Mock()
    mock_collection.name = "test_col"
    mock._client = Mock(list_collections=Mock(return_value=[mock_collection]))
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings."""
    return Mock()


class TestEngineerRAGStoreInitialization:
    """Test vector store initialization."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_initialization_default_params(self, mock_embeddings_cls, mock_chroma_cls):
        """Test initialization with default parameters."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = Mock()

        store = EngineerRAGStore()

        assert store.collection_name == "engineer_docs"
        assert "chroma_db" in store.persist_directory
        mock_embeddings_cls.assert_called_once_with(model="text-embedding-3-small")
        mock_chroma_cls.assert_called_once()

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_initialization_custom_params(self, mock_embeddings_cls, mock_chroma_cls):
        """Test initialization with custom parameters."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = Mock()

        store = EngineerRAGStore(
            collection_name="custom_docs",
            persist_directory="/tmp/test_chroma",
            embedding_model="text-embedding-3-large",
        )

        assert store.collection_name == "custom_docs"
        assert store.persist_directory == "/tmp/test_chroma"
        mock_embeddings_cls.assert_called_once_with(model="text-embedding-3-large")

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    @patch("src.tools.vector_store.Path")
    def test_creates_persist_directory(
        self, mock_path, mock_embeddings_cls, mock_chroma_cls
    ):
        """Test that persist directory is created if it doesn't exist."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = Mock()
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance

        EngineerRAGStore(persist_directory="/tmp/new_dir")

        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestEngineerRAGStoreAddDocuments:
    """Test adding documents to the vector store."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_add_documents_success(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test successful document addition."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        docs = [
            Document(page_content="Test 1", metadata={"source": "test1.pdf"}),
            Document(page_content="Test 2", metadata={"source": "test2.pdf"}),
        ]

        ids = store.add_documents(docs)

        assert len(ids) == 3  # Mock returns 3 IDs
        mock_chroma.add_documents.assert_called_once_with(docs)

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_add_documents_with_metadata(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test adding documents with additional metadata."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        docs = [Document(page_content="Test", metadata={"source": "test.pdf"})]
        additional_metadata = {"category": "research", "year": 2024}

        store.add_documents(docs, metadata=additional_metadata)

        # Verify metadata was added to documents
        added_docs = mock_chroma.add_documents.call_args[0][0]
        assert added_docs[0].metadata["category"] == "research"
        assert added_docs[0].metadata["year"] == 2024

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_add_documents_empty_list(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test adding empty document list."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        ids = store.add_documents([])

        assert ids == []
        mock_chroma.add_documents.assert_not_called()


class TestEngineerRAGStoreSimilaritySearch:
    """Test similarity search functionality."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_similarity_search_basic(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test basic similarity search."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        results = store.similarity_search("test query", k=5)

        assert len(results) > 0
        mock_chroma.similarity_search.assert_called_once_with(
            "test query", k=5, filter=None
        )

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_similarity_search_with_filter(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test similarity search with metadata filter."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        filter_dict = {"source": "specific.pdf"}
        store.similarity_search("test query", k=3, filter_dict=filter_dict)

        mock_chroma.similarity_search.assert_called_once_with(
            "test query", k=3, filter=filter_dict
        )

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_similarity_search_with_score_threshold(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test similarity search with score threshold."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma.similarity_search_with_score.return_value = [
            (Document(page_content="High score", metadata={}), 0.95),
            (Document(page_content="Low score", metadata={}), 0.50),
        ]
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        results = store.similarity_search("query", k=5, score_threshold=0.8)

        # Should only return documents with score >= 0.8
        assert len(results) == 1
        assert results[0].page_content == "High score"


class TestEngineerRAGStoreSimilaritySearchWithScore:
    """Test similarity search with scores."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_similarity_search_with_score(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test similarity search returning scores."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        results = store.similarity_search_with_score("test query", k=5)

        assert len(results) > 0
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2  # (document, score)
        mock_chroma.similarity_search_with_score.assert_called_once()


class TestEngineerRAGStoreRetriever:
    """Test retriever interface."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_get_retriever_default(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test getting retriever with default parameters."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma.as_retriever.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        retriever = store.get_retriever()

        assert retriever is not None
        mock_chroma.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_get_retriever_custom_kwargs(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test getting retriever with custom search kwargs."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma.as_retriever.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        custom_kwargs = {"k": 10, "score_threshold": 0.8}
        store.get_retriever(search_kwargs=custom_kwargs)

        mock_chroma.as_retriever.assert_called_once_with(search_kwargs=custom_kwargs)


class TestEngineerRAGStoreDeleteOperations:
    """Test document deletion operations."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_delete_documents_by_ids(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test deleting documents by IDs."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        ids_to_delete = ["id1", "id2", "id3"]
        store.delete_documents(ids_to_delete)

        mock_chroma.delete.assert_called_once_with(ids=ids_to_delete)

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_delete_by_metadata(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test deleting documents by metadata filter."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        docs_to_delete = [
            Document(page_content="test", metadata={"id": "doc1", "source": "old.pdf"}),
            Document(page_content="test", metadata={"id": "doc2", "source": "old.pdf"}),
        ]
        mock_chroma.similarity_search.return_value = docs_to_delete
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        filter_dict = {"source": "old.pdf"}
        store.delete_by_metadata(filter_dict)

        mock_chroma.similarity_search.assert_called_once()
        mock_chroma.delete.assert_called_once_with(ids=["doc1", "doc2"])

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_clear_collection(self, mock_embeddings_cls, mock_chroma_cls, mock_chroma):
        """Test clearing entire collection."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        store.clear_collection()

        mock_chroma.delete_collection.assert_called_once()
        # Should also reinitialize
        assert mock_chroma_cls.call_count == 2  # Initial + after clear


class TestEngineerRAGStoreUtilities:
    """Test utility methods."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_get_collection_count(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test getting collection count."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        count = store.get_collection_count()

        assert count == 10

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_list_collections(self, mock_embeddings_cls, mock_chroma_cls, mock_chroma):
        """Test listing all collections."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        collections = store.list_collections()

        assert "test_col" in collections


@pytest.mark.unit
class TestEngineerRAGStoreEdgeCases:
    """Test edge cases and error scenarios."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_get_collection_count_error(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test collection count with error."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma._collection = Mock()
        mock_chroma._collection.count.side_effect = Exception("DB error")
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        count = store.get_collection_count()

        assert count == 0  # Should return 0 on error

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_list_collections_error(
        self, mock_embeddings_cls, mock_chroma_cls, mock_chroma
    ):
        """Test list collections with error."""
        from src.tools.vector_store import EngineerRAGStore

        mock_embeddings_cls.return_value = Mock()
        mock_chroma._client = Mock()
        mock_chroma._client.list_collections.side_effect = Exception("DB error")
        mock_chroma_cls.return_value = mock_chroma

        store = EngineerRAGStore()
        collections = store.list_collections()

        assert collections == []  # Should return empty list on error
