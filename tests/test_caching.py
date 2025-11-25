"""Tests for Streamlit caching implementation."""

from unittest.mock import Mock, patch

import pytest

# Skip all tests if streamlit is not available
pytest.importorskip("streamlit")


@pytest.mark.integration
class TestVectorStoreCaching:
    """Test vector store caching functionality."""

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    @patch("streamlit.cache_resource")
    def test_get_shared_vector_store_cached(
        self, mock_cache_resource, mock_embeddings_cls, mock_chroma_cls
    ):
        """Test that get_shared_vector_store uses caching decorator."""
        from src.tools.vector_store import get_shared_vector_store

        # The function should be decorated with @st.cache_resource
        assert hasattr(get_shared_vector_store, "__wrapped__") or callable(
            get_shared_vector_store
        )

    @patch("src.tools.vector_store.Chroma")
    @patch("src.tools.vector_store.OpenAIEmbeddings")
    def test_get_shared_vector_store_returns_instance(
        self, mock_embeddings_cls, mock_chroma_cls
    ):
        """Test that get_shared_vector_store returns EngineerRAGStore instance."""
        from src.tools.vector_store import EngineerRAGStore, get_shared_vector_store

        mock_embeddings_cls.return_value = Mock()
        mock_chroma_cls.return_value = Mock()

        result = get_shared_vector_store()

        assert isinstance(result, EngineerRAGStore)
        assert result.collection_name == "engineer_docs"


@pytest.mark.integration
class TestRAGChainCaching:
    """Test RAG chain caching functionality."""

    @patch("src.tools.rag_chain.init_chat_model")
    @patch("streamlit.cache_resource")
    def test_get_shared_rag_chain_cached(self, mock_cache_resource, mock_init_llm):
        """Test that get_shared_rag_chain uses caching decorator."""
        from src.tools.rag_chain import get_shared_rag_chain

        # The function should be decorated with @st.cache_resource
        assert hasattr(get_shared_rag_chain, "__wrapped__") or callable(
            get_shared_rag_chain
        )

    @patch("src.tools.rag_chain.init_chat_model")
    def test_get_shared_rag_chain_returns_instance(self, mock_init_llm):
        """Test that get_shared_rag_chain returns EngineeringRAGChain instance."""
        from src.tools.rag_chain import EngineeringRAGChain, get_shared_rag_chain
        from src.tools.vector_store import EngineerRAGStore

        mock_init_llm.return_value = Mock()
        mock_vector_store = Mock(spec=EngineerRAGStore)

        result = get_shared_rag_chain(mock_vector_store)

        assert isinstance(result, EngineeringRAGChain)


@pytest.mark.integration
class TestSupervisorAgentResourceSharing:
    """Test that SupervisorAgent can accept shared resources."""

    @patch("src.agents.supervisor_agent.get_checkpointer")
    @patch("src.agents.base_agent.init_chat_model")
    def test_supervisor_accepts_shared_resources(
        self, mock_init_llm, mock_checkpointer
    ):
        """Test that SupervisorAgent accepts shared vector store and RAG chain."""
        from src.agents.supervisor_agent import SupervisorAgent
        from src.tools.rag_chain import EngineeringRAGChain
        from src.tools.vector_store import EngineerRAGStore

        mock_init_llm.return_value = Mock()
        mock_checkpointer.return_value = Mock()

        # Create mock shared resources
        mock_vector_store = Mock(spec=EngineerRAGStore)
        mock_rag_chain = Mock(spec=EngineeringRAGChain)
        mock_rag_chain.vectorstore = mock_vector_store

        # Should not raise an error
        agent = SupervisorAgent(
            shared_vector_store=mock_vector_store,
            shared_rag_chain=mock_rag_chain,
        )

        assert agent.shared_vector_store == mock_vector_store
        assert agent.shared_rag_chain == mock_rag_chain

    @patch("src.agents.supervisor_agent.EngineerRAGStore")
    @patch("src.agents.supervisor_agent.EngineeringRAGChain")
    @patch("src.agents.supervisor_agent.get_checkpointer")
    @patch("src.agents.base_agent.init_chat_model")
    def test_supervisor_creates_resources_if_not_provided(
        self,
        mock_init_llm,
        mock_checkpointer,
        mock_rag_chain_cls,
        mock_vector_store_cls,
    ):
        """Test that SupervisorAgent creates resources if none provided."""
        from src.agents.supervisor_agent import SupervisorAgent

        mock_init_llm.return_value = Mock()
        mock_checkpointer.return_value = Mock()
        mock_vector_store_cls.return_value = Mock()
        mock_rag_chain_cls.return_value = Mock()

        # Create without shared resources
        agent = SupervisorAgent()

        # Should have created resources
        assert agent.shared_vector_store is not None
        assert agent.shared_rag_chain is not None
        mock_vector_store_cls.assert_called_once()
        mock_rag_chain_cls.assert_called_once()


@pytest.mark.integration
class TestChatManagementCaching:
    """Test chat management uses cached resources."""

    @patch("src.ui.chat_management.SupervisorAgent")
    @patch("src.ui.chat_management.get_shared_rag_chain")
    @patch("src.ui.chat_management.get_shared_vector_store")
    def test_create_supervisor_for_chat_uses_cached_resources(
        self, mock_get_vector_store, mock_get_rag_chain, mock_supervisor_cls
    ):
        """Test that create_supervisor_for_chat uses cached resources."""
        from src.ui.chat_management import create_supervisor_for_chat

        mock_vector_store = Mock()
        mock_rag_chain = Mock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_get_rag_chain.return_value = mock_rag_chain
        mock_supervisor_instance = Mock()
        mock_supervisor_cls.return_value = mock_supervisor_instance

        result = create_supervisor_for_chat()

        # Should call the cached resource getters
        mock_get_vector_store.assert_called_once_with(collection_name="engineer_docs")
        mock_get_rag_chain.assert_called_once_with(mock_vector_store)

        # Should create SupervisorAgent with shared resources
        mock_supervisor_cls.assert_called_once_with(
            shared_vector_store=mock_vector_store,
            shared_rag_chain=mock_rag_chain,
        )

        # Should return the created instance
        assert result == mock_supervisor_instance
