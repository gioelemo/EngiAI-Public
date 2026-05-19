"""Tests for the RAG tools factory (create_rag_tools).

Verifies that:
1. create_rag_tools() produces the correct tools from an MMOREClient
2. read_only mode returns only search and list tools
"""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_mmore_client():
    """Mock MMORE client for testing."""
    mock = Mock()
    mock.health_check.return_value = True
    mock.retrieve.return_value = [
        Document(
            page_content="Topology optimization content",
            metadata={"source": "eng_paper.pdf", "chunk_id": "1", "score": 0.92},
        )
    ]
    mock.upload_file.return_value = {"status": "success", "fileId": "test_id"}
    mock.delete_file.return_value = {"status": "success"}
    mock.list_files.return_value = ["eng_paper.pdf", "design_guide.pdf"]
    return mock


# ============================================================================
# RAG TOOLS FACTORY TESTS
# ============================================================================


class TestCreateRagTools:
    """Test the create_rag_tools factory function."""

    @pytest.mark.unit
    def test_creates_all_five_tools(self, mock_mmore_client):
        """Test that create_rag_tools returns exactly 5 tools."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)

        assert len(tools) == 5

    @pytest.mark.unit
    def test_tool_names(self, mock_mmore_client):
        """Test that all expected tool names are present."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)
        tool_names = {t.name for t in tools}

        assert tool_names == {
            "search_documents",
            "add_document",
            "add_url_to_knowledge_base",
            "list_documents",
            "delete_document",
        }

    @pytest.mark.unit
    def test_read_only_returns_two_tools(self, mock_mmore_client):
        """Test that read_only=True returns only search and list tools."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client, read_only=True)

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"search_documents", "list_documents"}

    @pytest.mark.unit
    def test_tools_have_descriptions(self, mock_mmore_client):
        """Test that all tools have non-empty descriptions."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)

        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"
            assert len(tool.description) > 10

    @pytest.mark.unit
    def test_search_tool_calls_retrieve(self, mock_mmore_client):
        """Test that search_documents calls mmore_client.retrieve."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)
        search_tool = next(t for t in tools if t.name == "search_documents")

        result = search_tool.invoke({"query": "topology optimization"})

        mock_mmore_client.retrieve.assert_called_once_with(
            query="topology optimization", max_matches=5, min_similarity=0.3
        )
        assert "eng_paper.pdf" in result
        assert "0.92" in result

    @pytest.mark.unit
    def test_search_tool_no_results(self, mock_mmore_client):
        """Test search_documents when no documents are found."""
        from src.tools.rag_tools import create_rag_tools

        mock_mmore_client.retrieve.return_value = []
        tools = create_rag_tools(mock_mmore_client)
        search_tool = next(t for t in tools if t.name == "search_documents")

        result = search_tool.invoke({"query": "nonexistent topic"})

        assert "No relevant documents found" in result

    @pytest.mark.unit
    def test_search_tool_error_handling(self, mock_mmore_client):
        """Test search_documents handles exceptions gracefully."""
        from src.tools.rag_tools import create_rag_tools

        mock_mmore_client.retrieve.side_effect = Exception("Connection refused")
        tools = create_rag_tools(mock_mmore_client)
        search_tool = next(t for t in tools if t.name == "search_documents")

        result = search_tool.invoke({"query": "test"})

        assert "Error searching documents" in result

    @pytest.mark.unit
    def test_list_documents_tool(self, mock_mmore_client):
        """Test list_documents returns file list."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)
        list_tool = next(t for t in tools if t.name == "list_documents")

        result = list_tool.invoke({})

        assert "2 document" in result
        assert "eng_paper.pdf" in result
        assert "design_guide.pdf" in result

    @pytest.mark.unit
    def test_delete_document_tool(self, mock_mmore_client):
        """Test delete_document calls mmore_client.delete_file."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)
        delete_tool = next(t for t in tools if t.name == "delete_document")

        result = delete_tool.invoke({"file_id": "eng_paper.pdf"})

        mock_mmore_client.delete_file.assert_called_once_with("eng_paper.pdf")
        assert "Deleted" in result
