"""
Tests for arxiv_tools module.

These tests cover ArXiv paper search, retrieval, and download functionality.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.tools.arxiv_tools import (
    MAX_AUTHORS_DISPLAY,
    SUMMARY_PREVIEW_LENGTH,
    create_arxiv_tools,
    get_arxiv_paper,
    search_arxiv,
)

# ============================================================================
# HELPER FIXTURES
# ============================================================================


@pytest.fixture
def mock_paper():
    """Create a mock ArXiv paper object."""
    paper = MagicMock()
    paper.title = "Attention Is All You Need"
    paper.entry_id = "http://arxiv.org/abs/1706.03762"
    paper.authors = [
        MagicMock(name="Ashish Vaswani"),
        MagicMock(name="Noam Shazeer"),
        MagicMock(name="Niki Parmar"),
        MagicMock(name="Jakob Uszkoreit"),
    ]
    # Set the name attribute properly for MagicMock
    paper.authors[0].name = "Ashish Vaswani"
    paper.authors[1].name = "Noam Shazeer"
    paper.authors[2].name = "Niki Parmar"
    paper.authors[3].name = "Jakob Uszkoreit"
    paper.published = datetime(2017, 6, 12)
    paper.updated = datetime(2017, 12, 6)
    paper.summary = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder."
    paper.pdf_url = "http://arxiv.org/pdf/1706.03762"
    paper.categories = ["cs.CL", "cs.LG"]
    paper.doi = "10.5555/3295222.3295349"
    return paper


# ============================================================================
# SEARCH ARXIV TESTS
# ============================================================================


@pytest.mark.unit
def test_search_arxiv_success(mock_paper):
    """Test successful ArXiv search."""
    mock_search = MagicMock()
    mock_search.results.return_value = iter([mock_paper])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search):
        result = search_arxiv.invoke(
            {"query": "attention mechanisms", "max_results": 5}
        )

    assert "Attention Is All You Need" in result
    assert "1706.03762" in result
    assert "Found 1 papers" in result


@pytest.mark.unit
def test_search_arxiv_no_results():
    """Test search with no results."""
    mock_search = MagicMock()
    mock_search.results.return_value = iter([])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search):
        result = search_arxiv.invoke({"query": "nonexistent topic xyz123"})

    assert "No papers found" in result


@pytest.mark.unit
def test_search_arxiv_error():
    """Test search with API error."""
    with patch(
        "src.tools.arxiv_tools.arxiv.Search", side_effect=Exception("API Error")
    ):
        result = search_arxiv.invoke({"query": "test query"})

    assert "Error searching ArXiv" in result


@pytest.mark.unit
def test_search_arxiv_truncates_authors(mock_paper):
    """Test that author list is truncated after MAX_AUTHORS_DISPLAY."""
    # Add more authors
    extra_authors = [MagicMock() for _ in range(5)]
    for i, author in enumerate(extra_authors):
        author.name = f"Author {i}"
    mock_paper.authors = extra_authors

    mock_search = MagicMock()
    mock_search.results.return_value = iter([mock_paper])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search):
        result = search_arxiv.invoke({"query": "test"})

    assert "et al." in result
    assert f"{len(extra_authors)} authors" in result


@pytest.mark.unit
def test_search_arxiv_truncates_summary(mock_paper):
    """Test that long summaries are truncated."""
    mock_paper.summary = "x" * 500  # Longer than SUMMARY_PREVIEW_LENGTH

    mock_search = MagicMock()
    mock_search.results.return_value = iter([mock_paper])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search):
        result = search_arxiv.invoke({"query": "test"})

    assert "..." in result


@pytest.mark.unit
def test_search_arxiv_default_max_results():
    """Test search uses default max_results of 5."""
    mock_search = MagicMock()
    mock_search.results.return_value = iter([])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search) as mock:
        search_arxiv.invoke({"query": "test"})

    # Check that max_results was set to 5 (default)
    mock.assert_called_once()
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs.get("max_results") == 5


# ============================================================================
# GET ARXIV PAPER TESTS
# ============================================================================


@pytest.mark.unit
def test_get_arxiv_paper_success(mock_paper):
    """Test successful paper retrieval."""
    mock_search = MagicMock()
    mock_search.results.return_value = iter([mock_paper])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search):
        result = get_arxiv_paper.invoke({"arxiv_id": "1706.03762"})

    assert "Attention Is All You Need" in result
    assert "Authors:" in result
    assert "Abstract:" in result
    assert "PDF:" in result


@pytest.mark.unit
def test_get_arxiv_paper_with_prefix():
    """Test paper retrieval with arxiv: prefix."""
    mock_paper = MagicMock()
    mock_paper.title = "Test Paper"
    mock_paper.entry_id = "http://arxiv.org/abs/1234.5678"
    mock_paper.authors = [MagicMock(name="Test Author")]
    mock_paper.authors[0].name = "Test Author"
    mock_paper.published = datetime(2020, 1, 1)
    mock_paper.updated = datetime(2020, 1, 1)
    mock_paper.summary = "Test summary"
    mock_paper.pdf_url = "http://arxiv.org/pdf/1234.5678"
    mock_paper.categories = ["cs.AI"]
    mock_paper.doi = None

    mock_search = MagicMock()
    mock_search.results.return_value = iter([mock_paper])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search) as mock:
        result = get_arxiv_paper.invoke({"arxiv_id": "arxiv:1234.5678"})

    # Should strip the prefix
    mock.assert_called_once()
    assert "1234.5678" in mock.call_args.kwargs.get("id_list", [""])[0]
    assert "Test Paper" in result


@pytest.mark.unit
def test_get_arxiv_paper_not_found():
    """Test paper retrieval when paper not found."""
    mock_search = MagicMock()
    mock_search.results.return_value = iter([])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search):
        result = get_arxiv_paper.invoke({"arxiv_id": "0000.00000"})

    assert "not found" in result


@pytest.mark.unit
def test_get_arxiv_paper_error():
    """Test paper retrieval with API error."""
    with patch(
        "src.tools.arxiv_tools.arxiv.Search", side_effect=Exception("API Error")
    ):
        result = get_arxiv_paper.invoke({"arxiv_id": "1234.5678"})

    assert "Error fetching paper" in result


@pytest.mark.unit
def test_get_arxiv_paper_no_doi(mock_paper):
    """Test paper display when DOI is None."""
    mock_paper.doi = None

    mock_search = MagicMock()
    mock_search.results.return_value = iter([mock_paper])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search):
        result = get_arxiv_paper.invoke({"arxiv_id": "1706.03762"})

    assert "DOI:" in result
    assert "N/A" in result


# ============================================================================
# CREATE ARXIV TOOLS TESTS
# ============================================================================


@pytest.mark.unit
def test_create_arxiv_tools_without_dependencies():
    """Test that create_arxiv_tools returns basic tools without dependencies."""
    tools = create_arxiv_tools()

    assert len(tools) == 2
    tool_names = [tool.name for tool in tools]
    assert "search_arxiv" in tool_names
    assert "get_arxiv_paper" in tool_names


@pytest.mark.unit
def test_create_arxiv_tools_with_dependencies():
    """Test that create_arxiv_tools returns all tools with MMORE dependencies."""
    mock_mmore_client = MagicMock()
    mock_db_manager = MagicMock()

    tools = create_arxiv_tools(
        mmore_client=mock_mmore_client, db_manager=mock_db_manager
    )

    assert len(tools) == 5
    tool_names = [tool.name for tool in tools]
    assert "search_arxiv" in tool_names
    assert "get_arxiv_paper" in tool_names
    assert "download_and_analyze_paper" in tool_names
    assert "ask_about_papers" in tool_names
    assert "list_analyzed_papers" in tool_names


@pytest.mark.unit
def test_create_arxiv_tools_are_callable():
    """Test that all returned tools are callable."""
    tools = create_arxiv_tools()

    for tool in tools:
        assert hasattr(tool, "invoke")
        assert callable(tool.invoke)


# ============================================================================
# CONSTANTS TESTS
# ============================================================================


@pytest.mark.unit
def test_max_authors_display_constant():
    """Test MAX_AUTHORS_DISPLAY constant."""
    assert MAX_AUTHORS_DISPLAY == 3


@pytest.mark.unit
def test_summary_preview_length_constant():
    """Test SUMMARY_PREVIEW_LENGTH constant."""
    assert SUMMARY_PREVIEW_LENGTH == 300


# ============================================================================
# EDGE CASES
# ============================================================================


@pytest.mark.unit
def test_search_arxiv_multiple_results():
    """Test search with multiple results."""
    papers = []
    for i in range(3):
        paper = MagicMock()
        paper.title = f"Paper {i}"
        paper.entry_id = f"http://arxiv.org/abs/2020.{i:05d}"
        paper.authors = [MagicMock(name=f"Author {i}")]
        paper.authors[0].name = f"Author {i}"
        paper.published = datetime(2020, 1, i + 1)
        paper.summary = f"Summary {i}"
        paper.pdf_url = f"http://arxiv.org/pdf/2020.{i:05d}"
        papers.append(paper)

    mock_search = MagicMock()
    mock_search.results.return_value = iter(papers)

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search):
        result = search_arxiv.invoke({"query": "test", "max_results": 3})

    assert "Found 3 papers" in result
    assert "Paper 0" in result
    assert "Paper 1" in result
    assert "Paper 2" in result


@pytest.mark.unit
def test_get_arxiv_paper_strips_arxiv_capital_prefix():
    """Test that arXiv: prefix (capital X) is also handled."""
    mock_paper = MagicMock()
    mock_paper.title = "Test"
    mock_paper.entry_id = "http://arxiv.org/abs/1234.5678"
    mock_paper.authors = []
    mock_paper.published = datetime(2020, 1, 1)
    mock_paper.updated = datetime(2020, 1, 1)
    mock_paper.summary = ""
    mock_paper.pdf_url = ""
    mock_paper.categories = []
    mock_paper.doi = None

    mock_search = MagicMock()
    mock_search.results.return_value = iter([mock_paper])

    with patch("src.tools.arxiv_tools.arxiv.Search", return_value=mock_search) as mock:
        get_arxiv_paper.invoke({"arxiv_id": "arXiv:1234.5678"})

    # Should handle both lowercase and mixed case prefix
    call_args = mock.call_args.kwargs
    assert "arXiv:" not in call_args.get("id_list", ["arXiv:"])[0]
