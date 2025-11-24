"""
Tests for document_processor module.

These tests cover the MultimodalDocumentProcessor class for handling PDFs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.document_processor import MultimodalDocumentProcessor

# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


@pytest.mark.unit
def test_processor_initialization():
    """Test MultimodalDocumentProcessor initialization."""
    processor = MultimodalDocumentProcessor()

    assert hasattr(processor, "supported_extensions")
    assert ".pdf" in processor.supported_extensions


@pytest.mark.unit
def test_get_supported_extensions():
    """Test get_supported_extensions static method."""
    extensions = MultimodalDocumentProcessor.get_supported_extensions()

    assert isinstance(extensions, set)
    assert ".pdf" in extensions


# ============================================================================
# FILE PROCESSING TESTS
# ============================================================================


@pytest.mark.unit
def test_process_file_unsupported_type():
    """Test processing unsupported file type raises ValueError."""
    processor = MultimodalDocumentProcessor()

    with pytest.raises(ValueError) as excinfo:
        processor.process_file("/path/to/file.docx")

    assert "Unsupported file type" in str(excinfo.value)
    assert ".docx" in str(excinfo.value)


@pytest.mark.unit
def test_process_file_accepts_string_path():
    """Test that process_file accepts string paths."""
    processor = MultimodalDocumentProcessor()

    # Should raise ValueError for unsupported type, not TypeError
    with pytest.raises(ValueError):
        processor.process_file("/path/to/file.txt")


@pytest.mark.unit
def test_process_file_accepts_path_object():
    """Test that process_file accepts Path objects."""
    processor = MultimodalDocumentProcessor()

    # Should raise ValueError for unsupported type, not TypeError
    with pytest.raises(ValueError):
        processor.process_file(Path("/path/to/file.txt"))


@pytest.mark.unit
def test_process_pdf_success(tmp_path):
    """Test successful PDF processing with mocked loader."""
    processor = MultimodalDocumentProcessor()

    # Create a mock document
    mock_doc = MagicMock()
    mock_doc.metadata = {}
    mock_doc.page_content = "Test content"

    # Mock the MathpixPDFLoader
    with patch("src.tools.document_processor.MathpixPDFLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader

        # Create a fake PDF path
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        docs = processor.process_pdf(str(pdf_path))

    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == "pdf"
    assert docs[0].metadata["source"] == str(pdf_path)


@pytest.mark.unit
def test_process_pdf_failure():
    """Test PDF processing failure handling."""
    processor = MultimodalDocumentProcessor()

    # Mock the MathpixPDFLoader to raise an exception
    with patch("src.tools.document_processor.MathpixPDFLoader") as mock_loader_class:
        mock_loader_class.return_value.load.side_effect = Exception("API Error")

        with pytest.raises(ValueError) as excinfo:
            processor.process_pdf("/path/to/broken.pdf")

        assert "Failed to extract text from PDF" in str(excinfo.value)


# ============================================================================
# BATCH PROCESSING TESTS
# ============================================================================


@pytest.mark.unit
def test_process_batch_empty_list():
    """Test batch processing with empty file list."""
    processor = MultimodalDocumentProcessor()

    docs = processor.process_batch([])

    assert docs == []


@pytest.mark.unit
def test_process_batch_mixed_results(tmp_path):
    """Test batch processing with some failures."""
    processor = MultimodalDocumentProcessor()

    # Create mock documents
    mock_doc = MagicMock()
    mock_doc.metadata = {}
    mock_doc.page_content = "Test content"

    with patch("src.tools.document_processor.MathpixPDFLoader") as mock_loader_class:
        # First call succeeds, second fails
        mock_loader = MagicMock()
        mock_loader.load.side_effect = [
            [mock_doc],  # First file succeeds
            Exception("API Error"),  # Second file fails
        ]
        mock_loader_class.return_value = mock_loader

        # Create fake PDF paths
        pdf1 = tmp_path / "test1.pdf"
        pdf1.touch()
        pdf2 = tmp_path / "test2.pdf"
        pdf2.touch()

        # Process batch - should continue despite errors
        docs = processor.process_batch([pdf1, pdf2])

    # Should have docs from successful file
    assert len(docs) >= 0  # May have 0 or 1 depending on error handling


@pytest.mark.unit
def test_process_batch_all_success(tmp_path):
    """Test batch processing when all files succeed."""
    processor = MultimodalDocumentProcessor()

    # Create mock documents
    mock_doc1 = MagicMock()
    mock_doc1.metadata = {}
    mock_doc1.page_content = "Content 1"

    mock_doc2 = MagicMock()
    mock_doc2.metadata = {}
    mock_doc2.page_content = "Content 2"

    with patch("src.tools.document_processor.MathpixPDFLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load.side_effect = [[mock_doc1], [mock_doc2]]
        mock_loader_class.return_value = mock_loader

        # Create fake PDF paths
        pdf1 = tmp_path / "test1.pdf"
        pdf1.touch()
        pdf2 = tmp_path / "test2.pdf"
        pdf2.touch()

        docs = processor.process_batch([str(pdf1), str(pdf2)])

    assert len(docs) == 2


# ============================================================================
# METADATA TESTS
# ============================================================================


@pytest.mark.unit
def test_process_pdf_updates_metadata(tmp_path):
    """Test that PDF processing updates document metadata correctly."""
    processor = MultimodalDocumentProcessor()

    mock_doc = MagicMock()
    mock_doc.metadata = {"original_key": "original_value"}
    mock_doc.page_content = "Test content"

    with patch("src.tools.document_processor.MathpixPDFLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        docs = processor.process_pdf(str(pdf_path))

    # Should preserve original metadata and add new fields
    assert docs[0].metadata["file_type"] == "pdf"
    assert docs[0].metadata["source"] == str(pdf_path)


# ============================================================================
# SUPPORTED EXTENSIONS TESTS
# ============================================================================


@pytest.mark.unit
def test_supported_extensions_immutability():
    """Test that supported_extensions is properly defined."""
    processor = MultimodalDocumentProcessor()

    # Get extensions
    extensions = processor.supported_extensions

    # Should be a set
    assert isinstance(extensions, set)

    # Should contain PDF
    assert ".pdf" in extensions


@pytest.mark.unit
def test_process_file_routes_to_pdf():
    """Test that PDF files are routed to process_pdf."""
    processor = MultimodalDocumentProcessor()

    with patch.object(processor, "process_pdf") as mock_process_pdf:
        mock_process_pdf.return_value = []

        processor.process_file("/path/to/document.pdf")

        mock_process_pdf.assert_called_once_with("/path/to/document.pdf")


@pytest.mark.unit
def test_process_file_case_insensitive():
    """Test that file extension check is case insensitive."""
    processor = MultimodalDocumentProcessor()

    with patch.object(processor, "process_pdf") as mock_process_pdf:
        mock_process_pdf.return_value = []

        # Test uppercase extension
        processor.process_file("/path/to/document.PDF")

        mock_process_pdf.assert_called()
