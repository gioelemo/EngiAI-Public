"""
Multimodal document processor for extracting content from various file types.

Handles PDFs with text, images, tables, and equations using MathpixPDFLoader.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import MathpixPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MultimodalDocumentProcessor:
    """Process various document types with multimodal content extraction."""

    def __init__(self):
        """Initialize the document processor."""
        self.supported_extensions = {".pdf"}

    def process_file(self, file_path: str | Path) -> list[Document]:
        """Process a file and extract multimodal content.

        Args:
            file_path: Path to the file to process

        Returns:
            List of Document objects with extracted content

        Raises:
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(  # noqa: TRY003
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported types: {self.supported_extensions}"
            )

        if file_path.suffix.lower() == ".pdf":
            return self.process_pdf(str(file_path))

        raise ValueError(f"Handler not implemented for {file_path.suffix}")  # noqa: TRY003

    def process_pdf(self, file_path: str) -> list[Document]:
        """Process PDF with text using MathpixPDFLoader.

        MathpixPDFLoader returns a single Document with all pages combined.

        Args:
            file_path: Path to the PDF file

        Returns:
            List containing single Document with entire PDF content
        """
        logger.info(f"Processing PDF: {file_path}")

        # Use MathpixPDFLoader - returns single document with all content
        try:
            logger.debug("Using MathpixPDFLoader...")
            loader = MathpixPDFLoader(file_path)
            docs = loader.load()  # Returns [1 Document with all pages combined]

            # Update metadata
            for doc in docs:
                doc.metadata.update(
                    {
                        "source": file_path,
                        "file_type": "pdf",
                    }
                )

            logger.debug(f"Mathpix extracted {len(docs)} document from PDF")
            logger.info(f"Mathpix extracted {len(docs)} document(s) from PDF")

        except Exception as e:
            logger.exception("Mathpix PDF extraction failed")
            raise ValueError(  # noqa: TRY003
                f"Failed to extract text from PDF: {e}"
            ) from e

        else:
            return docs

    def process_batch(self, file_paths: list[str | Path]) -> list[Document]:
        """Process multiple files in batch.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of all extracted documents
        """
        all_docs = []

        for file_path in file_paths:
            try:
                docs = self.process_file(file_path)
                all_docs.extend(docs)
                logger.info(f"Processed {file_path}: {len(docs)} documents")
            except Exception:
                logger.exception(f"Failed to process {file_path}")

        logger.info(f"Total documents extracted: {len(all_docs)}")
        return all_docs

    @staticmethod
    def get_supported_extensions() -> set:
        """Get the set of supported file extensions.

        Returns:
            Set of supported file extensions (e.g., {'.pdf', '.docx'})
        """
        return {".pdf"}
