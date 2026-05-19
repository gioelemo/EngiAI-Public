"""
MMORE API Client for multimodal document retrieval.

Provides interface to mmore service for advanced RAG capabilities.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# HTTP status codes
HTTP_OK = 200

# Progress callback - set by UI to receive progress updates
_progress_callback: Callable[[str, str], None] | None = None


def set_mmore_progress_callback(callback: Callable[[str, str], None] | None) -> None:
    """
    Set the progress callback for MMORE operations.

    Args:
        callback: Callback function(step: str, message: str) or None to clear
    """
    global _progress_callback  # noqa: PLW0603
    _progress_callback = callback


def get_mmore_progress_callback() -> Callable[[str, str], None] | None:
    """Get the current MMORE progress callback."""
    return _progress_callback


def _report_progress(step: str, message: str) -> None:
    """Report progress if callback is set."""
    callback = get_mmore_progress_callback()
    if callback:
        callback(step, message)


class MMOREFileNotFoundError(FileNotFoundError):
    """Exception raised when a file is not found for MMORE operations."""


class MMOREValueError(ValueError):
    """Exception raised for invalid MMORE operation arguments."""


class MMOREClient:
    """Client for interacting with MMORE RAG service API."""

    def __init__(self, base_url: str | None = None):
        """Initialize MMORE client.

        Args:
            base_url: Base URL of mmore service (defaults to MMORE_RAG_URL env var)
        """
        base_url_str: str = base_url or (
            os.getenv("MMORE_RAG_URL") or "http://localhost:8000"
        )
        self.base_url = base_url_str.rstrip("/")
        logger.info(f"MMORE client initialized with URL: {self.base_url}")

    def health_check(self) -> bool:
        """Check if mmore service is healthy.

        Returns:
            True if service is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
        except requests.RequestException as e:
            logger.warning(f"MMORE health check failed: {e}")
            return False
        else:
            return response.status_code == HTTP_OK

    def upload_file(
        self,
        file_path: str,
        file_id: str | None = None,
        original_name: str | None = None,
    ) -> dict[str, Any]:
        """Upload a file to mmore for indexing.

        Args:
            file_path: Path to the file to upload
            file_id: Optional unique identifier (defaults to filename)
            original_name: Optional original file name for display (uses file_path name if not provided)

        Returns:
            Response from mmore API

        Raises:
            requests.HTTPError: If upload fails
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise MMOREFileNotFoundError(file_path)

        file_id = file_id or file_path_obj.stem
        display_name = original_name or file_path_obj.name

        # Report progress: preparing
        file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
        _report_progress(
            "prepare", f"Preparing to upload '{display_name}' ({file_size_mb:.1f} MB)"
        )

        with file_path_obj.open("rb") as f:
            files = {"file": (file_path_obj.name, f)}
            data = {"fileId": file_id}

            # Report progress: uploading
            _report_progress(
                "upload", f"Uploading '{display_name}' to MMORE service..."
            )

            # Start the upload - MMORE will process during this request
            response = requests.post(
                f"{self.base_url}/v1/files",
                files=files,
                data=data,
                timeout=300,  # 5 min timeout for large files
                # Note: MMORE processes the file during this request, so when it returns, processing is complete
            )

            response.raise_for_status()

        logger.info(f"Uploaded file {file_id} to mmore")

        return response.json()

    def retrieve(
        self,
        query: str,
        file_ids: list[str] | None = None,
        max_matches: int = 5,
        min_similarity: float = 0.0,
    ) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            file_ids: Optional list of file IDs to restrict search to
            max_matches: Maximum number of results to return
            min_similarity: Minimum similarity score threshold

        Returns:
            List of LangChain Document objects with retrieved content

        Raises:
            requests.HTTPError: If retrieval fails
        """
        # Empty-RAG evaluation mode: tools are available but index is empty.
        if os.getenv("MMORE_EMPTY_RAG", "false").lower() == "true":
            logger.debug("MMORE_EMPTY_RAG=true: returning empty results")
            return []

        payload = {
            "query": query,
            "fileIds": file_ids or [],
            "maxMatches": max_matches,
            "minSimilarity": min_similarity,
        }

        response = requests.post(
            f"{self.base_url}/v1/retrieve",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        # MMORE returns a list directly, not a dict with 'docs' key
        docs_list = data if isinstance(data, list) else data.get("docs", [])
        logger.debug(f"Retrieved {len(docs_list)} documents from mmore")

        # Convert to LangChain documents
        documents = []
        for doc_info in docs_list:
            doc = Document(
                page_content=doc_info.get("content", ""),
                metadata={
                    "source": doc_info.get("fileId", "unknown"),
                    "chunk_id": doc_info.get("chunkId"),
                    "score": doc_info.get("similarity", doc_info.get("score", 0.0)),
                    **doc_info.get("metadata", {}),
                },
            )
            documents.append(doc)

        return documents

    def delete_file(self, file_id: str) -> dict[str, Any]:
        """Delete a file from mmore index.

        Args:
            file_id: ID of the file to delete

        Returns:
            Response from mmore API

        Raises:
            requests.HTTPError: If deletion fails
        """
        response = requests.delete(
            f"{self.base_url}/v1/files/{file_id}",
            timeout=30,
        )
        response.raise_for_status()

        logger.info(f"Deleted file {file_id} from mmore")
        return response.json()

    def list_files(self, collection_name: str = "my_docs") -> list[str]:
        """List all files stored in the mmore index.

        Args:
            collection_name: Name of the Milvus collection (defaults to 'my_docs')

        Returns:
            List of file IDs

        Raises:
            requests.HTTPError: If listing fails
        """
        # Empty-RAG evaluation mode: tools are available but index is empty.
        if os.getenv("MMORE_EMPTY_RAG", "false").lower() == "true":
            logger.debug("MMORE_EMPTY_RAG=true: returning empty file list")
            return []

        response = requests.get(
            f"{self.base_url}/list_files",
            params={"collection_name": collection_name},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        # API returns list of {"id": ..., "filename": ...} objects, extract IDs
        file_ids = [item["id"] for item in data] if data else []
        logger.info(f"Listed {len(file_ids)} files from mmore")
        return file_ids
