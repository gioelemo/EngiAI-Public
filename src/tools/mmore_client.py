"""
MMORE API Client for multimodal document retrieval.

Provides interface to mmore service for advanced RAG capabilities.
"""

import logging
import os
from pathlib import Path
from typing import Any

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# HTTP status codes
HTTP_OK = 200


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
        base_url_str: str = (
            base_url
            if base_url
            else (os.getenv("MMORE_RAG_URL") or "http://localhost:8000")
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

    def upload_file(self, file_path: str, file_id: str | None = None) -> dict[str, Any]:
        """Upload a file to mmore for indexing.

        Args:
            file_path: Path to the file to upload
            file_id: Optional unique identifier (defaults to filename)

        Returns:
            Response from mmore API

        Raises:
            requests.HTTPError: If upload fails
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise MMOREFileNotFoundError(file_path)

        file_id = file_id or file_path_obj.stem

        with file_path_obj.open("rb") as f:
            files = {"file": (file_path_obj.name, f)}
            data = {"fileId": file_id}

            response = requests.post(
                f"{self.base_url}/v1/files",
                files=files,
                data=data,
                timeout=300,  # 5 min timeout for large files
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

    def upload_bulk(
        self, file_paths: list[str], file_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """Upload multiple files at once.

        Args:
            file_paths: List of file paths to upload
            file_ids: Optional list of IDs (must match length of file_paths)

        Returns:
            Response from mmore API

        Raises:
            requests.HTTPError: If upload fails
            ValueError: If file_ids length doesn't match file_paths
        """
        if file_ids and len(file_ids) != len(file_paths):
            raise MMOREValueError("file_ids_length_mismatch")

        if not file_ids:
            file_ids = [Path(fp).stem for fp in file_paths]

        # Use proper context managers for file handles
        file_handles = []
        files_data = []
        try:
            for fp in file_paths:
                file_path_obj = Path(fp)
                if not file_path_obj.exists():
                    logger.warning(f"Skipping non-existent file: {fp}")
                    continue
                file_handle = file_path_obj.open("rb")
                file_handles.append(file_handle)
                files_data.append(("files", (file_path_obj.name, file_handle)))

            data = {"listIds": file_ids}
            response = requests.post(
                f"{self.base_url}/v1/files/bulk",
                files=files_data,
                data=data,
                timeout=600,  # 10 min for bulk upload
            )
            response.raise_for_status()

            logger.info(f"Bulk uploaded {len(file_paths)} files to mmore")
            return response.json()
        finally:
            # Close all file handles
            for fh in file_handles:
                fh.close()
