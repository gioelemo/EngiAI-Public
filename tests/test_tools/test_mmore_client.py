"""
Tests for MMORE client.

Tests multimodal RAG service client functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from langchain_core.documents import Document

from src.tools.mmore_client import (
    MMOREClient,
    MMOREFileNotFoundError,
    get_progress_callback,
    set_progress_callback,
)


@pytest.fixture
def mmore_client():
    """Create MMORE client instance."""
    return MMOREClient(base_url="http://test-mmore:8000")


@pytest.fixture
def temp_file():
    """Create temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test content for MMORE upload")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestMMOREClientInitialization:
    """Test MMORE client initialization."""

    def test_init_with_custom_url(self):
        """Test initialization with custom base URL."""
        client = MMOREClient(base_url="http://custom:9000")

        assert client.base_url == "http://custom:9000"

    def test_init_with_trailing_slash(self):
        """Test that trailing slash is stripped from URL."""
        client = MMOREClient(base_url="http://test:8000/")

        assert client.base_url == "http://test:8000"

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization from environment variable."""
        monkeypatch.setenv("MMORE_RAG_URL", "http://env-mmore:7000")

        client = MMOREClient()

        assert client.base_url == "http://env-mmore:7000"

    def test_init_default_url(self, monkeypatch):
        """Test initialization with default URL when no env var."""
        monkeypatch.delenv("MMORE_RAG_URL", raising=False)

        client = MMOREClient()

        assert client.base_url == "http://localhost:8000"


class TestMMOREHealthCheck:
    """Test MMORE health check functionality."""

    def test_health_check_success(self, mmore_client):
        """Test successful health check."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = mmore_client.health_check()

            assert result is True
            mock_get.assert_called_once_with("http://test-mmore:8000/", timeout=5)

    def test_health_check_failure_non_200(self, mmore_client):
        """Test health check with non-200 status."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            result = mmore_client.health_check()

            assert result is False

    def test_health_check_connection_error(self, mmore_client):
        """Test health check with connection error."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            result = mmore_client.health_check()

            assert result is False

    def test_health_check_timeout(self, mmore_client):
        """Test health check with timeout."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("Request timeout")

            result = mmore_client.health_check()

            assert result is False


class TestMMOREFileUpload:
    """Test MMORE file upload functionality."""

    def test_upload_file_success(self, mmore_client, temp_file):
        """Test successful file upload."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "fileId": "test-file",
                "status": "success",
            }
            mock_post.return_value = mock_response

            result = mmore_client.upload_file(str(temp_file))

            assert result["fileId"] == "test-file"
            assert result["status"] == "success"
            mock_post.assert_called_once()

    def test_upload_file_not_found(self, mmore_client):
        """Test upload with non-existent file."""
        with pytest.raises(MMOREFileNotFoundError):
            mmore_client.upload_file("/nonexistent/file.pdf")

    def test_upload_file_with_custom_id(self, mmore_client, temp_file):
        """Test upload with custom file ID."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"fileId": "custom-id"}
            mock_post.return_value = mock_response

            result = mmore_client.upload_file(str(temp_file), file_id="custom-id")

            assert result["fileId"] == "custom-id"
            # Verify fileId was sent in request data
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["data"]["fileId"] == "custom-id"

    def test_upload_file_http_error(self, mmore_client, temp_file):
        """Test upload with HTTP error response."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.HTTPError(
                "Server error"
            )
            mock_post.return_value = mock_response

            with pytest.raises(requests.HTTPError):
                mmore_client.upload_file(str(temp_file))

    def test_upload_file_with_progress_callback(self, mmore_client, temp_file):
        """Test upload with progress callback."""
        progress_calls = []

        def progress_cb(step, message):
            progress_calls.append((step, message))

        set_progress_callback(progress_cb)

        try:
            with patch("requests.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"fileId": "test"}
                mock_post.return_value = mock_response

                mmore_client.upload_file(str(temp_file))

                # Should have reported progress
                assert len(progress_calls) >= 2
                assert progress_calls[0][0] == "prepare"
                assert progress_calls[1][0] == "upload"
        finally:
            set_progress_callback(None)


class TestMMORERetrieve:
    """Test MMORE document retrieval functionality."""

    def test_retrieve_success(self, mmore_client):
        """Test successful document retrieval."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "content": "Test content",
                    "fileId": "doc1",
                    "chunkId": "chunk1",
                    "similarity": 0.95,
                    "metadata": {"page": 1},
                }
            ]
            mock_post.return_value = mock_response

            docs = mmore_client.retrieve("test query")

            assert len(docs) == 1
            assert isinstance(docs[0], Document)
            assert docs[0].page_content == "Test content"
            assert docs[0].metadata["source"] == "doc1"
            assert docs[0].metadata["score"] == 0.95

    def test_retrieve_with_file_ids(self, mmore_client):
        """Test retrieval restricted to specific file IDs."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_post.return_value = mock_response

            mmore_client.retrieve("query", file_ids=["doc1", "doc2"])

            # Verify fileIds were sent in request
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            assert payload["fileIds"] == ["doc1", "doc2"]

    def test_retrieve_with_parameters(self, mmore_client):
        """Test retrieval with custom parameters."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_post.return_value = mock_response

            mmore_client.retrieve("query", max_matches=10, min_similarity=0.7)

            # Verify parameters were sent
            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            assert payload["maxMatches"] == 10
            assert payload["minSimilarity"] == 0.7

    def test_retrieve_dict_response(self, mmore_client):
        """Test retrieval when response is dict with docs key."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "docs": [
                    {
                        "content": "Content",
                        "fileId": "doc1",
                        "score": 0.8,
                    }
                ]
            }
            mock_post.return_value = mock_response

            docs = mmore_client.retrieve("query")

            assert len(docs) == 1
            assert docs[0].page_content == "Content"

    def test_retrieve_http_error(self, mmore_client):
        """Test retrieval with HTTP error."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.HTTPError("Error")
            mock_post.return_value = mock_response

            with pytest.raises(requests.HTTPError):
                mmore_client.retrieve("query")

    def test_retrieve_empty_results(self, mmore_client):
        """Test retrieval with no results."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_post.return_value = mock_response

            docs = mmore_client.retrieve("query")

            assert len(docs) == 0


class TestMMOREDeleteFile:
    """Test MMORE file deletion functionality."""

    def test_delete_file_success(self, mmore_client):
        """Test successful file deletion."""
        with patch("requests.delete") as mock_delete:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "deleted"}
            mock_delete.return_value = mock_response

            result = mmore_client.delete_file("test-file")

            assert result["status"] == "deleted"
            mock_delete.assert_called_once_with(
                "http://test-mmore:8000/v1/files/test-file", timeout=30
            )

    def test_delete_file_http_error(self, mmore_client):
        """Test deletion with HTTP error."""
        with patch("requests.delete") as mock_delete:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.HTTPError("Not found")
            mock_delete.return_value = mock_response

            with pytest.raises(requests.HTTPError):
                mmore_client.delete_file("nonexistent")


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_set_and_get_callback(self):
        """Test setting and getting progress callback."""
        callback = Mock()

        set_progress_callback(callback)

        assert get_progress_callback() == callback

        # Clear callback
        set_progress_callback(None)
        assert get_progress_callback() is None

    def test_callback_not_called_when_not_set(self, mmore_client, temp_file):
        """Test that progress isn't reported when callback not set."""
        set_progress_callback(None)

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"fileId": "test"}
            mock_post.return_value = mock_response

            # Should not raise error even without callback
            mmore_client.upload_file(str(temp_file))
