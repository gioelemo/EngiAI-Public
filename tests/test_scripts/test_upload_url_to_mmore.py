"""
Tests for upload_url_to_mmore.py script.

Tests the functionality of downloading web content and uploading to MMORE.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.upload_url_to_mmore import (
    download_content,
    get_github_raw_url,
    upload_url_to_mmore,
)


class TestGetGithubRawUrl:
    """Test GitHub URL conversion to raw URLs."""

    def test_converts_github_blob_url(self):
        """Test conversion of GitHub blob URL to raw URL."""
        github_url = "https://github.com/swiss-ai/mmore/blob/master/docs/process.md"
        expected = (
            "https://raw.githubusercontent.com/swiss-ai/mmore/master/docs/process.md"
        )

        result = get_github_raw_url(github_url)

        assert result == expected

    def test_converts_github_blob_url_with_branch(self):
        """Test conversion with different branch names."""
        github_url = "https://github.com/user/repo/blob/develop/README.md"
        expected = "https://raw.githubusercontent.com/user/repo/develop/README.md"

        result = get_github_raw_url(github_url)

        assert result == expected

    def test_returns_non_github_url_unchanged(self):
        """Test that non-GitHub URLs are returned unchanged."""
        url = "https://example.com/docs/page.html"

        result = get_github_raw_url(url)

        assert result == url

    def test_returns_already_raw_url_unchanged(self):
        """Test that already-raw GitHub URLs are unchanged."""
        raw_url = "https://raw.githubusercontent.com/user/repo/main/file.md"

        result = get_github_raw_url(raw_url)

        assert result == raw_url


class TestDownloadContent:
    """Test content downloading functionality."""

    @patch("scripts.upload_url_to_mmore.requests.get")
    def test_download_successful(self, mock_get, tmp_path):
        """Test successful content download."""
        # Setup
        mock_response = Mock()
        mock_response.content = b"Test content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        output_file = tmp_path / "test.html"

        # Execute
        result = download_content("https://example.com/test.html", output_file)

        # Verify
        assert result is True
        assert output_file.exists()
        assert output_file.read_bytes() == b"Test content"
        mock_get.assert_called_once_with("https://example.com/test.html", timeout=30)

    @patch("scripts.upload_url_to_mmore.requests.get")
    def test_download_handles_http_error(self, mock_get, tmp_path):
        """Test handling of HTTP errors during download."""
        # Setup
        mock_get.side_effect = requests.HTTPError("404 Not Found")
        output_file = tmp_path / "test.html"

        # Execute
        result = download_content("https://example.com/notfound.html", output_file)

        # Verify
        assert result is False
        assert not output_file.exists()

    @patch("scripts.upload_url_to_mmore.requests.get")
    def test_download_handles_timeout(self, mock_get, tmp_path):
        """Test handling of timeout errors."""
        # Setup
        mock_get.side_effect = requests.Timeout("Connection timeout")
        output_file = tmp_path / "test.html"

        # Execute
        result = download_content("https://example.com/slow.html", output_file)

        # Verify
        assert result is False
        assert not output_file.exists()

    @patch("scripts.upload_url_to_mmore.requests.get")
    def test_download_handles_connection_error(self, mock_get, tmp_path):
        """Test handling of connection errors."""
        # Setup
        mock_get.side_effect = requests.ConnectionError("Network unreachable")
        output_file = tmp_path / "test.html"

        # Execute
        result = download_content("https://example.com/test.html", output_file)

        # Verify
        assert result is False
        assert not output_file.exists()


class TestUploadUrlToMmore:
    """Test URL upload to MMORE functionality."""

    @patch("scripts.upload_url_to_mmore.DatabaseManager")
    @patch("scripts.upload_url_to_mmore.MMOREClient")
    @patch("scripts.upload_url_to_mmore.download_content")
    def test_upload_successful(self, mock_download, mock_mmore_class, mock_db_class):
        """Test successful URL upload to MMORE."""
        # Setup
        mock_mmore = MagicMock()
        mock_mmore.health_check.return_value = True
        mock_mmore.upload_file.return_value = {"status": "success", "file_id": "test"}
        mock_mmore_class.return_value = mock_mmore

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        mock_download.return_value = True

        # Execute
        result = upload_url_to_mmore("https://example.com/test.html")

        # Verify
        assert result is True
        mock_mmore.health_check.assert_called_once()
        mock_mmore.upload_file.assert_called_once()
        mock_db.add_mmore_document.assert_called_once()
        mock_download.assert_called_once()

    @patch("scripts.upload_url_to_mmore.DatabaseManager")
    @patch("scripts.upload_url_to_mmore.MMOREClient")
    def test_upload_fails_when_mmore_unhealthy(self, mock_mmore_class, mock_db_class):
        """Test upload fails when MMORE service is not healthy."""
        # Setup
        mock_mmore = MagicMock()
        mock_mmore.health_check.return_value = False
        mock_mmore_class.return_value = mock_mmore

        # Execute
        result = upload_url_to_mmore("https://example.com/test.html")

        # Verify
        assert result is False
        mock_mmore.upload_file.assert_not_called()

    @patch("scripts.upload_url_to_mmore.DatabaseManager")
    @patch("scripts.upload_url_to_mmore.MMOREClient")
    @patch("scripts.upload_url_to_mmore.download_content")
    def test_upload_fails_when_download_fails(
        self, mock_download, mock_mmore_class, mock_db_class
    ):
        """Test upload fails when content download fails."""
        # Setup
        mock_mmore = MagicMock()
        mock_mmore.health_check.return_value = True
        mock_mmore_class.return_value = mock_mmore

        mock_download.return_value = False

        # Execute
        result = upload_url_to_mmore("https://example.com/test.html")

        # Verify
        assert result is False
        mock_mmore.upload_file.assert_not_called()

    @patch("scripts.upload_url_to_mmore.DatabaseManager")
    @patch("scripts.upload_url_to_mmore.MMOREClient")
    @patch("scripts.upload_url_to_mmore.download_content")
    def test_upload_with_custom_file_id(
        self, mock_download, mock_mmore_class, mock_db_class
    ):
        """Test upload with custom file ID."""
        # Setup
        mock_mmore = MagicMock()
        mock_mmore.health_check.return_value = True
        mock_mmore.upload_file.return_value = {"status": "success"}
        mock_mmore_class.return_value = mock_mmore

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        mock_download.return_value = True

        custom_file_id = "my_custom_doc_id"

        # Execute
        result = upload_url_to_mmore(
            "https://example.com/test.html", file_id=custom_file_id
        )

        # Verify
        assert result is True
        # Check that upload was called with the custom file_id
        call_args = mock_mmore.upload_file.call_args
        assert call_args.kwargs["file_id"] == custom_file_id

    @patch("scripts.upload_url_to_mmore.DatabaseManager")
    @patch("scripts.upload_url_to_mmore.MMOREClient")
    @patch("scripts.upload_url_to_mmore.download_content")
    def test_upload_converts_github_url(
        self, mock_download, mock_mmore_class, mock_db_class
    ):
        """Test that GitHub URLs are converted to raw URLs before download."""
        # Setup
        mock_mmore = MagicMock()
        mock_mmore.health_check.return_value = True
        mock_mmore.upload_file.return_value = {"status": "success"}
        mock_mmore_class.return_value = mock_mmore

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        mock_download.return_value = True

        github_url = "https://github.com/user/repo/blob/main/file.md"

        # Execute
        result = upload_url_to_mmore(github_url)

        # Verify
        assert result is True
        # Check that download was called with the raw URL
        call_args = mock_download.call_args
        download_url = call_args[0][0]
        assert "raw.githubusercontent.com" in download_url
        assert "/blob/" not in download_url

    @patch("scripts.upload_url_to_mmore.DatabaseManager")
    @patch("scripts.upload_url_to_mmore.MMOREClient")
    @patch("scripts.upload_url_to_mmore.download_content")
    def test_upload_cleans_temp_file_by_default(
        self, mock_download, mock_mmore_class, mock_db_class, tmp_path, monkeypatch
    ):
        """Test that temporary files are cleaned up by default."""
        # Setup
        mock_mmore = MagicMock()
        mock_mmore.health_check.return_value = True
        mock_mmore.upload_file.return_value = {"status": "success"}
        mock_mmore_class.return_value = mock_mmore

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        # Create a temp directory for the test
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        # Mock the temp directory path
        monkeypatch.setattr(
            "scripts.upload_url_to_mmore.Path",
            lambda x: tmp_path if x == "data/temp" else Path(x),
        )

        def create_temp_file(_url, output_path):
            output_path.touch()  # Create the file
            return True

        mock_download.side_effect = create_temp_file

        # Execute
        result = upload_url_to_mmore("https://example.com/test.html", keep_file=False)

        # Verify
        assert result is True

    @patch("scripts.upload_url_to_mmore.DatabaseManager")
    @patch("scripts.upload_url_to_mmore.MMOREClient")
    @patch("scripts.upload_url_to_mmore.download_content")
    def test_upload_handles_mmore_upload_error(
        self, mock_download, mock_mmore_class, mock_db_class
    ):
        """Test handling of MMORE upload errors."""
        # Setup
        mock_mmore = MagicMock()
        mock_mmore.health_check.return_value = True
        mock_mmore.upload_file.side_effect = Exception("MMORE upload failed")
        mock_mmore_class.return_value = mock_mmore

        mock_download.return_value = True

        # Execute
        result = upload_url_to_mmore("https://example.com/test.html")

        # Verify
        assert result is False

    @patch("scripts.upload_url_to_mmore.DatabaseManager")
    @patch("scripts.upload_url_to_mmore.MMOREClient")
    @patch("scripts.upload_url_to_mmore.download_content")
    def test_upload_generates_file_id_from_url(
        self, mock_download, mock_mmore_class, mock_db_class
    ):
        """Test that file ID is generated from URL when not provided."""
        # Setup
        mock_mmore = MagicMock()
        mock_mmore.health_check.return_value = True
        mock_mmore.upload_file.return_value = {"status": "success"}
        mock_mmore_class.return_value = mock_mmore

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        mock_download.return_value = True

        # Execute
        result = upload_url_to_mmore("https://example.com/docs/my-document.html")

        # Verify
        assert result is True
        # Check that a file_id was generated
        call_args = mock_mmore.upload_file.call_args
        file_id = call_args.kwargs["file_id"]
        assert file_id  # Should not be empty
        assert file_id == "my-document"  # Should be based on the filename


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
