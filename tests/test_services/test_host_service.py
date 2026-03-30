"""Tests for host service."""

import json
from unittest.mock import Mock, patch

import pytest

# Try to import host_service - skip tests if dependencies not available
try:
    from host_service import (
        ALLOWED_APPS,
        _find_app_path,
        _open_application_linux,
        _open_application_macos,
        app,
    )

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FLASK_AVAILABLE, reason="Flask not available - host_service tests skipped"
)


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAllowedApps:
    """Tests for ALLOWED_APPS configuration."""

    def test_allowed_apps_structure(self):
        """Test that ALLOWED_APPS has expected structure."""
        assert isinstance(ALLOWED_APPS, dict)
        assert len(ALLOWED_APPS) > 0

    def test_allowed_apps_contains_prusaslicer(self):
        """Test that PrusaSlicer is in allowed apps."""
        assert "prusaslicer" in ALLOWED_APPS
        assert "PrusaSlicer" in ALLOWED_APPS["prusaslicer"]

    def test_allowed_apps_contains_terminal(self):
        """Test that terminal apps are allowed."""
        assert "terminal" in ALLOWED_APPS
        assert len(ALLOWED_APPS["terminal"]) > 0

    def test_allowed_apps_contains_vscode(self):
        """Test that VS Code is allowed."""
        assert "vscode" in ALLOWED_APPS


class TestFindAppPath:
    """Tests for _find_app_path function."""

    def test_find_app_path_whitelisted(self):
        """Test finding whitelisted app."""
        result = _find_app_path("prusaslicer")
        assert result is not None
        assert isinstance(result, str)

    def test_find_app_path_case_insensitive(self):
        """Test that app name lookup is case insensitive."""
        result = _find_app_path("PrusaSlicer")
        assert result is not None

    def test_find_app_path_from_whitelist_value(self):
        """Test finding app by exact whitelist value."""
        result = _find_app_path("PrusaSlicer")
        assert result is not None

    def test_find_app_path_not_found(self):
        """Test that non-whitelisted apps return None."""
        result = _find_app_path("malicious_app")
        assert result is None

    def test_find_app_path_empty_string(self):
        """Test with empty string."""
        result = _find_app_path("")
        assert result is None

    @patch("host_service.Path")
    def test_find_app_path_existing_file(self, mock_path):
        """Test finding app when file exists."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        result = _find_app_path("prusaslicer")
        assert result is not None


class TestOpenApplicationMacOS:
    """Tests for _open_application_macos function."""

    @patch("host_service.subprocess.Popen")
    def test_open_application_macos_success(self, mock_popen):
        """Test successful app opening on macOS."""
        mock_popen.return_value = Mock()

        result = _open_application_macos("PrusaSlicer")

        assert result["success"] is True
        assert "PrusaSlicer" in result["message"]
        mock_popen.assert_called_once()

    @patch("host_service.subprocess.Popen")
    def test_open_application_macos_with_file(self, mock_popen):
        """Test opening app with file on macOS."""
        mock_popen.return_value = Mock()

        result = _open_application_macos("PrusaSlicer", "/path/to/file.stl")

        assert result["success"] is True
        assert "file.stl" in result["message"]

    @patch("host_service.subprocess.Popen")
    def test_open_application_macos_failure(self, mock_popen):
        """Test app opening failure on macOS."""
        mock_popen.side_effect = Exception("Failed to open")

        result = _open_application_macos("PrusaSlicer")

        assert result["success"] is False
        assert "Error" in result["message"]

    @patch("host_service.subprocess.Popen")
    @patch("host_service.Path")
    def test_open_application_macos_absolute_path(self, mock_path, mock_popen):
        """Test that file path is converted to absolute."""
        mock_popen.return_value = Mock()
        mock_path_instance = Mock()
        mock_path_instance.absolute.return_value = "/absolute/path/file.stl"
        mock_path.return_value = mock_path_instance

        result = _open_application_macos("App", "relative/file.stl")

        assert result["success"] is True


class TestOpenApplicationLinux:
    """Tests for _open_application_linux function."""

    @patch("host_service.subprocess.Popen")
    def test_open_application_linux_success(self, mock_popen):
        """Test successful app opening on Linux."""
        mock_popen.return_value = Mock()

        result = _open_application_linux("application")

        assert result["success"] is True
        assert "application" in result["message"]

    @patch("host_service.subprocess.Popen")
    def test_open_application_linux_with_file(self, mock_popen):
        """Test opening file on Linux."""
        mock_popen.return_value = Mock()

        result = _open_application_linux("app", "/path/to/file.stl")

        assert result["success"] is True
        assert "file.stl" in result["message"]

    @patch("host_service.subprocess.Popen")
    def test_open_application_linux_failure(self, mock_popen):
        """Test app opening failure on Linux."""
        mock_popen.side_effect = Exception("Failed")

        result = _open_application_linux("app")

        assert result["success"] is False
        assert "Error" in result["message"]


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "platform" in data

    def test_health_check_returns_platform(self, client):
        """Test that health check returns platform info."""
        response = client.get("/health")
        data = json.loads(response.data)

        assert isinstance(data["platform"], str)
        assert len(data["platform"]) > 0


class TestOpenEndpoint:
    """Tests for /open endpoint."""

    @patch("host_service._find_app_path")
    @patch("host_service.platform.system")
    @patch("host_service._open_application_macos")
    def test_open_application_success(
        self, mock_open_macos, mock_platform, mock_find_app, client
    ):
        """Test successful application opening."""
        mock_find_app.return_value = "PrusaSlicer"
        mock_platform.return_value = "Darwin"
        mock_open_macos.return_value = {
            "success": True,
            "message": "✓ Opened 'PrusaSlicer'",
        }

        response = client.post(
            "/open",
            data=json.dumps({"app_name": "prusaslicer"}),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

    def test_open_application_missing_app_name(self, client):
        """Test opening without app_name."""
        response = client.post(
            "/open",
            data=json.dumps({}),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
        assert "required" in data["message"]

    @patch("host_service._find_app_path")
    def test_open_application_not_found(self, mock_find_app, client):
        """Test opening non-existent application."""
        mock_find_app.return_value = None

        response = client.post(
            "/open",
            data=json.dumps({"app_name": "nonexistent"}),
            content_type="application/json",
        )

        assert response.status_code == 404
        data = json.loads(response.data)
        assert data["success"] is False
        assert "not found" in data["message"]

    @patch("host_service._find_app_path")
    @patch("host_service.Path")
    def test_open_application_invalid_file_path(self, mock_path, mock_find_app, client):
        """Test opening with non-existent file path."""
        mock_find_app.return_value = "PrusaSlicer"
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        response = client.post(
            "/open",
            data=json.dumps(
                {
                    "app_name": "prusaslicer",
                    "file_path": "/nonexistent/file.stl",
                }
            ),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
        assert "does not exist" in data["message"]

    @patch("host_service._find_app_path")
    @patch("host_service.platform.system")
    @patch("host_service._open_application_linux")
    @patch("host_service.Path")
    def test_open_application_linux_with_file(
        self, mock_path, mock_open_linux, mock_platform, mock_find_app, client
    ):
        """Test opening application on Linux with file."""
        mock_find_app.return_value = "app"
        mock_platform.return_value = "Linux"
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        mock_open_linux.return_value = {"success": True, "message": "Opened"}

        response = client.post(
            "/open",
            data=json.dumps(
                {
                    "app_name": "app",
                    "file_path": "/path/to/file.stl",
                }
            ),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

    def test_open_application_invalid_json(self, client):
        """Test with invalid JSON data."""
        response = client.post(
            "/open",
            data="invalid json",
            content_type="application/json",
        )

        # Should handle gracefully
        assert response.status_code in [400, 500]
