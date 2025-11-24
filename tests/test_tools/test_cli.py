"""
Comprehensive tests for CLI tools.

These tests cover all CLI tool functions with proper mocking to ensure
they can run in CI environments without requiring actual external tools.
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.tools.cli import (
    _get_host_service_url,
    _is_running_in_docker,
    _open_via_host_service,
    check_cli_tool_available,
    execute_cli_command,
    get_prusa_slicer_path,
    list_directory_contents,
    open_gui_application,
)

# ============================================================================
# EXECUTE CLI COMMAND TESTS
# ============================================================================


@pytest.mark.unit
def test_execute_cli_command_success():
    """Test successful command execution."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Command output"
    mock_result.stderr = ""

    with patch("src.tools.cli.subprocess.run", return_value=mock_result):
        result = execute_cli_command.invoke({"command": "echo test"})

        assert "SUCCESS" in result
        assert "Exit Code: 0" in result
        assert "Command output" in result


@pytest.mark.unit
def test_execute_cli_command_failure():
    """Test command execution failure."""
    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error message"

    with patch("src.tools.cli.subprocess.run", return_value=mock_result):
        result = execute_cli_command.invoke({"command": "false"})

        assert "FAILED" in result
        assert "Exit Code: 1" in result
        assert "Error message" in result


@pytest.mark.unit
def test_execute_cli_command_with_working_directory():
    """Test command execution with custom working directory."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Success"
    mock_result.stderr = ""

    with (
        patch("src.tools.cli.subprocess.run", return_value=mock_result) as mock_run,
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        result = execute_cli_command.invoke({"command": "ls", "working_dir": tmpdir})

        assert "SUCCESS" in result
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["cwd"] == tmpdir


@pytest.mark.unit
def test_execute_cli_command_timeout():
    """Test command timeout handling."""
    with patch(
        "src.tools.cli.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)
    ):
        result = execute_cli_command.invoke({"command": "sleep 100", "timeout": 30})

        assert "timed out" in result
        assert "30 seconds" in result


@pytest.mark.unit
def test_execute_cli_command_nonexistent_executable():
    """Test handling of nonexistent executable."""
    with patch("src.tools.cli.which", return_value=None):
        result = execute_cli_command.invoke({"command": "nonexistent_command_xyz"})

        assert "not found" in result
        assert "nonexistent_command_xyz" in result


@pytest.mark.unit
def test_execute_cli_command_invalid_working_directory():
    """Test handling of invalid working directory."""
    result = execute_cli_command.invoke(
        {"command": "ls", "working_dir": "/nonexistent/path/xyz"}
    )

    assert "does not exist" in result


@pytest.mark.unit
def test_execute_cli_command_empty_command():
    """Test handling of empty command."""
    result = execute_cli_command.invoke({"command": ""})

    assert "Error" in result
    assert "Empty command" in result


@pytest.mark.unit
def test_execute_cli_command_with_stdout_and_stderr():
    """Test command that produces both stdout and stderr."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Standard output"
    mock_result.stderr = "Warning message"

    with patch("src.tools.cli.subprocess.run", return_value=mock_result):
        result = execute_cli_command.invoke({"command": "echo test"})

        assert "STDOUT:" in result
        assert "Standard output" in result
        assert "STDERR:" in result
        assert "Warning message" in result


@pytest.mark.unit
def test_execute_cli_command_exception_handling():
    """Test general exception handling."""
    with patch(
        "src.tools.cli.subprocess.run", side_effect=Exception("Unexpected error")
    ):
        result = execute_cli_command.invoke({"command": "test", "check_exists": False})

        assert "Error executing command" in result
        assert "Unexpected error" in result


@pytest.mark.unit
def test_execute_cli_command_without_check_exists():
    """Test command execution without checking if executable exists."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Success"
    mock_result.stderr = ""

    with patch("src.tools.cli.subprocess.run", return_value=mock_result):
        result = execute_cli_command.invoke(
            {"command": "some_command", "check_exists": False}
        )

        assert "SUCCESS" in result


# ============================================================================
# CHECK CLI TOOL AVAILABLE TESTS
# ============================================================================


@pytest.mark.unit
def test_check_cli_tool_available_found_with_version():
    """Test checking available tool with version info."""
    mock_version = Mock()
    mock_version.returncode = 0
    mock_version.stdout = "tool version 1.2.3"
    mock_version.stderr = ""

    with (
        patch("src.tools.cli.which", return_value="/usr/bin/tool"),
        patch("src.tools.cli.subprocess.run", return_value=mock_version),
    ):
        result = check_cli_tool_available.invoke({"tool_name": "tool"})

        assert "available" in result
        assert "/usr/bin/tool" in result
        assert "version 1.2.3" in result


@pytest.mark.unit
def test_check_cli_tool_available_found_without_version():
    """Test checking available tool without version info."""
    with (
        patch("src.tools.cli.which", return_value="/usr/local/bin/tool"),
        patch("src.tools.cli.subprocess.run", side_effect=Exception("No version")),
    ):
        result = check_cli_tool_available.invoke({"tool_name": "tool"})

        assert "available" in result
        assert "/usr/local/bin/tool" in result


@pytest.mark.unit
def test_check_cli_tool_available_not_found():
    """Test checking unavailable tool."""
    with patch("src.tools.cli.which", return_value=None):
        result = check_cli_tool_available.invoke({"tool_name": "missing_tool"})

        assert "NOT available" in result
        assert "missing_tool" in result


@pytest.mark.unit
def test_check_cli_tool_available_exception():
    """Test exception handling in tool availability check."""
    with patch("src.tools.cli.which", side_effect=Exception("System error")):
        result = check_cli_tool_available.invoke({"tool_name": "tool"})

        assert "Error checking tool availability" in result


@pytest.mark.unit
def test_check_cli_tool_available_version_in_stderr():
    """Test tool that outputs version info to stderr."""
    mock_version = Mock()
    mock_version.returncode = 0
    mock_version.stdout = ""
    mock_version.stderr = "tool 2.0.0"

    with (
        patch("src.tools.cli.which", return_value="/usr/bin/tool"),
        patch("src.tools.cli.subprocess.run", return_value=mock_version),
    ):
        result = check_cli_tool_available.invoke({"tool_name": "tool"})

        assert "available" in result
        assert "2.0.0" in result


# ============================================================================
# LIST DIRECTORY CONTENTS TESTS
# ============================================================================


@pytest.mark.unit
def test_list_directory_contents_success():
    """Test listing directory contents successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_dir = Path(tmpdir)
        (test_dir / "file1.txt").write_text("test")
        (test_dir / "file2.py").write_text("code")
        (test_dir / "subdir").mkdir()

        result = list_directory_contents.invoke({"directory_path": tmpdir})

        assert "Found 3 item(s)" in result
        assert "file1.txt" in result
        assert "file2.py" in result
        assert "[DIR]" in result
        assert "subdir/" in result


@pytest.mark.unit
def test_list_directory_contents_with_pattern():
    """Test listing directory contents with glob pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        (test_dir / "file1.txt").write_text("test")
        (test_dir / "file2.py").write_text("code")
        (test_dir / "file3.txt").write_text("test2")

        result = list_directory_contents.invoke(
            {"directory_path": tmpdir, "pattern": "*.txt"}
        )

        assert "Found 2 item(s)" in result
        assert "file1.txt" in result
        assert "file3.txt" in result
        assert "file2.py" not in result


@pytest.mark.unit
def test_list_directory_contents_empty_directory():
    """Test listing empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = list_directory_contents.invoke({"directory_path": tmpdir})

        assert "No files found" in result


@pytest.mark.unit
def test_list_directory_contents_nonexistent_directory():
    """Test listing nonexistent directory."""
    result = list_directory_contents.invoke({"directory_path": "/nonexistent/path/xyz"})

    assert "does not exist" in result


@pytest.mark.unit
def test_list_directory_contents_not_a_directory():
    """Test listing when path is a file, not a directory."""
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        try:
            result = list_directory_contents.invoke({"directory_path": tmpfile.name})

            assert "not a directory" in result
        finally:
            Path(tmpfile.name).unlink()


@pytest.mark.unit
def test_list_directory_contents_with_file_sizes():
    """Test that file sizes are displayed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        # Create file larger than 1MB
        large_file = test_dir / "large.txt"
        large_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

        # Create file larger than 1KB
        medium_file = test_dir / "medium.txt"
        medium_file.write_bytes(b"x" * (5 * 1024))  # 5 KB

        # Create small file
        small_file = test_dir / "small.txt"
        small_file.write_bytes(b"x" * 100)  # 100 bytes

        result = list_directory_contents.invoke({"directory_path": tmpdir})

        assert "MB" in result  # Large file
        assert "KB" in result  # Medium file
        assert "bytes" in result  # Small file


@pytest.mark.unit
def test_list_directory_contents_no_matches_with_pattern():
    """Test listing with pattern that matches no files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        (test_dir / "file.txt").write_text("test")

        result = list_directory_contents.invoke(
            {"directory_path": tmpdir, "pattern": "*.xyz"}
        )

        assert "No files found" in result
        assert "*.xyz" in result


@pytest.mark.unit
def test_list_directory_contents_exception():
    """Test exception handling in directory listing."""
    with patch("pathlib.Path.exists", side_effect=Exception("Permission error")):
        result = list_directory_contents.invoke({"directory_path": "/tmp"})

        assert "Error listing directory" in result


# ============================================================================
# OPEN GUI APPLICATION TESTS
# ============================================================================


@pytest.mark.unit
def test_open_gui_application_without_file():
    """Test opening GUI application without a file."""
    with patch("src.tools.cli.subprocess.Popen") as mock_popen:
        result = open_gui_application.invoke({"app_name": "TestApp"})

        assert "Opened 'TestApp'" in result
        assert "background" in result
        mock_popen.assert_called_once()


@pytest.mark.unit
def test_open_gui_application_with_file():
    """Test opening GUI application with a file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmpfile:
        try:
            with patch("src.tools.cli.subprocess.Popen") as mock_popen:
                result = open_gui_application.invoke(
                    {"app_name": "TestApp", "file_path": tmpfile.name}
                )

                assert "Opened 'TestApp'" in result
                assert tmpfile.name in result
                mock_popen.assert_called_once()
        finally:
            Path(tmpfile.name).unlink()


@pytest.mark.unit
def test_open_gui_application_nonexistent_file():
    """Test opening GUI application with nonexistent file."""
    result = open_gui_application.invoke(
        {"app_name": "TestApp", "file_path": "/nonexistent/file.txt"}
    )

    assert "does not exist" in result


@pytest.mark.unit
def test_open_gui_application_wait_for_exit():
    """Test opening GUI application and waiting for exit."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("src.tools.cli.subprocess.run", return_value=mock_result):
        result = open_gui_application.invoke(
            {"app_name": "TestApp", "wait_for_exit": True}
        )

        assert "closed successfully" in result


@pytest.mark.unit
def test_open_gui_application_wait_for_exit_with_error():
    """Test opening GUI application that exits with error."""
    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stderr = "Application error"

    with patch("src.tools.cli.subprocess.run", return_value=mock_result):
        result = open_gui_application.invoke(
            {"app_name": "TestApp", "wait_for_exit": True}
        )

        assert "exited with code 1" in result
        assert "Application error" in result


@pytest.mark.unit
def test_open_gui_application_not_found():
    """Test opening nonexistent application."""
    with patch("src.tools.cli.subprocess.Popen", side_effect=FileNotFoundError):
        result = open_gui_application.invoke({"app_name": "NonexistentApp"})

        assert "not found" in result


@pytest.mark.unit
def test_open_gui_application_exception():
    """Test general exception handling."""
    with patch(
        "src.tools.cli.subprocess.Popen", side_effect=Exception("Unexpected error")
    ):
        result = open_gui_application.invoke({"app_name": "TestApp"})

        assert "Error opening application" in result


@pytest.mark.unit
@pytest.mark.parametrize("system", ["Darwin", "Windows", "Linux"])
def test_open_gui_application_different_platforms(system):
    """Test opening application on different platforms."""
    with (
        patch("src.tools.cli.platform.system", return_value=system),
        patch("src.tools.cli.subprocess.Popen") as mock_popen,
        tempfile.NamedTemporaryFile(delete=False) as tmpfile,
    ):
        try:
            result = open_gui_application.invoke(
                {"app_name": "TestApp", "file_path": tmpfile.name}
            )

            assert "Opened" in result
            mock_popen.assert_called_once()

            # Check platform-specific command structure
            call_args = mock_popen.call_args[0][0]
            if system == "Darwin":
                assert "open" in call_args
            elif system == "Windows":
                assert "start" in call_args
            else:  # Linux
                assert "xdg-open" in call_args
        finally:
            Path(tmpfile.name).unlink()


# ============================================================================
# GET PRUSA SLICER PATH TESTS
# ============================================================================


@pytest.mark.unit
def test_get_prusa_slicer_path_from_env():
    """Test getting PrusaSlicer path from environment variable."""
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch.dict("os.environ", {"PRUSA_SLICER_PATH": tmpdir}),
    ):
        result = get_prusa_slicer_path.invoke({})

        assert "configured at" in result
        assert tmpdir in result


@pytest.mark.unit
def test_get_prusa_slicer_path_env_not_exists():
    """Test when environment path doesn't exist."""
    with patch.dict(
        "os.environ", {"PRUSA_SLICER_PATH": "/nonexistent/path"}, clear=False
    ):
        result = get_prusa_slicer_path.invoke({})

        # Should fall back to checking common locations
        assert "common locations" in result or "configure" in result


@pytest.mark.unit
def test_get_prusa_slicer_path_no_env_macos():
    """Test getting PrusaSlicer suggestions on macOS."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.tools.cli.platform.system", return_value="Darwin"),
    ):
        result = get_prusa_slicer_path.invoke({})

        assert "macOS common locations" in result
        assert "/Applications" in result


@pytest.mark.unit
def test_get_prusa_slicer_path_no_env_windows():
    """Test getting PrusaSlicer suggestions on Windows."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.tools.cli.platform.system", return_value="Windows"),
    ):
        result = get_prusa_slicer_path.invoke({})

        assert "Windows common locations" in result
        assert "Program Files" in result


@pytest.mark.unit
def test_get_prusa_slicer_path_no_env_linux():
    """Test getting PrusaSlicer suggestions on Linux."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.tools.cli.platform.system", return_value="Linux"),
        patch("src.tools.cli.which", return_value=None),
    ):
        result = get_prusa_slicer_path.invoke({})

        assert "not found in PATH" in result or "download" in result


@pytest.mark.unit
def test_get_prusa_slicer_path_linux_in_path():
    """Test finding PrusaSlicer in PATH on Linux."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.tools.cli.platform.system", return_value="Linux"),
        patch("src.tools.cli.which", return_value="/usr/bin/prusa-slicer"),
    ):
        result = get_prusa_slicer_path.invoke({})

        assert "found in PATH" in result
        assert "/usr/bin/prusa-slicer" in result


@pytest.mark.unit
def test_get_prusa_slicer_path_unknown_os():
    """Test handling unknown operating system."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("src.tools.cli.platform.system", return_value="FreeBSD"),
    ):
        result = get_prusa_slicer_path.invoke({})

        assert "Unknown operating system" in result or "configure" in result


# ============================================================================
# EDGE CASE AND ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.unit
def test_execute_cli_command_with_special_characters():
    """Test command with special characters in arguments."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Success"
    mock_result.stderr = ""

    with patch("src.tools.cli.subprocess.run", return_value=mock_result):
        result = execute_cli_command.invoke(
            {"command": 'echo "test with spaces"', "check_exists": False}
        )

        assert "SUCCESS" in result


@pytest.mark.unit
def test_list_directory_contents_sorted_output():
    """Test that directory listing is sorted alphabetically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        # Create files in non-alphabetical order
        (test_dir / "zebra.txt").write_text("test")
        (test_dir / "alpha.txt").write_text("test")
        (test_dir / "beta.txt").write_text("test")

        result = list_directory_contents.invoke({"directory_path": tmpdir})

        # Check that files appear in alphabetical order
        alpha_pos = result.find("alpha.txt")
        beta_pos = result.find("beta.txt")
        zebra_pos = result.find("zebra.txt")

        assert alpha_pos < beta_pos < zebra_pos


@pytest.mark.unit
def test_execute_cli_command_custom_timeout():
    """Test command execution with custom timeout."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Success"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        execute_cli_command.invoke(
            {"command": "test_command", "timeout": 600, "check_exists": False}
        )

        # Verify timeout was passed correctly
        assert mock_run.call_args[1]["timeout"] == 600


# ============================================================================
# CI COMPATIBILITY TESTS
# ============================================================================


@pytest.mark.unit
def test_all_functions_work_without_external_dependencies():
    """Test that all functions can be tested without external tools."""
    # This test ensures mocking works correctly for CI
    with (
        patch("src.tools.cli.which", return_value=None),
        patch("src.tools.cli.subprocess.run") as mock_run,
        patch("src.tools.cli.subprocess.Popen") as mock_popen,
    ):
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        # Test execute_cli_command
        result1 = check_cli_tool_available.invoke({"tool_name": "fake_tool"})
        assert "NOT available" in result1

        # Test open_gui_application
        result2 = open_gui_application.invoke({"app_name": "FakeApp"})
        assert "not found" in result2 or mock_popen.called


@pytest.mark.smoke
def test_basic_cli_functionality():
    """Smoke test to ensure basic CLI functionality works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test directory listing - doesn't require mocking
        result = list_directory_contents.invoke({"directory_path": tmpdir})
        assert "Found" in result or "No files found" in result


# ============================================================================
# PARAMETRIZED TESTS FOR COMPREHENSIVE COVERAGE
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "command,expected_error",
    [
        ("", "Empty command"),
        ("nonexistent_xyz", "not found"),
    ],
)
def test_execute_cli_command_error_cases(command, expected_error):
    """Parametrized test for different error cases."""
    with patch("src.tools.cli.which", return_value=None):
        result = execute_cli_command.invoke({"command": command})
        assert expected_error in result


@pytest.mark.unit
@pytest.mark.parametrize(
    "pattern,expected_files",
    [
        ("*.txt", ["file1.txt", "file2.txt"]),
        ("*.py", ["script.py"]),
        ("file1.*", ["file1.txt"]),
    ],
)
def test_list_directory_contents_patterns(pattern, expected_files):
    """Parametrized test for different glob patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        # Create test files
        (test_dir / "file1.txt").write_text("test")
        (test_dir / "file2.txt").write_text("test")
        (test_dir / "script.py").write_text("code")
        (test_dir / "data.json").write_text("{}")

        result = list_directory_contents.invoke(
            {"directory_path": tmpdir, "pattern": pattern}
        )

        for expected in expected_files:
            assert expected in result


# ============================================================================
# DOCKER DETECTION TESTS
# ============================================================================


@pytest.mark.unit
def test_is_running_in_docker_with_dockerenv():
    """Test Docker detection via .dockerenv file."""
    with patch("pathlib.Path.exists", return_value=True):
        result = _is_running_in_docker()
        assert result is True


@pytest.mark.unit
def test_is_running_in_docker_via_cgroup():
    """Test Docker detection via cgroup."""
    mock_cgroup_content = "12:devices:/docker/abc123\n"

    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.open", mock_open_func(mock_cgroup_content)),
    ):
        result = _is_running_in_docker()
        assert result is True


@pytest.mark.unit
def test_is_running_in_docker_not_in_docker():
    """Test detection when not in Docker."""
    mock_cgroup_content = "12:devices:/\n"

    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.open", mock_open_func(mock_cgroup_content)),
    ):
        result = _is_running_in_docker()
        assert result is False


@pytest.mark.unit
def test_is_running_in_docker_exception():
    """Test Docker detection when cgroup read fails."""
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.open", side_effect=PermissionError),
    ):
        result = _is_running_in_docker()
        assert result is False


# ============================================================================
# HOST SERVICE URL TESTS
# ============================================================================


@pytest.mark.unit
def test_get_host_service_url_default():
    """Test default host service URL."""
    with patch.dict("os.environ", {}, clear=True):
        result = _get_host_service_url()
        assert "host.docker.internal" in result
        assert ":9999" in result


@pytest.mark.unit
def test_get_host_service_url_custom_port():
    """Test host service URL with custom port."""
    with patch.dict("os.environ", {"HOST_SERVICE_PORT": "8888"}):
        result = _get_host_service_url()
        assert ":8888" in result


@pytest.mark.unit
def test_get_host_service_url_linux():
    """Test host service URL on Linux platform."""
    with patch("src.tools.cli.platform.system", return_value="Linux"):
        result = _get_host_service_url()
        assert "host.docker.internal" in result


# ============================================================================
# HOST SERVICE OPEN TESTS
# ============================================================================


@pytest.mark.unit
def test_open_via_host_service_success():
    """Test successful opening via host service."""
    mock_health_response = Mock()
    mock_health_response.status_code = 200

    mock_open_response = Mock()
    mock_open_response.json.return_value = {
        "success": True,
        "message": "Opened successfully",
    }

    with (
        patch("src.tools.cli.requests.get", return_value=mock_health_response),
        patch("src.tools.cli.requests.post", return_value=mock_open_response),
    ):
        result = _open_via_host_service("TestApp")
        assert "successfully" in result


@pytest.mark.unit
def test_open_via_host_service_with_file():
    """Test opening with file path via host service."""
    mock_health_response = Mock()
    mock_health_response.status_code = 200

    mock_open_response = Mock()
    mock_open_response.json.return_value = {
        "success": True,
        "message": "Opened with file",
    }

    with (
        patch("src.tools.cli.requests.get", return_value=mock_health_response),
        patch(
            "src.tools.cli.requests.post", return_value=mock_open_response
        ) as mock_post,
    ):
        result = _open_via_host_service("TestApp", "/path/to/file.txt")

        # Verify file_path was passed
        call_json = mock_post.call_args[1]["json"]
        assert call_json["file_path"] == "/path/to/file.txt"
        assert "Opened with file" in result


@pytest.mark.unit
def test_open_via_host_service_health_check_fail():
    """Test host service when health check fails."""
    mock_response = Mock()
    mock_response.status_code = 500

    with patch("src.tools.cli.requests.get", return_value=mock_response):
        result = _open_via_host_service("TestApp")
        assert "not responding" in result


@pytest.mark.unit
def test_open_via_host_service_connection_error():
    """Test host service when connection fails."""
    import requests

    with patch(
        "src.tools.cli.requests.get",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ):
        result = _open_via_host_service("TestApp")
        assert "Cannot connect" in result
        assert "host_service.py" in result


@pytest.mark.unit
def test_open_via_host_service_open_error():
    """Test host service when open request fails."""
    mock_health_response = Mock()
    mock_health_response.status_code = 200

    mock_open_response = Mock()
    mock_open_response.json.return_value = {
        "success": False,
        "message": "Application not found",
    }

    with (
        patch("src.tools.cli.requests.get", return_value=mock_health_response),
        patch("src.tools.cli.requests.post", return_value=mock_open_response),
    ):
        result = _open_via_host_service("NonexistentApp")
        assert "Error" in result
        assert "Application not found" in result


@pytest.mark.unit
def test_open_via_host_service_exception():
    """Test host service with unexpected exception."""
    with patch("src.tools.cli.requests.get", side_effect=Exception("Unexpected")):
        result = _open_via_host_service("TestApp")
        assert "Error" in result


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def mock_open_func(content):
    """Create a mock for Path.open that returns file content."""
    from contextlib import contextmanager
    from io import StringIO

    @contextmanager
    def mock_open(*_args, **_kwargs):
        yield StringIO(content)

    return mock_open
