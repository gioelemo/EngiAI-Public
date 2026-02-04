"""CLI command execution tools for running local command-line applications."""

import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path
from shutil import which

import requests
from langchain_core.tools import tool

# File size constants
_KB = 1024
_MB = 1024 * 1024

# HTTP status codes
HTTP_OK = 200

logger = logging.getLogger(__name__)


def _is_running_in_docker() -> bool:
    """Detect if code is running inside a Docker container."""
    # Check for .dockerenv file
    if Path("/.dockerenv").exists():
        return True

    # Check cgroup for docker
    try:
        with Path("/proc/self/cgroup").open() as f:
            return any("docker" in line for line in f)
    except Exception as e:
        # Expected on non-Linux systems or if file doesn't exist
        logger.debug("Could not read cgroup file for Docker detection: %s", e)

    return False


def _get_host_service_url() -> str:
    """Get the host service URL for Docker environments."""
    # In Docker, host.docker.internal resolves to the host machine
    # For Linux, we need to use the host IP or bridge network
    port = os.getenv("HOST_SERVICE_PORT", "9999")

    # Try host.docker.internal first (works on Docker Desktop for Mac/Windows)
    host = "host.docker.internal"

    # On Linux, host.docker.internal doesn't work by default
    # We can use the default gateway IP or configure extra_hosts in docker-compose
    if platform.system() == "Linux":
        # This will be configured via extra_hosts in docker-compose
        host = "host.docker.internal"

    return f"http://{host}:{port}"


def _open_via_host_service(app_name: str, file_path: str | None = None) -> str:
    """Open an application via the host service (for Docker environments)."""
    try:
        host_service_url = _get_host_service_url()
        logger.info(
            f"[HOST_SERVICE] Attempting to open '{app_name}' via {host_service_url}"
        )

        # Test if host service is available
        try:
            logger.info(
                f"[HOST_SERVICE] Testing connection to {host_service_url}/health"
            )
            response = requests.get(f"{host_service_url}/health", timeout=2)
            logger.info(f"[HOST_SERVICE] Health check response: {response.status_code}")
            if response.status_code != HTTP_OK:
                return f"Error: Host service at {host_service_url} is not responding correctly. Please start host_service.py on your host machine."
        except requests.exceptions.RequestException as e:
            logger.exception("[HOST_SERVICE] Connection failed")
            return f"""Error: Cannot connect to host service at {host_service_url}.

To fix this:
1. On your HOST machine, run: python host_service.py
2. Make sure the service is running on port {os.getenv("HOST_SERVICE_PORT", "9999")}

Details: {e!s}"""

        # Send request to open application
        logger.info(f"[HOST_SERVICE] Sending POST request to open {app_name}")
        response = requests.post(
            f"{host_service_url}/open",
            json={"app_name": app_name, "file_path": file_path},
            timeout=5,
        )

        result = response.json()
        logger.info(f"[HOST_SERVICE] Response: {result}")
        if result.get("success"):
            return result.get("message", "Application opened successfully")
        else:
            return f"Error from host service: {result.get('message', 'Unknown error')}"

    except Exception as e:
        logger.exception("[HOST_SERVICE] Exception occurred")
        return f"Error communicating with host service: {e!s}"


@tool
def execute_cli_command(
    command: str,
    working_dir: str | None = None,
    timeout: int = 300,
    check_exists: bool = True,
) -> str:
    """Execute a CLI command and return its output.

    This tool allows running local command-line applications like PrusaSlicer,
    mesh processing tools, converters, and other CLI utilities.

    Args:
        command: The complete command to execute (e.g., "PrusaSlicer --slice input.stl --output output.gcode")
        working_dir: Optional working directory for the command. If None, uses current directory.
        timeout: Maximum time in seconds to wait for command completion (default: 300)
        check_exists: Whether to check if the command executable exists before running (default: True)

    Returns:
        A string containing the command output (stdout and stderr combined) and execution status.

    Example:
        >>> execute_cli_command("PrusaSlicer --version")
        >>> execute_cli_command("PrusaSlicer --slice model.stl --output model.gcode", working_dir="/path/to/files")
        >>> execute_cli_command("meshlab.meshlabserver -i input.obj -o output.stl", timeout=600)

    Note:
        - The command should be properly quoted if it contains special characters
        - File paths should be absolute or relative to working_dir
        - Common use cases: STL slicing, mesh conversion, file processing, CAD operations
        - Always verify the tool is installed on the system before execution
    """
    try:
        # Parse the command
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return "Error: Empty command provided"

        # Check if executable exists (optional safety check)
        if check_exists:
            executable = cmd_parts[0]
            # Try to find the executable

            if which(executable) is None:
                return f"Error: Command '{executable}' not found in PATH. Please ensure it's installed and accessible."

        # Set working directory
        cwd = working_dir or str(Path.cwd())
        if working_dir and not Path(working_dir).exists():
            return f"Error: Working directory '{working_dir}' does not exist"

        # Execute the command
        result = subprocess.run(
            cmd_parts,
            check=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # Security: never use shell=True
        )

        # Prepare output
        output_lines = []
        output_lines.append(f"Command: {command}")
        output_lines.append(f"Working Directory: {cwd}")
        output_lines.append(f"Exit Code: {result.returncode}")
        output_lines.append("-" * 60)

        if result.stdout:
            output_lines.append("STDOUT:")
            output_lines.append(result.stdout)

        if result.stderr:
            output_lines.append("STDERR:")
            output_lines.append(result.stderr)

        if result.returncode == 0:
            output_lines.append("-" * 60)
            output_lines.append("Status: SUCCESS")
        else:
            output_lines.append("-" * 60)
            output_lines.append(f"Status: FAILED (exit code {result.returncode})")

        return "\n".join(output_lines)

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e!s}"


def _validate_file_path(file_path: str | None) -> str | None:
    """Validate file path exists. Returns error message if invalid, None if valid."""
    if file_path and not Path(file_path).exists():
        return f"Error: File '{file_path}' does not exist"
    return None


def _build_launch_command(
    system: str, app_name: str, file_path: str | None
) -> list[str]:
    """Build the launch command based on OS and parameters."""
    if system == "Darwin":  # macOS
        cmd = ["open", "-a", app_name]
        if file_path:
            cmd.append(str(Path(file_path).absolute()))
    elif system == "Windows":
        cmd = (
            ["start", "", str(Path(file_path).absolute())] if file_path else [app_name]
        )
    else:  # Linux and others
        cmd = ["xdg-open", str(Path(file_path).absolute())] if file_path else [app_name]
    return cmd


@tool
def open_gui_application(
    app_name: str,
    file_path: str | None = None,
    wait_for_exit: bool = False,
) -> str:
    """Open a GUI application, optionally with a file.

    **IMPORTANT**: This tool OPENS GUI applications - it doesn't check if they exist first.
    Just call it directly when the user says "open [app name]". The tool will handle
    finding the application automatically.

    This tool launches GUI applications like PrusaSlicer, Blender, MeshLab, etc.
    so you can interact with them directly instead of using CLI commands.

    Args:
        app_name: Name or path to the application (e.g., "PrusaSlicer", "Blender", "Mail")
                  Just use the simple name - the tool will find it automatically
        file_path: Optional file to open with the application
        wait_for_exit: If True, waits for the application to close before returning (default: False)

    Returns:
        Success message or error description.

    Example:
        User says "open PrusaSlicer" → open_gui_application("PrusaSlicer")
        User says "open Blender" → open_gui_application("Blender")
        User says "open Mail" → open_gui_application("Mail")

    Note:
        - DO NOT check if the app exists first - just call this tool directly
        - On macOS, can open .app bundles directly
        - On Windows, looks for .exe files
        - On Linux, uses standard application launcher
        - By default, launches in background so you can continue working
        - When running in Docker, uses the host service to open apps on the host machine
        - The tool handles finding the app path automatically
    """
    logger.info(
        f"[OPEN_GUI_APP] Called with app_name='{app_name}', file_path={file_path}"
    )

    try:
        # Check if running in Docker
        is_docker = _is_running_in_docker()
        logger.info(f"[OPEN_GUI_APP] Running in Docker: {is_docker}")

        if is_docker:
            # Use host service to open application on host machine
            logger.info("[OPEN_GUI_APP] Delegating to host service")
            return _open_via_host_service(app_name, file_path)

        # Validate file path
        error = _validate_file_path(file_path)
        if error:
            return error

        # Build command
        system = platform.system()
        cmd = _build_launch_command(system, app_name, file_path)

        # Launch the application
        if wait_for_exit:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            return (
                f"✓ Application '{app_name}' opened and closed successfully"
                if result.returncode == 0
                else f"Application exited with code {result.returncode}\nError: {result.stderr}"
            )
        else:
            # Launch in background
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            success_msg = f"✓ Opened '{app_name}' in background"
            if file_path:
                success_msg += f" with file: {file_path}"
            success_msg += "\nYou can now interact with the application GUI directly."
            return success_msg

    except FileNotFoundError:
        return f"Error: Application '{app_name}' not found. Please provide the full path or ensure it's installed."
    except Exception as e:
        return f"Error opening application: {e!s}"
