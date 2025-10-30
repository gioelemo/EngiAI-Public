"""CLI command execution tools for running local command-line applications."""

import os
import platform
import shlex
import subprocess
from pathlib import Path
from shutil import which

from langchain_core.tools import tool

# File size constants
_KB = 1024
_MB = 1024 * 1024


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
        cwd = working_dir if working_dir else str(Path.cwd())
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


@tool
def check_cli_tool_available(tool_name: str) -> str:
    """Check if a CLI tool is available on the system.

    Args:
        tool_name: Name of the command/tool to check (e.g., "PrusaSlicer", "meshlab.meshlabserver")

    Returns:
        String indicating whether the tool is available and its path if found.

    Example:
        >>> check_cli_tool_available("PrusaSlicer")
        >>> check_cli_tool_available("git")
    """
    try:
        tool_path = which(tool_name)

        if tool_path:
            # Try to get version info
            try:
                version_result = subprocess.run(
                    [tool_name, "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version_info = (
                    version_result.stdout.strip() or version_result.stderr.strip()
                )
            except Exception:
                # If version check fails, still report tool is available
                return f"✓ Tool '{tool_name}' is available at: {tool_path}"
            else:
                return f"✓ Tool '{tool_name}' is available at: {tool_path}\nVersion info:\n{version_info}"
        else:
            return f"✗ Tool '{tool_name}' is NOT available in PATH. Please ensure it's installed."

    except Exception as e:
        return f"Error checking tool availability: {e!s}"


@tool
def list_directory_contents(directory_path: str, pattern: str | None = None) -> str:
    """List contents of a directory, optionally filtering by file pattern.

    Useful for checking available files before processing them with CLI tools.

    Args:
        directory_path: Path to the directory to list
        pattern: Optional glob pattern to filter files (e.g., "*.stl", "*.gcode")

    Returns:
        Formatted list of files and directories matching the pattern.

    Example:
        >>> list_directory_contents("/path/to/models")
        >>> list_directory_contents("/path/to/models", "*.stl")
    """
    try:
        dir_path = Path(directory_path)

        if not dir_path.exists():
            return f"Error: Directory '{directory_path}' does not exist"

        if not dir_path.is_dir():
            return f"Error: '{directory_path}' is not a directory"

        # Get file list
        if pattern:
            files = [p.name for p in dir_path.glob(pattern)]
        else:
            files = [p.name for p in dir_path.iterdir()]

        if not files:
            return f"No files found in '{directory_path}'" + (
                f" matching pattern '{pattern}'" if pattern else ""
            )

        # Sort files
        files.sort()

        # Format output
        output_lines = [f"Directory: {directory_path}"]
        if pattern:
            output_lines.append(f"Pattern: {pattern}")
        output_lines.append(f"Found {len(files)} item(s):")
        output_lines.append("-" * 60)

        for file in files:
            full_path = dir_path / file
            if full_path.is_dir():
                output_lines.append(f"  [DIR]  {file}/")
            else:
                # Get file size
                size = full_path.stat().st_size
                size_str = f"{size:,} bytes"
                if size > _MB:
                    size_str = f"{size / _MB:.2f} MB"
                elif size > _KB:
                    size_str = f"{size / _KB:.2f} KB"
                output_lines.append(f"  [FILE] {file} ({size_str})")

        return "\n".join(output_lines)

    except Exception as e:
        return f"Error listing directory: {e!s}"


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

    This tool launches GUI applications like PrusaSlicer, Blender, MeshLab, etc.
    so you can interact with them directly instead of using CLI commands.

    Args:
        app_name: Name or path to the application (e.g., "PrusaSlicer", "/Applications/PrusaSlicer.app")
        file_path: Optional file to open with the application
        wait_for_exit: If True, waits for the application to close before returning (default: False)

    Returns:
        Success message or error description.

    Example:
        >>> open_gui_application("PrusaSlicer")
        >>> open_gui_application("PrusaSlicer", file_path="model.stl")
        >>> open_gui_application("/Applications/Original Prusa Drivers/PrusaSlicer.app")

    Note:
        - On macOS, can open .app bundles directly
        - On Windows, looks for .exe files
        - On Linux, uses standard application launcher
        - By default, launches in background so you can continue working
    """
    try:
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


def _check_macos_prusa_paths() -> list[str]:
    """Check common PrusaSlicer paths on macOS."""
    suggestions = ["macOS common locations:"]
    common_paths = [
        "/Applications/Original Prusa Drivers/PrusaSlicer.app",
        "/Applications/PrusaSlicer.app",
        "~/Applications/PrusaSlicer.app",
    ]
    for path in common_paths:
        expanded = Path(path).expanduser()
        status = "✓ FOUND" if expanded.exists() else "✗ Not found"
        suggestions.append(f"  {status}: {path}")
    return suggestions


def _check_windows_prusa_paths() -> list[str]:
    """Check common PrusaSlicer paths on Windows."""
    suggestions = ["Windows common locations:"]
    common_paths = [
        r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe",
        r"C:\Program Files (x86)\Prusa3D\PrusaSlicer\prusa-slicer.exe",
    ]
    for path in common_paths:
        status = "✓ FOUND" if Path(path).exists() else "✗ Not found"
        suggestions.append(f"  {status}: {path}")
    return suggestions


def _check_linux_prusa_paths() -> list[str]:
    """Check common PrusaSlicer paths on Linux."""
    prusa_in_path = which("prusa-slicer")
    if prusa_in_path:
        return [f"✓ PrusaSlicer found in PATH: {prusa_in_path}"]
    return [
        "PrusaSlicer not found in PATH",
        "Try: sudo apt install prusa-slicer  (Ubuntu/Debian)",
        "Or download from: https://www.prusa3d.com/page/prusaslicer_424/",
    ]


@tool
def get_prusa_slicer_path() -> str:
    """Get the configured PrusaSlicer path from environment or suggest common locations.

    This tool helps locate PrusaSlicer on your system for GUI operations.

    Returns:
        The PrusaSlicer path or suggestions for common installation locations.

    Example:
        >>> get_prusa_slicer_path()
    """
    # Check environment variable
    prusa_path = os.getenv("PRUSA_SLICER_PATH")
    if prusa_path and Path(prusa_path).exists():
        return f"✓ PrusaSlicer configured at: {prusa_path}\n\nUse open_gui_application('{prusa_path}') to launch it."

    # Check common locations based on OS
    system = platform.system()
    path_checkers = {
        "Darwin": _check_macos_prusa_paths,
        "Windows": _check_windows_prusa_paths,
        "Linux": _check_linux_prusa_paths,
    }

    suggestions = (
        path_checkers[system]()
        if system in path_checkers
        else [f"Unknown operating system: {system}"]
    )

    suggestions.append("\nTo configure, set PRUSA_SLICER_PATH in your .env file")
    return "\n".join(suggestions)
