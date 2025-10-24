"""CLI command execution tools for running local command-line applications."""

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
        command: The complete command to execute (e.g., "prusa-slicer --slice input.stl --output output.gcode")
        working_dir: Optional working directory for the command. If None, uses current directory.
        timeout: Maximum time in seconds to wait for command completion (default: 300)
        check_exists: Whether to check if the command executable exists before running (default: True)

    Returns:
        A string containing the command output (stdout and stderr combined) and execution status.

    Example:
        >>> execute_cli_command("prusa-slicer --version")
        >>> execute_cli_command("prusa-slicer --slice model.stl --output model.gcode", working_dir="/path/to/files")
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
        tool_name: Name of the command/tool to check (e.g., "prusa-slicer", "meshlab.meshlabserver")

    Returns:
        String indicating whether the tool is available and its path if found.

    Example:
        >>> check_cli_tool_available("prusa-slicer")
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
