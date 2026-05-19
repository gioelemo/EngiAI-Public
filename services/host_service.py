#!/usr/bin/env python3
"""
Host Service for Docker Container GUI Integration.

This service runs on the HOST machine and allows the Docker container
to open GUI applications and execute host commands via HTTP requests.

Usage:
    python host_service.py

The service will listen on http://localhost:9999 by default.
"""

import logging
import os
import platform
import subprocess
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security: Whitelist of allowed applications
ALLOWED_APPS = {
    "prusaslicer": ["PrusaSlicer"],
    "prusa slicer": ["PrusaSlicer"],
    "terminal": ["Terminal", "iTerm", "Alacritty"],
    "vscode": ["Visual Studio Code", "code"],
    "finder": ["Finder"],
}


def _find_app_path(app_name: str) -> str | None:
    """Find the actual path/command for an application."""
    app_name_lower = app_name.lower()

    # Check whitelist
    if app_name_lower in ALLOWED_APPS:
        for candidate in ALLOWED_APPS[app_name_lower]:
            if Path(candidate).exists():
                return candidate
        # Return first candidate as fallback (might work if in PATH)
        return ALLOWED_APPS[app_name_lower][0]

    # Allow exact matches from whitelist values
    for allowed_list in ALLOWED_APPS.values():
        if app_name in allowed_list:
            if Path(app_name).exists():
                return app_name
            return app_name  # Might be in PATH

    return None


def _open_application_macos(app_path: str, file_path: str | None = None) -> dict:
    """Open application on macOS."""
    try:
        cmd = ["open", "-a", app_path]
        if file_path:
            cmd.append(str(Path(file_path).absolute()))

        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    except Exception as e:
        logger.exception("Error opening application")
        return {"success": False, "message": f"Error: {e!s}"}
    else:
        success_msg = f"✓ Opened '{app_path}'"
        if file_path:
            success_msg += f" with file: {file_path}"

        return {"success": True, "message": success_msg}


def _open_application_linux(app_path: str, file_path: str | None = None) -> dict:
    """Open application on Linux."""
    try:
        cmd = ["xdg-open", str(Path(file_path).absolute())] if file_path else [app_path]

        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    except Exception as e:
        logger.exception("Error opening application")
        return {"success": False, "message": f"Error: {e!s}"}
    else:
        success_msg = f"✓ Opened '{app_path}'"
        if file_path:
            success_msg += f" with file: {file_path}"

        return {"success": True, "message": success_msg}


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "platform": platform.system()})


@app.route("/open", methods=["POST"])
def open_application():
    """
    Open a GUI application on the host.

    Request JSON:
    {
        "app_name": "PrusaSlicer",
        "file_path": "/path/to/file.stl"  # optional
    }
    """
    try:
        data = request.get_json()
        app_name = data.get("app_name")
        file_path = data.get("file_path")

        if not app_name:
            return jsonify({"success": False, "message": "app_name is required"}), 400

        # Find the application
        app_path = _find_app_path(app_name)
        if not app_path:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Application '{app_name}' not found or not whitelisted. Allowed: {list(ALLOWED_APPS.keys())}",
                    }
                ),
                404,
            )

        # Validate file path if provided
        if file_path and not Path(file_path).exists():
            return (
                jsonify(
                    {"success": False, "message": f"File '{file_path}' does not exist"}
                ),
                400,
            )

        logger.info(f"Opening application: {app_path} with file: {file_path}")

        # Platform-specific opening
        system = platform.system()
        if system == "Darwin":
            result = _open_application_macos(app_path, file_path)
        elif system == "Linux":
            result = _open_application_linux(app_path, file_path)
        else:
            result = {
                "success": False,
                "message": f"Unsupported platform: {system}",
            }

        status_code = 200 if result["success"] else 500
        return jsonify(result), status_code

    except Exception as e:
        logger.exception("Error in open_application endpoint")
        return jsonify({"success": False, "message": f"Server error: {e!s}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("HOST_SERVICE_PORT", "9999"))
    logger.info(f"Starting Host Service on http://localhost:{port}")
    logger.info(f"Allowed applications: {list(ALLOWED_APPS.keys())}")
    logger.info(f"Platform: {platform.system()}")

    # Only listen on localhost for security
    app.run(host="127.0.0.1", port=port, debug=False)
