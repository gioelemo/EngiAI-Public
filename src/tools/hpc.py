"""
HPC cluster job management tools for SLURM job submission and monitoring.

This module provides tools for interacting with HPC clusters via SSH,
including job submission, status monitoring, cancellation, and output retrieval.
"""

import logging
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from langchain_core.tools import tool

from config import config
from src.tools.connection import HPCConnection

logger = logging.getLogger(__name__)

# Session credentials storage - keyed by session_id for multi-user isolation
_session_credentials_store: dict[str, dict[str, Any]] = {}

# Current session ID - set by Streamlit before tool invocation
_current_session_id: str | None = None

# Progress callback - set by Streamlit UI to receive progress updates
_progress_callback: Any | None = None

# Credential expiration time in seconds (20 minutes for security)
CREDENTIAL_EXPIRATION_SECONDS = 1200

# Encryption key for password storage (generated once per application lifetime)
# This provides defense-in-depth - passwords are encrypted in memory
_encryption_key = Fernet.generate_key()
_cipher = Fernet(_encryption_key)


def _encrypt_password(password: str) -> str:
    """Encrypt a password for secure in-memory storage."""
    return _cipher.encrypt(password.encode()).decode()


def _decrypt_password(encrypted_password: str) -> str:
    """Decrypt a password from secure storage."""
    return _cipher.decrypt(encrypted_password.encode()).decode()


# Hostname validation constants
MAX_HOSTNAME_LENGTH = 253
MAX_LABEL_LENGTH = 63
MAX_USERNAME_LENGTH = 64
MAX_PORT = 65535


def set_current_session_id(session_id: str | None) -> None:
    """
    Set the current session ID for credential isolation.

    This should be called by Streamlit before invoking any HPC tools.

    Args:
        session_id: Unique session identifier
    """
    global _current_session_id  # noqa: PLW0603
    _current_session_id = session_id


def set_hpc_progress_callback(callback: Any | None) -> None:
    """
    Set the progress callback for HPC operations.

    This should be called by Streamlit before invoking HPC tools.

    Args:
        callback: Callback function(step: str, message: str) or None to clear
    """
    global _progress_callback  # noqa: PLW0603
    _progress_callback = callback


def get_hpc_progress_callback() -> Any | None:
    """Get the current HPC progress callback."""
    return _progress_callback


def validate_hostname(hostname: str) -> bool:
    """
    Validate hostname format to prevent injection attacks.

    Args:
        hostname: The hostname to validate

    Returns:
        True if valid, False otherwise
    """
    # Allow alphanumeric, dots, and hyphens only
    pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?$"
    if not re.match(pattern, hostname):
        return False
    # Check length
    if len(hostname) > MAX_HOSTNAME_LENGTH:
        return False
    # Check each label
    labels = hostname.split(".")
    return all(len(label) <= MAX_LABEL_LENGTH for label in labels)


def validate_username(username: str) -> bool:
    """
    Validate username format.

    Args:
        username: The username to validate

    Returns:
        True if valid, False otherwise
    """
    # Allow alphanumeric, underscores, and hyphens
    pattern = r"^[a-zA-Z0-9_\-]+$"
    return bool(re.match(pattern, username)) and len(username) <= MAX_USERNAME_LENGTH


def set_ssh_credentials(
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
    port: int = 22,
    session_id: str | None = None,
) -> tuple[bool, str]:
    """
    Set SSH credentials for a specific session.

    Args:
        host: Hostname (e.g., "hpc.example.com")
        user: Username
        password: Password
        port: SSH port (default: 22)
        session_id: Session ID for isolation (uses current if not provided)

    Returns:
        Tuple of (success, message)
    """
    sid = session_id or _current_session_id
    if not sid:
        return False, "No session ID available"

    if not (host and user and password):
        # Clear credentials
        if sid in _session_credentials_store:
            del _session_credentials_store[sid]
            logger.info(f"Cleared SSH credentials for session {sid[:8]}...")
        return True, "Credentials cleared"

    # Validate inputs
    if not validate_hostname(host):
        return False, f"Invalid hostname format: {host}"

    if not validate_username(user):
        return False, f"Invalid username format: {user}"

    if not (1 <= port <= MAX_PORT):
        return False, f"Invalid port number: {port}"

    # Store credentials with expiration (password encrypted for security)
    _session_credentials_store[sid] = {
        "host": host,
        "user": user,
        "password": _encrypt_password(password),
        "port": port,
        "created_at": time.time(),
        "expires_at": time.time() + CREDENTIAL_EXPIRATION_SECONDS,
    }

    logger.info(
        f"SSH credentials set for session {sid[:8]}... (expires in {CREDENTIAL_EXPIRATION_SECONDS // 60} min)"
    )
    return (
        True,
        f"Credentials saved (expires in {CREDENTIAL_EXPIRATION_SECONDS // 60} minutes)",
    )


def clear_ssh_credentials(session_id: str | None = None) -> None:
    """
    Clear stored SSH credentials for a session.

    Args:
        session_id: Session ID (uses current if not provided)
    """
    sid = session_id or _current_session_id
    if sid and sid in _session_credentials_store:
        del _session_credentials_store[sid]
        logger.info(f"Cleared SSH credentials for session {sid[:8]}...")


def get_ssh_credentials(session_id: str | None = None) -> dict[str, Any] | None:
    """
    Get SSH credentials for a session.

    Args:
        session_id: Session ID (uses current if not provided)

    Returns:
        Credentials dict or None if not set/expired
    """
    sid = session_id or _current_session_id
    if not sid:
        return None

    creds = _session_credentials_store.get(sid)
    if not creds:
        return None

    # Check expiration
    if creds.get("expires_at", 0) < time.time():
        del _session_credentials_store[sid]
        logger.info(f"SSH credentials expired for session {sid[:8]}...")
        return None

    # Return with decrypted password (without internal fields)
    return {
        "host": creds["host"],
        "user": creds["user"],
        "password": _decrypt_password(creds["password"]),
        "port": creds["port"],
    }


@contextmanager
def get_ssh_credentials_secure(
    session_id: str | None = None,
) -> Generator[dict[str, Any] | None]:
    """
    Securely get SSH credentials and clear from memory after use.

    This context manager ensures the decrypted password is cleared
    from the returned dictionary after the block exits.

    Args:
        session_id: Session ID (uses current if not provided)

    Yields:
        dict or None: Credentials dict or None if not set/expired

    Example::

        with get_ssh_credentials_secure() as creds:
            if creds:
                connection = HPCConnection(..., password=creds["password"])
        # Password is cleared from creds after this point
    """
    creds = get_ssh_credentials(session_id)
    try:
        yield creds
    finally:
        # Clear the decrypted password from memory
        if creds and "password" in creds:
            # Overwrite with zeros before deletion
            creds["password"] = "\x00" * len(creds["password"])
            del creds["password"]


def cleanup_expired_credentials() -> int:
    """
    Remove all expired credentials from storage.

    Returns:
        Number of expired sessions cleaned up
    """
    current_time = time.time()
    expired = [
        sid
        for sid, creds in _session_credentials_store.items()
        if creds.get("expires_at", 0) < current_time
    ]
    for sid in expired:
        del _session_credentials_store[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired credential sessions")
    return len(expired)


def _create_hpc_connection(host_alias: str) -> HPCConnection:
    """
    Create an HPC connection using session credentials or SSH config.

    Args:
        host_alias: Host alias for SSH config fallback

    Returns:
        HPCConnection instance
    """
    # Clean up expired credentials periodically
    cleanup_expired_credentials()

    # Get progress callback if set
    progress_callback = get_hpc_progress_callback()

    # Use secure context manager to ensure password is cleared after use
    with get_ssh_credentials_secure() as creds:
        if creds:
            # Use password authentication
            sid = _current_session_id or "unknown"
            logger.info(
                f"Creating HPC connection with password auth (session {sid[:8]}...)"
            )
            return HPCConnection(
                host_alias=host_alias,
                host=creds["host"],
                user=creds["user"],
                password=creds["password"],
                port=creds["port"],
                progress_callback=progress_callback,
            )
        else:
            # Use SSH config (original behavior)
            logger.info(f"Creating HPC connection with SSH config for {host_alias}")
            return HPCConnection(
                host_alias=host_alias, progress_callback=progress_callback
            )


@tool
def test_hpc_connection(host_alias: str | None = None) -> dict[str, Any]:
    """
    Test SSH connection to HPC cluster.

    This tool verifies that the SSH connection to the HPC cluster is working
    by running a simple 'pwd' command on the remote server.

    Args:
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env

    Returns:
        Dictionary with connection status and remote directory
    """
    if host_alias is None:
        host_alias = config.hpc_host_alias

    try:
        hpc = _create_hpc_connection(host_alias)
        output = hpc.run_command("pwd")
    except Exception as e:
        return {
            "status": "failed",
            "message": f"SSH connection failed: {e}",
            "error": str(e),
        }
    else:
        auth_mode = "password" if get_ssh_credentials() else "SSH config"
        return {
            "status": "success",
            "message": f"SSH connection successful to {host_alias} ({auth_mode})",
            "remote_directory": output,
        }


@tool
def submit_slurm_job(
    slurm_file: str,
    host_alias: str | None = None,
    remote_dir: str = "~/slurm_jobs",
) -> dict[str, Any]:
    """
    Submit a SLURM job to HPC cluster.

    This tool transfers a SLURM script to the HPC cluster and submits it
    using the sbatch command.

    Args:
        slurm_file: Path to local SLURM script file
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env
        remote_dir: Remote directory to store scripts (default: ~/slurm_jobs)

    Returns:
        Dictionary with submission status and job ID
    """
    if host_alias is None:
        host_alias = config.hpc_host_alias

    slurm_path = Path(slurm_file)
    if not slurm_path.exists():
        return {
            "status": "failed",
            "message": f"SLURM file not found: {slurm_file}",
        }

    try:
        hpc = _create_hpc_connection(host_alias)
        job_id = hpc.submit_job(slurm_file, remote_dir=remote_dir)
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Failed to submit job: {e}",
            "error": str(e),
        }
    else:
        return {
            "status": "success",
            "message": "Job submitted successfully",
            "job_id": job_id,
            "remote_file": f"{remote_dir}/{slurm_path.name}",
        }


@tool
def get_slurm_job_status(
    job_id: str,
    host_alias: str | None = None,
) -> dict[str, Any]:
    """
    Get the status of a SLURM job.

    This tool checks the status of a submitted SLURM job using the squeue command.

    Args:
        job_id: SLURM job ID
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env

    Returns:
        Dictionary with job status information
    """
    if host_alias is None:
        host_alias = config.hpc_host_alias

    try:
        hpc = _create_hpc_connection(host_alias)
        status = hpc.get_job_status(job_id)
    except Exception as e:
        return {
            "status": "failed",
            "job_id": job_id,
            "message": f"Failed to get job status: {e}",
            "error": str(e),
        }
    else:
        return {
            "status": "success",
            "job_id": job_id,
            "status_output": status,
        }


@tool
def cancel_slurm_job(
    job_id: str,
    host_alias: str | None = None,
) -> dict[str, Any]:
    """
    Cancel a running SLURM job.

    This tool cancels a submitted SLURM job using the scancel command.

    Args:
        job_id: SLURM job ID
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env

    Returns:
        Dictionary with cancellation status
    """
    if host_alias is None:
        host_alias = config.hpc_host_alias

    try:
        hpc = _create_hpc_connection(host_alias)
        hpc.cancel_job(job_id)
    except Exception as e:
        return {
            "status": "failed",
            "job_id": job_id,
            "message": f"Failed to cancel job: {e}",
            "error": str(e),
        }
    else:
        return {
            "status": "success",
            "message": f"Job {job_id} cancelled successfully",
            "job_id": job_id,
        }


@tool
def download_job_outputs(
    job_id: str,
    host_alias: str | None = None,
    remote_dir: str = "~/slurm_jobs",
    local_dir: str = "outputs",
) -> dict[str, Any]:
    """
    Download job output files from HPC cluster.

    This tool downloads the .out and .err output files from a completed
    SLURM job to the local machine.

    Args:
        job_id: SLURM job ID
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env
        remote_dir: Remote directory where outputs are stored (default: ~/slurm_jobs)
        local_dir: Local directory to download to (default: outputs)

    Returns:
        Dictionary with download status and file locations
    """
    if host_alias is None:
        host_alias = config.hpc_host_alias

    try:
        hpc = _create_hpc_connection(host_alias)
        hpc.get_job_output(job_id, remote_dir=remote_dir, local_dir=local_dir)
    except Exception as e:
        return {
            "status": "failed",
            "job_id": job_id,
            "message": f"Failed to download outputs: {e}",
            "error": str(e),
        }
    else:
        # Build list of downloaded files
        output_dir = Path(local_dir)
        err_files = list(output_dir.glob(f"*{job_id}.err"))
        out_files = list(output_dir.glob(f"*{job_id}.out"))

        file_list = [str(f) for f in err_files + out_files]

        message = f"Job outputs downloaded successfully to {local_dir}/"
        if file_list:
            message += "\n\nDownloaded files:\n" + "\n".join(
                f"- {Path(f).name}" for f in file_list
            )

        return {
            "status": "success",
            "message": message,
            "job_id": job_id,
            "local_directory": local_dir,
            "downloaded_files": file_list,
        }
