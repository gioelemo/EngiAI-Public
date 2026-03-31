"""
Tests for HPC tools (SLURM job submission and monitoring).

These tests mock SSH connections to ensure they run in CI without requiring
actual HPC cluster access.
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.tools.hpc import (
    _decrypt_password,
    _encrypt_password,
    _session_credentials_store,
    cancel_slurm_job,
    cleanup_expired_credentials,
    clear_ssh_credentials,
    download_job_outputs,
    get_slurm_job_status,
    get_ssh_credentials,
    get_ssh_credentials_secure,
    set_current_session_id,
    set_ssh_credentials,
    submit_slurm_job,
    test_hpc_connection,
    validate_hostname,
    validate_username,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_hpc_connection():
    """Mock HPCConnection class for testing."""
    with patch("src.tools.hpc.HPCConnection") as mock_conn_class:
        mock_conn = Mock()
        mock_conn_class.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def temp_slurm_file(tmp_path):
    """Create a temporary SLURM script file."""
    slurm_file = tmp_path / "test_job.slurm"
    slurm_file.write_text("""#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --time=01:00:00
#SBATCH --ntasks=1

echo "Test job"
""")
    return str(slurm_file)


# ============================================================================
# TEST HPC CONNECTION
# ============================================================================


@pytest.mark.unit
def test_hpc_connection_success(mock_hpc_connection):
    """Test successful HPC connection."""
    mock_hpc_connection.run_command.return_value = "/home/user/work"

    result = test_hpc_connection.invoke({"host_alias": "euler"})

    assert result["status"] == "success"
    assert "SSH connection successful" in result["message"]
    assert result["remote_directory"] == "/home/user/work"
    mock_hpc_connection.run_command.assert_called_once_with("pwd")


@pytest.mark.unit
def test_hpc_connection_failure():
    """Test HPC connection failure."""
    with patch(
        "src.tools.hpc.HPCConnection",
        side_effect=RuntimeError("Connection refused"),
    ):
        result = test_hpc_connection.invoke({"host_alias": "euler"})

        assert result["status"] == "failed"
        assert "SSH connection failed" in result["message"]
        assert "Connection refused" in result["error"]


@pytest.mark.unit
def test_hpc_connection_uses_default_host():
    """Test that connection uses default host from config."""
    with patch("src.tools.hpc.HPCConnection") as mock_conn_class:
        mock_conn = Mock()
        mock_conn_class.return_value = mock_conn
        mock_conn.run_command.return_value = "/home/user"

        result = test_hpc_connection.invoke({})

        assert result["status"] == "success"


# ============================================================================
# SUBMIT SLURM JOB
# ============================================================================


@pytest.mark.unit
def test_submit_slurm_job_success(mock_hpc_connection, temp_slurm_file):
    """Test successful SLURM job submission."""
    mock_hpc_connection.submit_job.return_value = "12345"

    result = submit_slurm_job.invoke({"slurm_file": temp_slurm_file})

    assert result["status"] == "success"
    assert result["job_id"] == "12345"
    assert "Job submitted successfully" in result["message"]
    assert "remote_file" in result
    mock_hpc_connection.submit_job.assert_called_once()


@pytest.mark.unit
def test_submit_slurm_job_file_not_found():
    """Test submission with nonexistent file."""
    result = submit_slurm_job.invoke({"slurm_file": "/nonexistent/file.slurm"})

    assert result["status"] == "failed"
    assert "SLURM file not found" in result["message"]


@pytest.mark.unit
def test_submit_slurm_job_submission_failure(mock_hpc_connection, temp_slurm_file):
    """Test handling of submission failure."""
    mock_hpc_connection.submit_job.side_effect = RuntimeError(
        "sbatch: command not found"
    )

    result = submit_slurm_job.invoke({"slurm_file": temp_slurm_file})

    assert result["status"] == "failed"
    assert "Failed to submit job" in result["message"]
    assert "sbatch: command not found" in result["error"]


@pytest.mark.unit
def test_submit_slurm_job_custom_remote_dir(mock_hpc_connection, temp_slurm_file):
    """Test submission with custom remote directory."""
    mock_hpc_connection.submit_job.return_value = "67890"

    result = submit_slurm_job.invoke(
        {"slurm_file": temp_slurm_file, "remote_dir": "~/custom_jobs"}
    )

    assert result["status"] == "success"
    assert "custom_jobs" in result["remote_file"]


# ============================================================================
# GET SLURM JOB STATUS
# ============================================================================


@pytest.mark.unit
def test_get_job_status_success(mock_hpc_connection):
    """Test successful job status query."""
    mock_hpc_connection.get_job_status.return_value = "RUNNING"

    result = get_slurm_job_status.invoke({"job_id": "12345"})

    assert result["status"] == "success"
    assert result["job_id"] == "12345"
    assert result["status_output"] == "RUNNING"
    mock_hpc_connection.get_job_status.assert_called_once_with("12345")


@pytest.mark.unit
def test_get_job_status_completed(mock_hpc_connection):
    """Test status query for completed job."""
    mock_hpc_connection.get_job_status.return_value = "COMPLETED"

    result = get_slurm_job_status.invoke({"job_id": "67890"})

    assert result["status"] == "success"
    assert result["status_output"] == "COMPLETED"


@pytest.mark.unit
def test_get_job_status_failure(mock_hpc_connection):
    """Test handling of status query failure."""
    mock_hpc_connection.get_job_status.side_effect = RuntimeError("Job not found")

    result = get_slurm_job_status.invoke({"job_id": "99999"})

    assert result["status"] == "failed"
    assert "Failed to get job status" in result["message"]
    assert "Job not found" in result["error"]


# ============================================================================
# CANCEL SLURM JOB
# ============================================================================


@pytest.mark.unit
def test_cancel_job_success(mock_hpc_connection):
    """Test successful job cancellation."""
    mock_hpc_connection.cancel_job.return_value = None

    result = cancel_slurm_job.invoke({"job_id": "12345"})

    assert result["status"] == "success"
    assert "cancelled successfully" in result["message"]
    assert result["job_id"] == "12345"
    mock_hpc_connection.cancel_job.assert_called_once_with("12345")


@pytest.mark.unit
def test_cancel_job_failure(mock_hpc_connection):
    """Test handling of cancellation failure."""
    mock_hpc_connection.cancel_job.side_effect = RuntimeError("scancel: Invalid job id")

    result = cancel_slurm_job.invoke({"job_id": "99999"})

    assert result["status"] == "failed"
    assert "Failed to cancel job" in result["message"]
    assert "Invalid job id" in result["error"]


# ============================================================================
# DOWNLOAD JOB OUTPUTS
# ============================================================================


@pytest.mark.unit
def test_download_outputs_success(mock_hpc_connection, tmp_path):
    """Test successful output download."""
    # Create mock output files
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "test_job_12345.out").write_text("Job output")
    (output_dir / "test_job_12345.err").write_text("Job errors")

    mock_hpc_connection.get_job_output.return_value = None

    with patch("src.tools.hpc.Path") as mock_path_class:
        mock_path = Mock()
        mock_path.glob.side_effect = lambda pattern: (
            [output_dir / "test_job_12345.err"]
            if "err" in pattern
            else [output_dir / "test_job_12345.out"]
        )
        mock_path_class.return_value = mock_path

        result = download_job_outputs.invoke(
            {"job_id": "12345", "local_dir": str(output_dir)}
        )

        assert result["status"] == "success"
        assert "downloaded successfully" in result["message"]
        assert result["job_id"] == "12345"
        assert "downloaded_files" in result


@pytest.mark.unit
def test_download_outputs_failure(mock_hpc_connection):
    """Test handling of download failure."""
    mock_hpc_connection.get_job_output.side_effect = RuntimeError("File not found")

    result = download_job_outputs.invoke({"job_id": "12345"})

    assert result["status"] == "failed"
    assert "Failed to download outputs" in result["message"]
    assert "File not found" in result["error"]


@pytest.mark.unit
def test_download_outputs_custom_directories(mock_hpc_connection, tmp_path):
    """Test download with custom remote and local directories."""
    output_dir = tmp_path / "custom_outputs"
    output_dir.mkdir()

    mock_hpc_connection.get_job_output.return_value = None

    with patch("src.tools.hpc.Path") as mock_path_class:
        mock_path = Mock()
        mock_path.glob.return_value = []
        mock_path_class.return_value = mock_path

        result = download_job_outputs.invoke(
            {
                "job_id": "12345",
                "remote_dir": "~/custom_remote",
                "local_dir": str(output_dir),
            }
        )

        assert result["status"] == "success"
        mock_hpc_connection.get_job_output.assert_called_once_with(
            "12345", remote_dir="~/custom_remote", local_dir=str(output_dir)
        )


# ============================================================================
# INTEGRATION SCENARIOS
# ============================================================================


@pytest.mark.unit
def test_complete_job_workflow(mock_hpc_connection, temp_slurm_file, tmp_path):
    """Test complete workflow: submit -> status -> download."""
    # Setup mocks
    mock_hpc_connection.submit_job.return_value = "12345"
    mock_hpc_connection.get_job_status.return_value = "COMPLETED"
    mock_hpc_connection.get_job_output.return_value = None

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Submit job
    submit_result = submit_slurm_job.invoke({"slurm_file": temp_slurm_file})
    assert submit_result["status"] == "success"
    job_id = submit_result["job_id"]

    # Check status
    status_result = get_slurm_job_status.invoke({"job_id": job_id})
    assert status_result["status"] == "success"
    assert status_result["status_output"] == "COMPLETED"

    # Download outputs
    with patch("src.tools.hpc.Path") as mock_path_class:
        mock_path = Mock()
        mock_path.glob.return_value = []
        mock_path_class.return_value = mock_path

        download_result = download_job_outputs.invoke(
            {"job_id": job_id, "local_dir": str(output_dir)}
        )
        assert download_result["status"] == "success"


@pytest.mark.unit
def test_cancel_running_job(mock_hpc_connection, temp_slurm_file):
    """Test canceling a running job."""
    # Submit job
    mock_hpc_connection.submit_job.return_value = "12345"
    submit_result = submit_slurm_job.invoke({"slurm_file": temp_slurm_file})
    job_id = submit_result["job_id"]

    # Check it's running
    mock_hpc_connection.get_job_status.return_value = "RUNNING"
    status_result = get_slurm_job_status.invoke({"job_id": job_id})
    assert status_result["status_output"] == "RUNNING"

    # Cancel it
    mock_hpc_connection.cancel_job.return_value = None
    cancel_result = cancel_slurm_job.invoke({"job_id": job_id})
    assert cancel_result["status"] == "success"


# ============================================================================
# CREDENTIAL MANAGEMENT TESTS
# ============================================================================

SESSION_ID = "test-session-001"


@pytest.fixture(autouse=True)
def _clean_credential_state():
    """Reset credential store and session ID between tests."""
    _session_credentials_store.clear()
    set_current_session_id(None)
    yield
    _session_credentials_store.clear()
    set_current_session_id(None)


# --- Encryption ---


@pytest.mark.unit
def test_encrypt_decrypt_roundtrip():
    """Encrypt then decrypt must return original password."""
    password = "s3cret!Pass"
    encrypted = _encrypt_password(password)
    assert encrypted != password
    assert _decrypt_password(encrypted) == password


# --- Hostname validation ---


@pytest.mark.unit
@pytest.mark.parametrize(
    "hostname",
    ["euler.ethz.ch", "my-host", "a.b.c.d", "localhost", "h1"],
)
def test_validate_hostname_valid(hostname):
    assert validate_hostname(hostname) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "hostname",
    ["", ".start", "a" * 256, "host!@#", "-bad", "bad-", "a." + "x" * 64],
)
def test_validate_hostname_invalid(hostname):
    assert validate_hostname(hostname) is False


# --- Username validation ---


@pytest.mark.unit
@pytest.mark.parametrize("username", ["user", "user_name", "user-1", "a"])
def test_validate_username_valid(username):
    assert validate_username(username) is True


@pytest.mark.unit
@pytest.mark.parametrize("username", ["", "user name", "a" * 100, "root;rm", "u$er"])
def test_validate_username_invalid(username):
    assert validate_username(username) is False


# --- set_ssh_credentials ---


@pytest.mark.unit
def test_set_ssh_credentials_stores():
    """Full credential set stores and returns success."""
    set_current_session_id(SESSION_ID)
    ok, msg = set_ssh_credentials(
        host="euler.ethz.ch", user="testuser", password="pw123", port=22
    )
    assert ok is True
    assert "saved" in msg.lower() or "Credentials" in msg
    assert SESSION_ID in _session_credentials_store


@pytest.mark.unit
def test_set_ssh_credentials_clears_when_partial():
    """Passing None for required fields clears existing credentials."""
    set_current_session_id(SESSION_ID)
    set_ssh_credentials(host="euler.ethz.ch", user="testuser", password="pw123")
    assert SESSION_ID in _session_credentials_store

    ok, msg = set_ssh_credentials(host=None, user=None, password=None)
    assert ok is True
    assert "cleared" in msg.lower()
    assert SESSION_ID not in _session_credentials_store


@pytest.mark.unit
def test_set_ssh_credentials_no_session():
    """Without a session ID, set_ssh_credentials returns failure."""
    ok, msg = set_ssh_credentials(
        host="euler.ethz.ch", user="testuser", password="pw123"
    )
    assert ok is False
    assert "session" in msg.lower()


@pytest.mark.unit
def test_set_ssh_credentials_validates_hostname():
    """Invalid hostname must be rejected."""
    set_current_session_id(SESSION_ID)
    ok, msg = set_ssh_credentials(host="bad!host", user="testuser", password="pw123")
    assert ok is False
    assert "hostname" in msg.lower()


@pytest.mark.unit
def test_set_ssh_credentials_validates_port():
    """Out-of-range port must be rejected."""
    set_current_session_id(SESSION_ID)
    ok, msg = set_ssh_credentials(
        host="euler.ethz.ch", user="testuser", password="pw123", port=99999
    )
    assert ok is False
    assert "port" in msg.lower()


# --- clear_ssh_credentials ---


@pytest.mark.unit
def test_clear_ssh_credentials():
    """clear_ssh_credentials removes stored credentials."""
    set_current_session_id(SESSION_ID)
    set_ssh_credentials(host="euler.ethz.ch", user="testuser", password="pw123")
    assert SESSION_ID in _session_credentials_store
    clear_ssh_credentials(SESSION_ID)
    assert SESSION_ID not in _session_credentials_store


@pytest.mark.unit
def test_clear_ssh_credentials_noop():
    """Clearing non-existent credentials is a no-op."""
    clear_ssh_credentials("nonexistent-session")
    # Should not raise


# --- get_ssh_credentials ---


@pytest.mark.unit
def test_get_ssh_credentials_valid():
    """Stored credentials are returned with decrypted password."""
    set_current_session_id(SESSION_ID)
    set_ssh_credentials(host="euler.ethz.ch", user="testuser", password="pw123")
    creds = get_ssh_credentials(SESSION_ID)
    assert creds is not None
    assert creds["host"] == "euler.ethz.ch"
    assert creds["user"] == "testuser"
    assert creds["password"] == "pw123"
    assert creds["port"] == 22


@pytest.mark.unit
def test_get_ssh_credentials_expired():
    """Expired credentials return None and are cleaned up."""
    set_current_session_id(SESSION_ID)
    set_ssh_credentials(host="euler.ethz.ch", user="testuser", password="pw123")
    # Force expiration
    _session_credentials_store[SESSION_ID]["expires_at"] = time.time() - 1
    assert get_ssh_credentials(SESSION_ID) is None
    assert SESSION_ID not in _session_credentials_store


@pytest.mark.unit
def test_get_ssh_credentials_no_session():
    """No session ID returns None."""
    assert get_ssh_credentials(None) is None


# --- get_ssh_credentials_secure ---


@pytest.mark.unit
def test_get_ssh_credentials_secure_cleanup():
    """Context manager overwrites password after block exits."""
    set_current_session_id(SESSION_ID)
    set_ssh_credentials(host="euler.ethz.ch", user="testuser", password="pw123")
    with get_ssh_credentials_secure(SESSION_ID) as creds:
        assert creds is not None
        assert creds["password"] == "pw123"
    # After exiting, password should be zeroed/removed
    assert "password" not in creds


# --- cleanup_expired_credentials ---


@pytest.mark.unit
def test_cleanup_expired_credentials():
    """Expired sessions are removed, valid ones are kept."""
    set_current_session_id("s1")
    set_ssh_credentials(host="euler.ethz.ch", user="u1", password="p1", session_id="s1")
    set_ssh_credentials(host="euler.ethz.ch", user="u2", password="p2", session_id="s2")
    # Expire s1
    _session_credentials_store["s1"]["expires_at"] = time.time() - 1

    removed = cleanup_expired_credentials()
    assert removed == 1
    assert "s1" not in _session_credentials_store
    assert "s2" in _session_credentials_store
