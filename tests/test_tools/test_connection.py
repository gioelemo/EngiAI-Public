"""
Tests for HPC connection module (SSH and SFTP operations).

These tests mock Fabric connections to ensure they run in CI without
requiring actual SSH access to an HPC cluster.
"""

from unittest.mock import Mock, patch

import pytest

from src.tools.connection import HPCConnection

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_fabric_connection():
    """Mock Fabric Connection object."""
    with patch("src.tools.connection.Connection") as mock_conn_class:
        mock_conn = Mock()
        mock_conn_class.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def temp_slurm_file(tmp_path):
    """Create a temporary SLURM script file."""
    slurm_file = tmp_path / "test_job.slurm"
    slurm_file.write_text("""#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=01:00:00

echo "Test"
""")
    return slurm_file


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


@pytest.mark.unit
def test_hpc_connection_init_success(mock_fabric_connection):
    """Test successful HPC connection initialization."""
    hpc = HPCConnection(host_alias="euler")

    assert hpc.host_alias == "euler"
    assert hpc.connection == mock_fabric_connection


@pytest.mark.unit
def test_hpc_connection_init_failure():
    """Test HPC connection initialization failure."""
    with patch(
        "src.tools.connection.Connection",
        side_effect=RuntimeError("Connection failed"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            HPCConnection(host_alias="invalid_host")

        assert "Failed to connect to 'invalid_host'" in str(exc_info.value)
        assert "Make sure:" in str(exc_info.value)


# ============================================================================
# RUN COMMAND TESTS
# ============================================================================


@pytest.mark.unit
def test_run_command_success(mock_fabric_connection):
    """Test running a successful command."""
    mock_result = Mock()
    mock_result.failed = False
    mock_result.stdout = "Command output\n"
    mock_fabric_connection.run.return_value = mock_result

    hpc = HPCConnection(host_alias="euler")
    output = hpc.run_command("pwd")

    assert output == "Command output"
    mock_fabric_connection.run.assert_called_once_with("pwd", hide=False, warn=False)


@pytest.mark.unit
def test_run_command_failure(mock_fabric_connection):
    """Test running a failed command."""
    mock_result = Mock()
    mock_result.failed = True
    mock_result.stderr = "Command not found"
    mock_fabric_connection.run.return_value = mock_result

    hpc = HPCConnection(host_alias="euler")

    with pytest.raises(RuntimeError) as exc_info:
        hpc.run_command("invalid_command")

    assert "SSH command failed" in str(exc_info.value)
    assert "Command not found" in str(exc_info.value)


# ============================================================================
# PUT FILE TESTS
# ============================================================================


@pytest.mark.unit
def test_put_file_success(mock_fabric_connection, temp_slurm_file):
    """Test successful file transfer."""
    mock_fabric_connection.run.return_value = Mock(stdout="/home/user", failed=False)

    hpc = HPCConnection(host_alias="euler")
    hpc.put_file(str(temp_slurm_file), "~/remote_file.slurm")

    # Verify put was called
    mock_fabric_connection.put.assert_called_once()
    call_args = mock_fabric_connection.put.call_args[0]
    assert str(temp_slurm_file) in call_args[0]


@pytest.mark.unit
def test_put_file_not_found():
    """Test file transfer with nonexistent file."""
    with patch("src.tools.connection.Connection"):
        hpc = HPCConnection(host_alias="euler")

        with pytest.raises(FileNotFoundError) as exc_info:
            hpc.put_file("/nonexistent/file.txt", "~/remote.txt")

        assert "Local file not found" in str(exc_info.value)


@pytest.mark.unit
def test_put_file_transfer_failure(mock_fabric_connection, temp_slurm_file):
    """Test handling of file transfer failure."""
    mock_fabric_connection.run.return_value = Mock(stdout="/home/user", failed=False)
    mock_fabric_connection.put.side_effect = Exception("Permission denied")

    hpc = HPCConnection(host_alias="euler")

    with pytest.raises(RuntimeError) as exc_info:
        hpc.put_file(str(temp_slurm_file), "~/remote.txt")

    assert "SFTP transfer failed" in str(exc_info.value)


@pytest.mark.unit
def test_put_file_with_absolute_path(mock_fabric_connection, temp_slurm_file):
    """Test file transfer with absolute remote path."""
    hpc = HPCConnection(host_alias="euler")
    hpc.put_file(str(temp_slurm_file), "/scratch/user/file.slurm")

    mock_fabric_connection.put.assert_called_once()
    call_args = mock_fabric_connection.put.call_args[0]
    assert "/scratch/user/file.slurm" in call_args[1]


# ============================================================================
# GET FILE TESTS
# ============================================================================


@pytest.mark.unit
def test_get_file_success(mock_fabric_connection, tmp_path):
    """Test successful file download."""
    local_path = tmp_path / "downloaded.txt"

    hpc = HPCConnection(host_alias="euler")
    hpc.get_file("~/remote_file.txt", str(local_path))

    mock_fabric_connection.get.assert_called_once()


@pytest.mark.unit
def test_get_file_failure(mock_fabric_connection, tmp_path):
    """Test handling of file download failure."""
    mock_fabric_connection.get.side_effect = RuntimeError(
        "SFTP download failed: File not found on remote"
    )
    local_path = tmp_path / "downloaded.txt"

    hpc = HPCConnection(host_alias="euler")

    with pytest.raises(RuntimeError) as exc_info:
        hpc.get_file("~/nonexistent.txt", str(local_path))

    assert "SFTP download failed" in str(exc_info.value)


# ============================================================================
# SUBMIT JOB TESTS
# ============================================================================


@pytest.mark.unit
def test_submit_job_success(mock_fabric_connection, temp_slurm_file):
    """Test successful job submission."""
    # Mock remote mkdir and put
    mock_fabric_connection.run.side_effect = [
        Mock(stdout="/home/user", failed=False),  # echo $HOME
        Mock(stdout="", failed=False),  # mkdir
        Mock(stdout="Submitted batch job 12345\n", failed=False),  # sbatch
    ]

    hpc = HPCConnection(host_alias="euler")
    job_id = hpc.submit_job(str(temp_slurm_file), remote_dir="~/jobs")

    assert job_id == "12345"
    # Verify sbatch was called
    assert any(
        "sbatch" in str(call) for call in mock_fabric_connection.run.call_args_list
    )


@pytest.mark.unit
def test_submit_job_invalid_file():
    """Test job submission with nonexistent file."""
    hpc = HPCConnection(host_alias="euler")

    with pytest.raises(FileNotFoundError):
        hpc.submit_job("/nonexistent/job.slurm")


@pytest.mark.unit
def test_submit_job_sbatch_failure(mock_fabric_connection, temp_slurm_file):
    """Test handling of sbatch failure."""
    mock_fabric_connection.run.side_effect = [
        Mock(stdout="/home/user", failed=False),  # echo $HOME
        Mock(stdout="", failed=False),  # mkdir
        Mock(stdout="", stderr="sbatch: error", failed=True),  # sbatch fails
    ]

    hpc = HPCConnection(host_alias="euler")

    with pytest.raises(RuntimeError) as exc_info:
        hpc.submit_job(str(temp_slurm_file))

    assert "sbatch failed" in str(exc_info.value) or "SSH command failed" in str(
        exc_info.value
    )


# ============================================================================
# JOB STATUS TESTS
# ============================================================================


@pytest.mark.unit
def test_get_job_status_running(mock_fabric_connection):
    """Test getting status of a running job."""
    mock_result = Mock()
    mock_result.failed = False
    mock_result.stdout = "12345 RUNNING  test_job"
    mock_fabric_connection.run.return_value = mock_result

    hpc = HPCConnection(host_alias="euler")
    status = hpc.get_job_status("12345")

    assert "RUNNING" in status
    mock_fabric_connection.run.assert_called_with(
        "squeue -j 12345", hide=False, warn=False
    )


@pytest.mark.unit
def test_get_job_status_completed(mock_fabric_connection):
    """Test getting status of completed job."""
    mock_result = Mock()
    mock_result.failed = False
    mock_result.stdout = ""  # squeue returns empty for completed jobs
    mock_fabric_connection.run.return_value = mock_result

    hpc = HPCConnection(host_alias="euler")
    status = hpc.get_job_status("12345")

    assert "not found" in status or status == ""


# ============================================================================
# CANCEL JOB TESTS
# ============================================================================


@pytest.mark.unit
def test_cancel_job_success(mock_fabric_connection):
    """Test successful job cancellation."""
    mock_result = Mock()
    mock_result.failed = False
    mock_result.stdout = ""
    mock_fabric_connection.run.return_value = mock_result

    hpc = HPCConnection(host_alias="euler")
    hpc.cancel_job("12345")

    mock_fabric_connection.run.assert_called_with(
        "scancel 12345", hide=False, warn=False
    )


@pytest.mark.unit
def test_cancel_job_failure(mock_fabric_connection):
    """Test handling of cancellation failure."""
    mock_result = Mock()
    mock_result.failed = True
    mock_result.stderr = "Invalid job id specified"
    mock_fabric_connection.run.return_value = mock_result

    hpc = HPCConnection(host_alias="euler")

    with pytest.raises(RuntimeError) as exc_info:
        hpc.cancel_job("99999")

    assert "SSH command failed" in str(exc_info.value)


# ============================================================================
# JOB OUTPUT TESTS
# ============================================================================


@pytest.mark.unit
def test_get_job_output_success(mock_fabric_connection, tmp_path):
    """Test downloading job output files."""
    mock_fabric_connection.run.return_value = Mock(stdout="/home/user", failed=False)

    hpc = HPCConnection(host_alias="euler")
    hpc.get_job_output("12345", remote_dir="~/jobs", local_dir=str(tmp_path))

    # Verify get was called twice (for .out and .err files)
    assert mock_fabric_connection.get.call_count == 2


@pytest.mark.unit
def test_get_job_output_creates_local_dir(mock_fabric_connection, tmp_path):
    """Test that local directory is created if it doesn't exist."""
    local_dir = tmp_path / "new_output_dir"
    mock_fabric_connection.run.return_value = Mock(stdout="/home/user", failed=False)

    hpc = HPCConnection(host_alias="euler")
    hpc.get_job_output("12345", local_dir=str(local_dir))

    assert local_dir.exists()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.unit
def test_complete_job_workflow(mock_fabric_connection, temp_slurm_file, tmp_path):
    """Test complete workflow: submit, monitor, cancel, download."""

    # Create a generator that can be called multiple times
    def run_side_effect():
        responses = [
            Mock(stdout="/home/user", failed=False),  # home dir for submit
            Mock(stdout="", failed=False),  # mkdir
            Mock(stdout="Submitted batch job 12345\n", failed=False),  # sbatch
            Mock(stdout="12345 RUNNING test", failed=False),  # squeue
            Mock(stdout="", failed=False),  # scancel
            Mock(stdout="", failed=False),  # ls for .err files in get_job_output
            Mock(stdout="", failed=False),  # ls for .out files in get_job_output
        ]
        yield from responses

    response_gen = run_side_effect()
    mock_fabric_connection.run.side_effect = lambda *_args, **_kwargs: next(
        response_gen
    )

    hpc = HPCConnection(host_alias="euler")

    # Submit
    job_id = hpc.submit_job(str(temp_slurm_file))
    assert job_id == "12345"

    # Check status
    status = hpc.get_job_status(job_id)
    assert "RUNNING" in status

    # Cancel
    hpc.cancel_job(job_id)

    # Download outputs (will find no files, but should not error)
    hpc.get_job_output(job_id, local_dir=str(tmp_path))
    assert tmp_path.exists()
