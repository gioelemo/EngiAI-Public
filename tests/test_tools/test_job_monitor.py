"""
Tests for job monitoring tools.

These tests cover SLURM job monitoring, status tracking, and notification functionality.
All tests use mocking to avoid requiring actual HPC connections.
"""

from unittest.mock import Mock, patch

import pytest

from src.tools.job_monitor import (
    _job_status_cache,
    check_job_status_change,
    get_active_jobs_summary,
    monitor_job_until_complete,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_job_cache():
    """Reset job status cache before each test."""
    _job_status_cache.clear()
    yield
    _job_status_cache.clear()


@pytest.fixture
def mock_hpc_connection():
    """Create a mock HPC connection."""
    with patch("src.tools.job_monitor.HPCConnection") as mock_conn_class:
        mock_conn = Mock()
        mock_conn_class.return_value = mock_conn
        yield mock_conn


@pytest.fixture(autouse=True)
def _mock_config():
    """Mock config with HPC host alias - autouse fixture."""
    with patch("src.tools.job_monitor.config") as mock_cfg:
        mock_cfg.hpc_host_alias = "euler"
        mock_cfg.slurm_email_user = "user@example.com"
        yield mock_cfg


# ============================================================================
# MONITOR JOB UNTIL COMPLETE TESTS
# ============================================================================


@pytest.mark.unit
def test_monitor_job_immediate_completion(mock_hpc_connection):
    """Test monitoring a job that completes immediately."""
    # Job is already completed (not found in queue)
    mock_hpc_connection.get_job_status.return_value = "Job not found"

    with patch("src.tools.job_monitor.download_job_outputs") as mock_download:
        mock_download.invoke.return_value = {"status": "success"}

        result = monitor_job_until_complete.invoke(
            {"job_id": "12345", "host_alias": "euler", "auto_download": True}
        )

    assert result["status"] == "completed"
    assert result["job_id"] == "12345"
    assert result["checks_performed"] == 1
    assert "download_result" in result
    mock_download.invoke.assert_called_once()


@pytest.mark.unit
def test_monitor_job_multiple_checks(mock_hpc_connection):
    """Test monitoring a job that completes after several checks."""
    # First 2 checks: job is running, 3rd check: job completed
    mock_hpc_connection.get_job_status.side_effect = [
        "JOBID USER STATE\n12345 user RUNNING",  # Check 1
        "JOBID USER STATE\n12345 user RUNNING",  # Check 2
        "Job 12345 not found",  # Check 3 - completed
    ]

    with (
        patch("src.tools.job_monitor.time.sleep"),
        patch("src.tools.job_monitor.download_job_outputs") as mock_download,
    ):
        mock_download.invoke.return_value = {"status": "success"}

        result = monitor_job_until_complete.invoke(
            {
                "job_id": "12345",
                "host_alias": "euler",
                "check_interval": 1,
                "auto_download": False,
            }
        )

    assert result["status"] == "completed"
    assert result["checks_performed"] == 3
    assert "download_result" not in result  # auto_download=False


@pytest.mark.unit
def test_monitor_job_timeout(mock_hpc_connection):
    """Test monitoring timeout when max checks reached."""
    # Job never completes
    mock_hpc_connection.get_job_status.return_value = (
        "JOBID USER STATE\n12345 user RUNNING"
    )

    with patch("src.tools.job_monitor.time.sleep"):
        result = monitor_job_until_complete.invoke(
            {
                "job_id": "12345",
                "host_alias": "euler",
                "check_interval": 1,
                "max_checks": 3,
            }
        )

    assert result["status"] == "timeout"
    assert result["checks_performed"] == 3
    assert "still be running" in result["message"]


@pytest.mark.unit
def test_monitor_job_completed_status(mock_hpc_connection):
    """Test detection of 'completed' in status output."""
    mock_hpc_connection.get_job_status.return_value = "Job 12345 completed successfully"

    with patch("src.tools.job_monitor.download_job_outputs") as mock_download:
        mock_download.invoke.return_value = {"status": "success"}

        result = monitor_job_until_complete.invoke(
            {"job_id": "12345", "auto_download": False}
        )

    assert result["status"] == "completed"
    assert result["checks_performed"] == 1


@pytest.mark.unit
def test_monitor_job_empty_queue_output(mock_hpc_connection):
    """Test detection of job completion via empty queue output."""
    # Only header line, no job data
    mock_hpc_connection.get_job_status.return_value = "JOBID USER STATE"

    with patch("src.tools.job_monitor.download_job_outputs") as mock_download:
        mock_download.invoke.return_value = {"status": "success"}

        result = monitor_job_until_complete.invoke(
            {"job_id": "12345", "auto_download": False}
        )

    assert result["status"] == "completed"


@pytest.mark.unit
def test_monitor_job_error_handling(mock_hpc_connection):
    """Test error handling during monitoring."""
    mock_hpc_connection.get_job_status.side_effect = RuntimeError("Connection failed")

    result = monitor_job_until_complete.invoke({"job_id": "12345"})

    assert result["status"] == "failed"
    assert "error" in result
    assert "Connection failed" in result["error"]


@pytest.mark.unit
def test_monitor_job_default_host_alias(mock_hpc_connection, _mock_config):
    """Test using default host alias from config."""
    mock_hpc_connection.get_job_status.return_value = "Job not found"
    _mock_config.hpc_host_alias = "default_host"

    with patch("src.tools.job_monitor.download_job_outputs") as mock_download:
        mock_download.invoke.return_value = {"status": "success"}

        result = monitor_job_until_complete.invoke(
            {"job_id": "12345", "auto_download": False}
        )

    assert result["status"] == "completed"


# ============================================================================
# CHECK JOB STATUS CHANGE TESTS
# ============================================================================


@pytest.mark.unit
def test_check_status_change_first_check(mock_hpc_connection):
    """Test first status check with no previous data."""
    mock_hpc_connection.get_job_status.return_value = (
        "JOBID USER STATE\n12345 user RUNNING"
    )

    result = check_job_status_change.invoke({"job_id": "12345", "host_alias": "euler"})

    assert result["job_id"] == "12345"
    assert result["status_changed"] is False
    assert result["previous_status"] is None
    assert result["is_completed"] is False


@pytest.mark.unit
def test_check_status_change_no_change(mock_hpc_connection):
    """Test when status hasn't changed."""
    status_output = "JOBID USER STATE\n12345 user RUNNING"
    mock_hpc_connection.get_job_status.return_value = status_output

    # First check
    result1 = check_job_status_change.invoke({"job_id": "12345"})
    assert result1["status_changed"] is False

    # Second check with same status
    result2 = check_job_status_change.invoke({"job_id": "12345"})
    assert result2["status_changed"] is False
    assert result2["notification"] is None


@pytest.mark.unit
def test_check_status_change_detected(mock_hpc_connection):
    """Test detection of status change."""
    mock_hpc_connection.get_job_status.side_effect = [
        "JOBID USER STATE\n12345 user RUNNING",
        "JOBID USER STATE\n12345 user PENDING",
    ]

    # First check
    result1 = check_job_status_change.invoke({"job_id": "12345"})
    assert result1["status_changed"] is False

    # Second check with different status
    result2 = check_job_status_change.invoke({"job_id": "12345"})
    assert result2["status_changed"] is True
    assert result2["notification"] == "STATUS_CHANGED"


@pytest.mark.unit
def test_check_status_change_completion(mock_hpc_connection):
    """Test detection of job completion."""
    mock_hpc_connection.get_job_status.side_effect = [
        "JOBID USER STATE\n12345 user RUNNING",
        "Job 12345 not found",
    ]

    # First check - running
    result1 = check_job_status_change.invoke({"job_id": "12345"})
    assert result1["is_completed"] is False

    # Second check - completed
    result2 = check_job_status_change.invoke({"job_id": "12345"})
    assert result2["is_completed"] is True
    assert result2["status_changed"] is True
    assert result2["notification"] == "JOB_COMPLETED"
    assert "COMPLETED" in result2["message"]


@pytest.mark.unit
def test_check_status_cache_isolation(mock_hpc_connection):
    """Test that different jobs have isolated cache entries."""
    mock_hpc_connection.get_job_status.return_value = (
        "JOBID USER STATE\n{} user RUNNING"
    )

    # Check two different jobs
    result1 = check_job_status_change.invoke({"job_id": "12345"})
    result2 = check_job_status_change.invoke({"job_id": "67890"})

    # Both should be first checks (no previous status)
    assert result1["previous_status"] is None
    assert result2["previous_status"] is None


@pytest.mark.unit
def test_check_status_error_handling(mock_hpc_connection):
    """Test error handling in status change check."""
    mock_hpc_connection.get_job_status.side_effect = RuntimeError("Connection error")

    result = check_job_status_change.invoke({"job_id": "12345"})

    assert result["status"] == "failed"
    assert "error" in result


# ============================================================================
# GET ACTIVE JOBS SUMMARY TESTS
# ============================================================================


@pytest.mark.unit
def test_get_active_jobs_summary_multiple_jobs(mock_hpc_connection):
    """Test getting summary with multiple active jobs."""
    mock_hpc_connection.run_command.return_value = """JOBID USER STATE
12345 user RUNNING
67890 user PENDING
11111 user RUNNING"""

    result = get_active_jobs_summary.invoke({"host_alias": "euler"})

    assert result["status"] == "success"
    assert result["active_job_count"] == 3
    assert "12345" in result["jobs_output"]


@pytest.mark.unit
def test_get_active_jobs_summary_no_jobs(mock_hpc_connection):
    """Test getting summary with no active jobs."""
    mock_hpc_connection.run_command.return_value = "JOBID USER STATE"

    result = get_active_jobs_summary.invoke({})

    assert result["status"] == "success"
    assert result["active_job_count"] == 0
    assert "0 active job" in result["message"]


@pytest.mark.unit
def test_get_active_jobs_summary_error(mock_hpc_connection):
    """Test error handling in jobs summary."""
    mock_hpc_connection.run_command.side_effect = RuntimeError("squeue failed")

    result = get_active_jobs_summary.invoke({"host_alias": "euler"})

    assert result["status"] == "failed"
    assert "error" in result


@pytest.mark.unit
def test_get_active_jobs_summary_default_host(mock_hpc_connection, _mock_config):
    """Test using default host alias."""
    mock_hpc_connection.run_command.return_value = (
        "JOBID USER STATE\n12345 user RUNNING"
    )
    _mock_config.hpc_host_alias = "default_host"

    result = get_active_jobs_summary.invoke({})

    assert result["status"] == "success"
    assert result["active_job_count"] == 1


# ============================================================================
# GENERATE SLURM SCRIPT WITH NOTIFICATIONS TESTS
# ============================================================================


@pytest.mark.unit
def test_generate_script_add_notifications(tmp_path):
    """Test adding email notifications to a script."""


# ============================================================================
# EDGE CASES AND INTEGRATION
# ============================================================================


@pytest.mark.unit
def test_job_status_cache_structure():
    """Test that job status cache has expected structure."""
    assert isinstance(_job_status_cache, dict)
    # Initially empty
    assert len(_job_status_cache) == 0


@pytest.mark.unit
def test_monitor_job_state_extraction(mock_hpc_connection):
    """Test extraction of job state from squeue output."""
    # Realistic squeue output with state column
    mock_hpc_connection.get_job_status.side_effect = [
        "JOBID USER STATE TIME\n12345 user PD 0:00",  # PENDING
        "JOBID USER STATE TIME\n12345 user R 1:23",  # RUNNING
        "Job not found",  # COMPLETED
    ]

    with (
        patch("src.tools.job_monitor.time.sleep"),
        patch("src.tools.job_monitor.download_job_outputs") as mock_download,
    ):
        mock_download.invoke.return_value = {"status": "success"}

        result = monitor_job_until_complete.invoke(
            {"job_id": "12345", "check_interval": 1, "auto_download": False}
        )

    assert result["status"] == "completed"
    assert result["checks_performed"] == 3


@pytest.mark.unit
def test_check_status_preserves_timestamp(mock_hpc_connection):
    """Test that timestamp is stored in cache."""
    mock_hpc_connection.get_job_status.return_value = (
        "JOBID USER STATE\n12345 user RUNNING"
    )

    # First check
    check_job_status_change.invoke({"job_id": "12345"})

    # Verify cache has timestamp
    cache_key = "euler:12345"
    assert cache_key in _job_status_cache
    assert "timestamp" in _job_status_cache[cache_key]
    assert isinstance(_job_status_cache[cache_key]["timestamp"], str)


@pytest.mark.unit
def test_multiple_jobs_in_cache(mock_hpc_connection):
    """Test tracking multiple jobs simultaneously."""
    mock_hpc_connection.get_job_status.return_value = (
        "JOBID USER STATE\n{} user RUNNING"
    )

    # Check multiple jobs
    for job_id in ["12345", "67890", "11111"]:
        check_job_status_change.invoke({"job_id": job_id})

    # All should be in cache
    assert len(_job_status_cache) == 3
    assert "euler:12345" in _job_status_cache
    assert "euler:67890" in _job_status_cache
    assert "euler:11111" in _job_status_cache
