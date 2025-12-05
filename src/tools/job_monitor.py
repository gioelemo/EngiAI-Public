"""
Job monitoring and notification system for HPC SLURM jobs.

This module provides tools for monitoring job status and getting notifications
when jobs complete. It supports:
1. Email notifications via SLURM (configured in job script)
2. Polling-based monitoring with status change detection
3. Background monitoring for long-running jobs
"""

import logging
import time
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from config import config
from src.tools.connection import HPCConnection
from src.tools.hpc import download_job_outputs

logger = logging.getLogger(__name__)

# Global job status cache for tracking changes
_job_status_cache: dict[str, dict[str, Any]] = {}


@tool
def monitor_job_until_complete(
    job_id: str,
    host_alias: str | None = None,
    check_interval: int = 60,
    max_checks: int = 100,
    auto_download: bool = True,
) -> dict[str, Any]:
    """
    Monitor a SLURM job until it completes, then optionally download outputs.

    This tool periodically checks the job status and returns when the job
    finishes (completes, fails, or is cancelled). Useful for waiting on
    job completion before proceeding with next steps.

    Args:
        job_id: SLURM job ID to monitor
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env
        check_interval: Seconds between status checks (default: 60)
        max_checks: Maximum number of checks before giving up (default: 100)
        auto_download: Automatically download outputs when job completes (default: True)

    Returns:
        Dictionary with final job status and monitoring information
    """
    if host_alias is None:
        host_alias = config.hpc_host_alias

    try:
        hpc = HPCConnection(host_alias=host_alias)
        start_time = datetime.now()
        checks_performed = 0

        logger.info(f"Starting to monitor job {job_id}")
        logger.debug(f"Check interval: {check_interval}s, Max checks: {max_checks}")

        while checks_performed < max_checks:
            checks_performed += 1
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()

            # Check job status
            status_output = hpc.get_job_status(job_id)

            # Check if job is completed
            # Job is completed if:
            # 1. Output contains "not found" or "completed"
            # 2. Output is empty or only contains header line (no job data)
            is_completed = False

            if (
                "not found" in status_output.lower()
                or "completed" in status_output.lower()
            ):
                is_completed = True
            else:
                # Check if output only contains the squeue header (no actual job data)
                lines = [
                    line.strip()
                    for line in status_output.strip().split("\n")
                    if line.strip()
                ]
                # If only 1 line (header) or empty, job is not in queue anymore
                if len(lines) <= 1:
                    is_completed = True

            if is_completed:
                logger.info(f"Job {job_id} has completed!")
                logger.debug(
                    f"Total monitoring time: {elapsed:.0f}s ({checks_performed} checks)"
                )

                result = {
                    "status": "completed",
                    "job_id": job_id,
                    "message": f"Job {job_id} completed after {elapsed:.0f}s",
                    "monitoring_time_seconds": elapsed,
                    "checks_performed": checks_performed,
                }

                # Auto-download outputs if requested
                if auto_download:
                    logger.info("Auto-downloading job outputs...")
                    download_result = download_job_outputs.invoke(
                        {
                            "job_id": job_id,
                            "host_alias": host_alias,
                        }
                    )
                    result["download_result"] = download_result

                return result

            # Job still running or pending - show the actual status
            # Extract job state from squeue output if available
            job_state = "in queue"
            lines = status_output.strip().split("\n")
            if len(lines) > 1:
                # Try to extract the state from the output (typically in column with ST or STATE)
                parts = lines[1].split()
                # ST column is typically the 5th column (index 4)
                squeue_state_column_index = 4
                if len(parts) > squeue_state_column_index:
                    job_state = parts[squeue_state_column_index]

            logger.debug(
                f"Check {checks_performed}/{max_checks}: Job {job_state} (elapsed: {elapsed:.0f}s)"
            )

            # Wait before next check (unless it's the last check)
            if checks_performed < max_checks:
                time.sleep(check_interval)

    except Exception as e:
        return {
            "status": "failed",
            "job_id": job_id,
            "message": f"Error monitoring job: {e}",
            "error": str(e),
        }
    else:
        # Max checks reached without completion
        return {
            "status": "timeout",
            "job_id": job_id,
            "message": f"Monitoring stopped after {max_checks} checks ({elapsed:.0f}s). Job may still be running.",
            "monitoring_time_seconds": elapsed,
            "checks_performed": checks_performed,
        }


@tool
def check_job_status_change(
    job_id: str,
    host_alias: str | None = None,
) -> dict[str, Any]:
    """
    Check if a job's status has changed since last check.

    This tool tracks job status changes and only reports when the status
    actually changes (e.g., from RUNNING to COMPLETED). Useful for
    getting notifications without constant polling.

    Args:
        job_id: SLURM job ID to check
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env

    Returns:
        Dictionary with status change information
    """
    if host_alias is None:
        host_alias = config.hpc_host_alias

    try:
        hpc = HPCConnection(host_alias=host_alias)
        current_status = hpc.get_job_status(job_id)

        # Create unique key for this job
        cache_key = f"{host_alias}:{job_id}"

        # Get previous status from cache
        previous_data = _job_status_cache.get(cache_key, {})
        previous_status = previous_data.get("status", None)
        previous_check_time = previous_data.get("timestamp", None)

        # Determine if job completed using same logic as monitor_job_until_complete
        is_completed = False

        if (
            "not found" in current_status.lower()
            or "completed" in current_status.lower()
        ):
            is_completed = True
        else:
            # Check if output only contains the squeue header (no actual job data)
            lines = [
                line.strip()
                for line in current_status.strip().split("\n")
                if line.strip()
            ]
            # If only 1 line (header) or empty, job is not in queue anymore
            if len(lines) <= 1:
                is_completed = True

        # Update cache
        _job_status_cache[cache_key] = {
            "status": current_status,
            "timestamp": datetime.now().isoformat(),
            "completed": is_completed,
        }

        # Detect status change
        status_changed = previous_status != current_status if previous_status else False

        result = {
            "job_id": job_id,
            "current_status": current_status,
            "previous_status": previous_status,
            "status_changed": status_changed,
            "is_completed": is_completed,
            "previous_check_time": previous_check_time,
        }

        if status_changed:
            if is_completed:
                result["message"] = f"🎉 Job {job_id} has COMPLETED!"
                result["notification"] = "JOB_COMPLETED"
            else:
                result["message"] = (
                    f"Status changed from '{previous_status}' to '{current_status}'"
                )
                result["notification"] = "STATUS_CHANGED"
        else:
            result["message"] = "No status change detected"
            result["notification"] = None

    except Exception as e:
        return {
            "status": "failed",
            "job_id": job_id,
            "message": f"Error checking job status: {e}",
            "error": str(e),
        }
    else:
        return result


@tool
def get_active_jobs_summary(
    host_alias: str | None = None,
) -> dict[str, Any]:
    """
    Get a summary of all currently running/pending jobs for the user.

    This tool shows all active jobs in the queue, useful for checking
    which jobs are still running before expecting notifications.

    Args:
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env

    Returns:
        Dictionary with active jobs information
    """
    if host_alias is None:
        host_alias = config.hpc_host_alias

    try:
        hpc = HPCConnection(host_alias=host_alias)
        # Get all user jobs (squeue without -j shows user's jobs)
        output = hpc.run_command("squeue -u $USER")

        # Count lines (minus header)
        lines = output.strip().split("\n")
        job_count = len(lines) - 1 if len(lines) > 1 else 0

    except Exception as e:
        return {
            "status": "failed",
            "message": f"Error getting jobs summary: {e}",
            "error": str(e),
        }
    else:
        return {
            "status": "success",
            "active_job_count": job_count,
            "jobs_output": output,
            "message": f"Found {job_count} active job(s) in the queue",
        }
