"""
HPC cluster job management tools for SLURM job submission and monitoring.

This module provides tools for interacting with HPC clusters via SSH,
including job submission, status monitoring, cancellation, and output retrieval.
"""

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.tools.connection import HPCConnection


@tool
def test_hpc_connection(host_alias: str = "euler") -> dict[str, Any]:
    """
    Test SSH connection to HPC cluster.

    This tool verifies that the SSH connection to the HPC cluster is working
    by running a simple 'pwd' command on the remote server.

    Args:
        host_alias: Host alias from ~/.ssh/config (e.g., "euler")

    Returns:
        Dictionary with connection status and remote directory
    """
    try:
        hpc = HPCConnection(host_alias=host_alias)
        output = hpc.run_command("pwd")
    except Exception as e:
        return {
            "status": "failed",
            "message": f"SSH connection failed: {e}",
            "error": str(e),
        }
    else:
        return {
            "status": "success",
            "message": f"SSH connection successful to {host_alias}",
            "remote_directory": output,
        }


@tool
def submit_slurm_job(
    slurm_file: str,
    host_alias: str = "euler",
    remote_dir: str = "~/slurm_jobs",
) -> dict[str, Any]:
    """
    Submit a SLURM job to HPC cluster.

    This tool transfers a SLURM script to the HPC cluster and submits it
    using the sbatch command.

    Args:
        slurm_file: Path to local SLURM script file
        host_alias: Host alias from ~/.ssh/config (e.g., "euler")
        remote_dir: Remote directory to store scripts (default: ~/slurm_jobs)

    Returns:
        Dictionary with submission status and job ID
    """
    slurm_path = Path(slurm_file)
    if not slurm_path.exists():
        return {
            "status": "failed",
            "message": f"SLURM file not found: {slurm_file}",
        }

    try:
        hpc = HPCConnection(host_alias=host_alias)
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
    host_alias: str = "euler",
) -> dict[str, Any]:
    """
    Get the status of a SLURM job.

    This tool checks the status of a submitted SLURM job using the squeue command.

    Args:
        job_id: SLURM job ID
        host_alias: Host alias from ~/.ssh/config (e.g., "euler")

    Returns:
        Dictionary with job status information
    """
    try:
        hpc = HPCConnection(host_alias=host_alias)
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
    host_alias: str = "euler",
) -> dict[str, Any]:
    """
    Cancel a running SLURM job.

    This tool cancels a submitted SLURM job using the scancel command.

    Args:
        job_id: SLURM job ID
        host_alias: Host alias from ~/.ssh/config (e.g., "euler")

    Returns:
        Dictionary with cancellation status
    """
    try:
        hpc = HPCConnection(host_alias=host_alias)
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
    host_alias: str = "euler",
    remote_dir: str = "~/slurm_jobs",
    local_dir: str = "outputs",
) -> dict[str, Any]:
    """
    Download job output files from HPC cluster.

    This tool downloads the .out and .err output files from a completed
    SLURM job to the local machine.

    Args:
        job_id: SLURM job ID
        host_alias: Host alias from ~/.ssh/config (e.g., "euler")
        remote_dir: Remote directory where outputs are stored (default: ~/slurm_jobs)
        local_dir: Local directory to download to (default: outputs)

    Returns:
        Dictionary with download status and file locations
    """
    try:
        hpc = HPCConnection(host_alias=host_alias)
        hpc.get_job_output(job_id, remote_dir=remote_dir, local_dir=local_dir)
    except Exception as e:
        return {
            "status": "failed",
            "job_id": job_id,
            "message": f"Failed to download outputs: {e}",
            "error": str(e),
        }
    else:
        return {
            "status": "success",
            "message": "Job outputs downloaded successfully",
            "job_id": job_id,
            "local_directory": local_dir,
        }
