"""
SSH connection helper for transferring SLURM scripts to HPC and submitting jobs.

Uses Fabric (https://www.fabfile.org/) for robust SSH connection management.
This module reads your SSH config file and provides utilities to:
1. Transfer generated SLURM scripts to a remote HPC cluster
2. Submit jobs via sbatch
3. Monitor job status
4. Download results

Usage:
    from src.tools.connection import HPCConnection

    # Initialize connection (reads ~/.ssh/config)
    hpc = HPCConnection(host_alias="euler")

    # Copy SLURM script and submit job
    job_id = hpc.submit_job(slurm_file="outputs/train_cgan_cnn_2d_beams2d_seed1.slurm")
    print(f"Job submitted with ID: {job_id}")

    # Check job status
    status = hpc.get_job_status(job_id)
    print(status)
"""

import logging
import re
import sys
from pathlib import Path

from fabric import Connection

logger = logging.getLogger(__name__)


class HPCConnection:
    """SSH connection to HPC cluster for job submission using Fabric."""

    def __init__(
        self,
        host_alias: str = "euler",
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int = 22,
    ) -> None:
        """
        Initialize HPC connection.

        Supports two authentication modes:
        1. SSH config mode (default): Uses ~/.ssh/config with SSH keys
        2. Password mode: Uses explicit username/password credentials

        Args:
            host_alias: Host alias from ~/.ssh/config (e.g., "euler")
            host: Explicit hostname (e.g., "euler.ethz.ch"). If provided, uses password auth.
            user: Username for password authentication
            password: Password for authentication
            port: SSH port (default: 22)
        """
        self.host_alias = host_alias

        # Determine authentication mode
        if host and user and password:
            # Password authentication mode (don't log credentials for security)
            logger.info("Establishing SSH connection (password auth)")
            try:
                self.connection = Connection(
                    host=host,
                    user=user,
                    port=port,
                    connect_kwargs={
                        "password": password,
                        "look_for_keys": False,
                        "allow_agent": False,
                    },
                )
                self.host_alias = host
            except Exception as e:
                # Don't expose credentials in error messages
                msg = "SSH connection failed. Please verify:\n"
                msg += "  1. Hostname is correct\n"
                msg += "  2. Username and password are correct\n"
                msg += "  3. SSH access is enabled on the server\n"
                msg += "  4. Network connectivity to the host"
                logger.exception(f"SSH connection failed: {type(e).__name__}")
                raise RuntimeError(msg) from e
        else:
            # SSH config mode (original behavior)
            # Fabric automatically uses ~/.ssh/config for connection details
            logger.info(f"Connecting to {host_alias} (SSH config mode)")
            logger.info(
                "Using host alias: %s (from HPC_HOST_ALIAS in .env)", host_alias
            )
            try:
                # Enable SSH agent to handle encrypted private keys
                self.connection = Connection(
                    host_alias,
                    connect_kwargs={
                        "allow_agent": True,  # Use SSH agent for encrypted keys
                        "look_for_keys": True,  # Look for keys in standard locations
                    },
                )
                logger.info(f"SSH config connection established for {host_alias}")
            except Exception as e:
                logger.exception(f"SSH config connection failed for {host_alias}")
                msg = f"Failed to connect to '{host_alias}': {e}\n"
                msg += "Make sure:\n"
                msg += f"  1. Host {host_alias} is configured in ~/.ssh/config\n"
                msg += "  2. SSH key is set up correctly\n"
                msg += (
                    "  3. SSH agent is running (if using Docker, mount SSH_AUTH_SOCK)\n"
                )
                msg += f"  4. Test manually: ssh {host_alias}"
                raise RuntimeError(msg) from e

    def run_command(self, command: str) -> str:
        """
        Run a command over SSH.

        Args:
            command: Command to execute on remote

        Returns:
            Command output
        """
        result = self.connection.run(command, hide=False, warn=False)
        if result.failed:
            msg = f"SSH command failed: {command}\nError: {result.stderr}"
            raise RuntimeError(msg)
        return result.stdout.strip()

    def put_file(self, local_file: str, remote_path: str) -> None:
        """
        Copy a file to remote using SFTP.

        Args:
            local_file: Path to local file
            remote_path: Destination path on remote (can be relative to home)
        """
        local_path = Path(local_file)
        if not local_path.exists():
            msg = f"Local file not found: {local_file}"
            raise FileNotFoundError(msg)

        # Expand tilde in remote path using remote home directory
        if remote_path.startswith("~"):
            home_dir = self.run_command("echo $HOME").strip()
            expanded_remote = remote_path.replace("~", home_dir, 1)
        else:
            expanded_remote = remote_path

        # Fabric's put method handles SFTP file transfer
        try:
            self.connection.put(str(local_path.absolute()), expanded_remote)
        except Exception as e:
            msg = f"SFTP transfer failed: {e}"
            raise RuntimeError(msg) from e

    def get_file(self, remote_file: str, local_path: str) -> None:
        """
        Download a file from remote using SFTP.

        Args:
            remote_file: Path to remote file
            local_path: Destination path on local
        """
        self.connection.get(remote_file, local_path)

    def submit_job(
        self,
        slurm_file: str,
        remote_dir: str = "~/slurm_jobs",
    ) -> str:
        """
        Transfer SLURM script and submit job to HPC.

        Args:
            slurm_file: Path to local SLURM script
            remote_dir: Directory on remote to store scripts (default: ~/slurm_jobs)

        Returns:
            Job ID returned by sbatch
        """
        local_path = Path(slurm_file)
        if not local_path.exists():
            msg = f"SLURM script not found: {slurm_file}"
            raise FileNotFoundError(msg)

        filename = local_path.name

        # Ensure remote directory exists
        try:
            self.run_command(f"mkdir -p {remote_dir}")
        except RuntimeError as e:
            msg = f"Failed to create remote directory {remote_dir}: {e}"
            raise RuntimeError(msg) from e

        # Copy file to remote using SFTP
        remote_path = f"{remote_dir}/{filename}"
        logger.info(f"Copying {filename} to {self.host_alias}:{remote_path}")
        self.put_file(slurm_file, remote_path)
        logger.info("File transferred successfully")

        # Submit job
        logger.info("Submitting SLURM job...")
        sbatch_cmd = f"cd {remote_dir} && sbatch {filename}"
        output = self.run_command(sbatch_cmd)

        # Parse job ID from sbatch output
        # Expected format: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", output)
        if not match:
            msg = f"Could not parse job ID from sbatch output: {output}"
            raise RuntimeError(msg)

        job_id = match.group(1)
        logger.info(f"Job submitted with ID: {job_id}")
        return job_id

    def get_job_status(self, job_id: str) -> str:
        """
        Get status of a submitted job.

        Args:
            job_id: SLURM job ID

        Returns:
            Job status output
        """
        try:
            output = self.run_command(f"squeue -j {job_id}")
        except RuntimeError as e:
            return f"Error getting job status: {e}"
        else:
            if output:
                return output
            return f"Job {job_id} not found or completed"

    def cancel_job(self, job_id: str) -> None:
        """
        Cancel a running job.

        Args:
            job_id: SLURM job ID
        """
        logger.info(f"Cancelling job {job_id}")
        self.run_command(f"scancel {job_id}")
        logger.info(f"Job {job_id} cancelled")

    def get_job_output(
        self,
        job_id: str,
        remote_dir: str = "~/slurm_jobs",
        local_dir: str = "outputs",
    ) -> None:
        """
        Download job output files.

        Args:
            job_id: SLURM job ID
            remote_dir: Remote directory where SLURM outputs are stored
            local_dir: Local directory to download to
        """
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # Download output and error files
        for file_type in ["out", "err"]:
            # List files matching pattern in remote directory
            list_cmd = (
                f"ls {remote_dir}/engiopt_*_{job_id}.{file_type} 2>/dev/null || true"
            )
            try:
                file_list_output = self.run_command(list_cmd)
                if not file_list_output:
                    logger.warning(f"No .{file_type} files found for job {job_id}")
                    continue

                # Download each file
                for remote_file in file_list_output.split("\n"):
                    if remote_file.strip():
                        local_dest = str(local_path / Path(remote_file).name)
                        self.get_file(remote_file, local_dest)
                        logger.info(
                            f"Downloaded {Path(remote_file).name} to {local_dir}/"
                        )
            except RuntimeError as e:
                if "No such file" not in str(e):
                    logger.warning(f"Warning downloading .{file_type} files: {e}")


if __name__ == "__main__":
    # Example usage
    import sys

    min_args = 2
    if len(sys.argv) < min_args:
        print("Usage:")
        print(
            "  python connection.py test                           # Test SSH connection"
        )
        print("  python connection.py submit <slurm_file>")
        print("  python connection.py status <job_id>")
        print("  python connection.py cancel <job_id>")
        print("  python connection.py download <job_id>")
        sys.exit(1)

    command = sys.argv[1]

    try:
        hpc = HPCConnection(host_alias="euler")

        if command == "test":
            output = hpc.run_command("pwd")
            print("✅ SSH connection successful!")
            print(f"   Remote directory: {output}")

        elif command == "submit" and len(sys.argv) > min_args:
            job_id = hpc.submit_job(sys.argv[2])
            print(f"Job ID: {job_id}")

        elif command == "status" and len(sys.argv) > min_args:
            status = hpc.get_job_status(sys.argv[2])
            print(status)

        elif command == "cancel" and len(sys.argv) > min_args:
            hpc.cancel_job(sys.argv[2])

        elif command == "download" and len(sys.argv) > min_args:
            hpc.get_job_output(sys.argv[2])

        else:
            msg = f"Unknown command: {command}"
            print(msg)
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
