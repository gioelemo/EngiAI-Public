"""
HPC cluster agent for job submission and monitoring.

This agent specializes in managing SLURM jobs on HPC clusters using SSH connections.
It handles job submission, status monitoring, job cancellation, and output retrieval.
"""

from src.agents.base_agent import BaseAgent
from src.tools.hpc import (
    cancel_slurm_job,
    download_job_outputs,
    get_slurm_job_status,
    submit_slurm_job,
    test_hpc_connection,
)
from src.tools.job_monitor import (
    check_job_status_change,
    get_active_jobs_summary,
    monitor_job_until_complete,
)
from src.utils.prompts import get_hpc_agent_system_prompt


class HPCAgent(BaseAgent):
    """Agent specialized in HPC cluster job management."""

    def __init__(self, model_name: str | None = None, temperature: float | None = None):
        """Initialize the HPC agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
        """
        super().__init__(model_name=model_name, temperature=temperature)

    def _create_tools(self) -> list:
        """Create the list of HPC tools.

        Returns:
            List of LangChain tools for HPC operations
        """
        return [
            test_hpc_connection,
            submit_slurm_job,
            get_slurm_job_status,
            cancel_slurm_job,
            download_job_outputs,
            monitor_job_until_complete,
            check_job_status_change,
            get_active_jobs_summary,
        ]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the HPC agent.

        Returns:
            System prompt string
        """
        return get_hpc_agent_system_prompt()
