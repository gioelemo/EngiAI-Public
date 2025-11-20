"""
Engineering design agent with EngiBench integration.

This agent specializes in structural optimization and engineering design
problems using the EngiBench library.
"""

from src.agents.base_agent import BaseAgent
from src.tools.engibench import (
    create_problem,
    get_dataset_info,
    get_problem_details,
    get_problem_info,
    optimize_design,
    render_design,
    simulate_design,
)
from src.tools.engiopt import (
    download_wandb_model,
    generate_training_command,
    list_available_algorithms,
    load_wandb_model,
    sample_designs_from_model,
)
from src.tools.stl_export import convert_design_to_stl
from src.utils.prompts import ENGINEERING_AGENT_SYSTEM_PROMPT


class EngineeringAgent(BaseAgent):
    """Agent specialized in engineering design and structural optimization."""

    def __init__(self, model_name: str | None = None, temperature: float | None = None):
        """Initialize the engineering agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
        """
        super().__init__(model_name=model_name, temperature=temperature)

    def _create_tools(self) -> list:
        """Create the list of engineering tools.

        Returns:
            List of LangChain tools for engineering tasks
        """
        return [
            # Unified EngiBench tools
            create_problem,
            simulate_design,
            optimize_design,
            render_design,
            get_problem_info,
            get_problem_details,
            get_dataset_info,
            # Export tools
            convert_design_to_stl,
            # WandB model download tools
            download_wandb_model,
            list_available_algorithms,
            load_wandb_model,
            sample_designs_from_model,
            # Model training command generator
            generate_training_command,
        ]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the engineering agent.

        Returns:
            System prompt string
        """
        return ENGINEERING_AGENT_SYSTEM_PROMPT
