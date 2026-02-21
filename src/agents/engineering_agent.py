"""
Engineering design agent with EngiBench integration.

This agent specializes in structural optimization and engineering design
problems using the EngiBench library.
"""

import logging

from src.agents.base_agent import BaseAgent
from src.tools.engibench import (
    create_problem,
    get_dataset_info,
    get_problem_details,
    optimize_design,
    render_design,
    simulate_design,
)
from src.tools.engiopt import (
    download_wandb_model,
    evaluate_model,
    generate_training_command,
    list_available_algorithms,
    load_wandb_model,
    sample_designs_from_model,
)
from src.tools.human_input import ask_human_for_clarification
from src.tools.stl_export import convert_design_to_stl
from src.utils.prompts import _build_engineering_agent_prompt

logger = logging.getLogger(__name__)


class EngineeringAgent(BaseAgent):
    """Agent specialized in engineering design and structural optimization."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ):
        """Initialize the engineering agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            seed: Random seed for model (defaults to config.llm_seed)
        """
        super().__init__(model_name=model_name, temperature=temperature, seed=seed)

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
            get_problem_details,
            get_dataset_info,
            # Export tools
            convert_design_to_stl,
            # Clarification tool
            ask_human_for_clarification,
            # WandB model download tools
            download_wandb_model,
            list_available_algorithms,
            load_wandb_model,
            sample_designs_from_model,
            # Model training command generator
            generate_training_command,
            # Model evaluation
            evaluate_model,
        ]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the engineering agent.

        Returns:
            System prompt string
        """
        return _build_engineering_agent_prompt()
