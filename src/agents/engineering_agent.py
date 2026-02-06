"""
Engineering design agent with EngiBench integration.

This agent specializes in structural optimization and engineering design
problems using the EngiBench library. It also has access to the RAG
knowledge base for looking up reference material during design tasks.
"""

import logging
import os

from src.agents.base_agent import BaseAgent
from src.tools import MMOREClient
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
    generate_training_command,
    list_available_algorithms,
    load_wandb_model,
    sample_designs_from_model,
)
from src.tools.rag_tools import create_rag_tools
from src.tools.stl_export import convert_design_to_stl
from src.utils.prompts import _build_engineering_agent_prompt

logger = logging.getLogger(__name__)


class EngineeringAgent(BaseAgent):
    """Agent specialized in engineering design and structural optimization."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        mmore_url: str | None = None,
        seed: int | None = None,
    ):
        """Initialize the engineering agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            mmore_url: URL of MMORE service (defaults to MMORE_RAG_URL env var)
            seed: Random seed for model (defaults to config.llm_seed)
        """
        # Initialize MMORE client before super().__init__() so _create_tools() can use it
        skip_mmore = os.getenv("SKIP_MMORE", "false").lower() == "true"

        if skip_mmore:
            logger.info(
                "SKIP_MMORE=true: Engineering Agent initialized without MMORE client"
            )
            self.mmore_client: MMOREClient | None = None
        else:
            self.mmore_client = MMOREClient(base_url=mmore_url)

        super().__init__(model_name=model_name, temperature=temperature, seed=seed)

        # Log MMORE status
        if self.mmore_client is not None:
            if self.mmore_client.health_check():
                logger.info("Engineering Agent initialized with MMORE RAG tools")
            else:
                logger.warning("MMORE service not reachable - RAG tools may not work")

    def _create_tools(self) -> list:
        """Create the list of engineering tools.

        Returns:
            List of LangChain tools for engineering tasks, plus RAG tools if MMORE is available
        """
        tools = [
            # Unified EngiBench tools
            create_problem,
            simulate_design,
            optimize_design,
            render_design,
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

        # Add RAG tools if MMORE is available
        if self.mmore_client is not None:
            tools.extend(create_rag_tools(self.mmore_client))

        return tools

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the engineering agent.

        Returns:
            System prompt string
        """
        # Call builder function dynamically to pick up current config.mmore_enabled
        return _build_engineering_agent_prompt()
