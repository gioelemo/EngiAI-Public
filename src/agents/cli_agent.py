"""
CLI agent for executing local command-line tools.

This agent specializes in running command-line applications like PrusaSlicer,
mesh processing tools, file converters, and other CLI utilities.

Uses LangGraph's interrupt_before mechanism for human-in-the-loop confirmation.
"""

from src.agents.base_agent import BaseAgent
from src.tools.cli import (
    execute_cli_command,
    list_directory_contents,
    open_gui_application,
    open_terminal,
)
from src.utils.prompts import CLI_AGENT_SYSTEM_PROMPT


class CLIAgent(BaseAgent):
    """Agent specialized in executing local command-line tools."""

    def __init__(
        self,
        model_name: str | None = None,
        require_confirmation: bool = False,
        temperature: float | None = None,
    ):
        """Initialize the CLI agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            require_confirmation: Whether to require user confirmation before executing commands (default: False)
            temperature: Model temperature (defaults to config.llm_temperature)
        """
        super().__init__(
            model_name=model_name,
            require_confirmation=require_confirmation,
            temperature=temperature,
        )

    def _create_tools(self) -> list:
        """Create the list of CLI tools.

        Returns:
            List of LangChain tools for CLI operations
        """
        return [
            execute_cli_command,
            list_directory_contents,
            open_gui_application,
            open_terminal,
        ]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the CLI agent.

        Returns:
            System prompt string
        """
        return CLI_AGENT_SYSTEM_PROMPT
