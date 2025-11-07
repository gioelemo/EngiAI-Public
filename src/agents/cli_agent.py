"""
CLI agent for executing local command-line tools.

This agent specializes in running command-line applications like PrusaSlicer,
mesh processing tools, file converters, and other CLI utilities.

Uses LangGraph's interrupt_before mechanism for human-in-the-loop confirmation.
"""

from src.agents.base_agent import BaseAgent
from src.tools.cli import (
    check_cli_tool_available,
    execute_cli_command,
    get_prusa_slicer_path,
    list_directory_contents,
    open_gui_application,
)
from src.utils.prompts import CLI_AGENT_SYSTEM_PROMPT


class CLIAgent(BaseAgent):
    """Agent specialized in executing local command-line tools."""

    def __init__(
        self, model_name: str | None = None, require_confirmation: bool = True
    ):
        """Initialize the CLI agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            require_confirmation: Whether to require user confirmation before executing commands (default: True)
        """
        super().__init__(
            model_name=model_name, require_confirmation=require_confirmation
        )

    def _create_tools(self) -> list:
        """Create the list of CLI tools.

        Returns:
            List of LangChain tools for CLI operations
        """
        return [
            execute_cli_command,
            check_cli_tool_available,
            list_directory_contents,
            open_gui_application,
            get_prusa_slicer_path,
        ]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the CLI agent.

        Returns:
            System prompt string
        """
        return CLI_AGENT_SYSTEM_PROMPT
