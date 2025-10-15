"""
Main entry point for the engineer assistant application.
"""

import warnings

from src.cli.chat import main_supervisor

# Suppress Pydantic warnings from LangChain
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)


def main() -> None:
    """Run the engineer assistant application."""
    print("Starting Multi-Agent Supervisor System...\n")
    main_supervisor()


if __name__ == "__main__":
    main()
