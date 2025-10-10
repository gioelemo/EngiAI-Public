"""
Main entry point for the engineer assistant application.
"""

from src.cli.chat import main as chat_main


def main() -> None:
    """Run the engineer assistant application."""
    chat_main()


if __name__ == "__main__":
    main()
