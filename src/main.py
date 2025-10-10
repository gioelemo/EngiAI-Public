"""
Main entry point for the engineer assistant application.
"""

import sys

from src.cli.chat import (
    main_engineering,
    main_search,
    main_supervisor,
)


def main() -> None:
    """Run the engineer assistant application."""
    # Check if user specified an agent type
    if len(sys.argv) > 1:
        agent_type = sys.argv[1].lower()
        if agent_type in ["search", "s"]:
            print("Starting Search Assistant...\n")
            main_search()
        elif agent_type in ["engineering", "eng", "e"]:
            print("Starting Engineering Assistant...\n")
            main_engineering()
        elif agent_type in ["supervisor", "super", "multi", "team"]:
            print("Starting Multi-Agent Supervisor System...\n")
            main_supervisor()
        else:
            print(f"Unknown agent type: {agent_type}")
            print("Available agents: search, engineering, supervisor")
            print("\nUsage: python -m src.main [agent_type]")
            print("  search (s)        - Web search only")
            print(
                "  engineering (e)   - Structural design & optimization with EngiBench"
            )
            print(
                "  supervisor (team) - Multi-agent system coordinating engineering, CAD, and search (default)\n"
            )
            sys.exit(1)
    else:
        # Default to supervisor agent
        main_supervisor()


if __name__ == "__main__":
    main()
