"""
Main entry point for the engineer assistant application.
"""

import sys

from src.cli.chat import main_engineering, main_general, main_math, main_search


def main() -> None:
    """Run the engineer assistant application."""
    # Check if user specified an agent type
    if len(sys.argv) > 1:
        agent_type = sys.argv[1].lower()
        if agent_type in ["math", "m"]:
            print("Starting Math Assistant...\n")
            main_math()
        elif agent_type in ["search", "s"]:
            print("Starting Search Assistant...\n")
            main_search()
        elif agent_type in ["general", "g"]:
            print("Starting General Assistant...\n")
            main_general()
        elif agent_type in ["engineering", "eng", "e"]:
            print("Starting Engineering Assistant...\n")
            main_engineering()
        else:
            print(f"Unknown agent type: {agent_type}")
            print("Available agents: math, search, general, engineering")
            print("\nUsage: python -m src.main [agent_type]")
            print("  math (m)        - Math operations only")
            print("  search (s)      - Web search only")
            print("  general (g)     - Both math and search (default)")
            print(
                "  engineering (e) - Structural design & optimization with EngiBench\n"
            )
            sys.exit(1)
    else:
        # Default to general agent
        main_general()


if __name__ == "__main__":
    main()
