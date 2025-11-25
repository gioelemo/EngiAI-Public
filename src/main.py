"""
Main entry point for the engineer assistant application.

DEPRECATED: The CLI interface has been deprecated in favor of the Streamlit UI.
"""

import sys


def main() -> None:
    """Show deprecation message and exit."""
    print("\n" + "=" * 70)
    print("  ⚠️  CLI INTERFACE DEPRECATED")
    print("=" * 70)
    print("\nThe command-line interface has been deprecated.")
    print("\nPlease use the Streamlit web interface instead:")
    print("\n  🚀 Start the app:")
    print("     streamlit run src/ui/streamlit_app.py")
    print("\n  Or use the Makefile:")
    print("     make run-ui")
    print("\n  Or use Docker:")
    print("     docker-compose up -d")
    print("     Then visit: http://localhost:8501")
    print("\n" + "=" * 70 + "\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
