"""
Interactive chat CLI for the engineer assistant.
"""

from langchain_core.messages import HumanMessage

from src.agents.math_agent import MathAgent
from src.models.state import MessagesState


class ChatCLI:
    """Interactive command-line interface for chatting with agents."""

    def __init__(self, agent: MathAgent):
        """Initialize the chat CLI.

        Args:
            agent: The agent to chat with
        """
        self.agent = agent
        self.state: MessagesState = {"messages": []}
        self.config = {"configurable": {"thread_id": "1"}}

    def print_banner(self) -> None:
        """Print welcome banner."""
        print("🤖 Interactive Assistant")
        print("=" * 50)
        print("I can help you with arithmetic operations or search on the web!")
        print("Commands: 'exit', 'quit', 'clear' to start fresh")
        print("=" * 50)
        print()

    def process_command(self, user_input: str) -> bool:
        """Process special commands.

        Args:
            user_input: User's input text

        Returns:
            True if a command was processed, False otherwise
        """
        command = user_input.lower()

        if command in ["exit", "quit", "bye"]:
            print("\n👋 Goodbye!")
            return True

        if command == "clear":
            self.state = {"messages": []}
            print("🔄 Conversation cleared!\n")
            return True

        return False

    def run(self) -> None:
        """Run the interactive chat loop."""
        self.print_banner()

        while True:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if self.process_command(user_input):
                if user_input.lower() in ["exit", "quit", "bye"]:
                    break
                continue

            # Add user message to state
            self.state["messages"].append(HumanMessage(content=user_input))

            try:
                # Invoke the agent
                result = self.agent.invoke(self.state, self.config)  # type: ignore[arg-type]

                # Update state with result
                self.state = result

                # Print all messages from this turn using pretty_print
                print()
                for message in self.state["messages"]:
                    message.pretty_print()
                print()

            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                # Remove the last user message on error
                if self.state["messages"]:
                    self.state["messages"].pop()


def main() -> None:
    """Main entry point for the chat CLI."""
    agent = MathAgent()
    cli = ChatCLI(agent)
    cli.run()


if __name__ == "__main__":
    main()
