"""
Interactive chat CLI for different agent types.
"""

from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.engineering_agent import EngineeringAgent
from src.agents.search_agent import SearchAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.models.state import MessagesState


class ChatCLI:
    """Interactive command-line interface for chatting with agents."""

    def __init__(
        self,
        agent: SearchAgent | EngineeringAgent | SupervisorAgent,
    ) -> None:
        """Initialize the chat CLI.

        Args:
            agent: The agent to chat with
        """
        self.agent = agent
        self.state: MessagesState = {"messages": []}
        self.config: dict[str, dict[str, str]] = {"configurable": {"thread_id": "1"}}

    def print_banner(self, agent_type: str, capabilities: str) -> None:
        """Print welcome banner.

        Args:
            agent_type: Name of the agent type
            capabilities: Description of agent capabilities
        """
        print(f"🤖 {agent_type}")
        print("=" * 50)
        print(capabilities)
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

    def _check_for_interrupt(self) -> tuple[bool, str]:
        """Check if the graph execution was interrupted.

        Returns:
            Tuple of (is_interrupted, user_request)
        """
        try:
            snapshot = self.agent.graph.get_state(self.config)  # type: ignore[union-attr]

            if hasattr(snapshot, "next") and snapshot.next:
                next_nodes = str(snapshot.next)

                if "cli_agent" in next_nodes:
                    # Extract user's original request for context
                    user_request = ""
                    for msg in reversed(self.state["messages"]):
                        if hasattr(msg, "type") and msg.type == "human":
                            if isinstance(msg.content, str):
                                user_request = msg.content
                            break
                    return True, user_request

        except Exception as e:
            print(f"[DEBUG] Error checking snapshot: {type(e).__name__}: {e}")

        return False, ""

    def _handle_confirmation(self, user_request: str) -> bool:
        """Show confirmation prompt and get user response.

        Args:
            user_request: The user's original request

        Returns:
            True if user confirmed, False if cancelled
        """
        print("\n⚠️  CLI Command Execution Pending")
        if user_request:
            print(f'Based on your request: "{user_request}"')
        print("The agent will execute a command-line tool.")
        confirmation = input("\nProceed? (yes/no): ").strip().lower()

        if confirmation in [
            "yes",
            "y",
            "si",
            "sì",
            "ok",
            "proceed",
            "confermo",
            "certo",
        ]:
            print("✓ Executing...\n")
            return True
        else:
            print("✗ Cancelled\n")
            return False

    def _display_new_messages(self, messages_before: int) -> None:
        """Display new messages from the conversation.

        Args:
            messages_before: Number of messages before the last operation
        """
        print()
        new_messages = self.state["messages"][messages_before:]

        for message in new_messages:
            # Skip re-printing user messages
            if isinstance(message, HumanMessage):
                continue

            # Show tool calls before the message (if any)
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name", "Unknown")
                    print(f"🔨 Tool: {tool_name}")

            # Show tool results or AI responses with content
            should_print = isinstance(message, ToolMessage) or (
                hasattr(message, "content") and message.content
            )
            if should_print:
                message.pretty_print()
                print()

    def run(self, agent_type: str, capabilities: str) -> None:
        """Run the interactive chat loop.

        Args:
            agent_type: Name of the agent type for the banner
            capabilities: Description of agent capabilities for the banner
        """
        self.print_banner(agent_type, capabilities)

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
                # Track messages before invocation
                messages_before = len(self.state["messages"])

                # First invocation - may be interrupted
                result = self.agent.invoke(self.state, self.config)  # type: ignore[arg-type]

                # Check if graph was interrupted for confirmation
                is_interrupted, user_request = self._check_for_interrupt()

                if is_interrupted:
                    # Update state before confirmation (partial result)
                    self.state = result

                    # Show confirmation and get user response
                    if self._handle_confirmation(user_request):
                        # User confirmed - track messages before resume
                        messages_before = len(self.state["messages"])
                        # Resume execution
                        result = self.agent.invoke(None, self.config)  # type: ignore[arg-type]
                    else:
                        # User cancelled - skip to next iteration
                        continue

                # Update state with result
                self.state = result

                # Display new messages
                self._display_new_messages(messages_before)

            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                # Remove the last user message on error
                if self.state["messages"]:
                    self.state["messages"].pop()


def main_supervisor() -> None:
    """Main entry point for the supervisor multi-agent system."""
    agent = SupervisorAgent()
    cli = ChatCLI(agent)
    cli.run(
        agent_type="Multi-Agent Engineering System",
        capabilities="I coordinate specialized agents to solve complex engineering tasks!\n"
        "Agents: Engineering (optimization), CAD (STL conversion), Search (research)\n"
        "I can handle complete workflows from design to 3D printing.",
    )


# Default to supervisor agent
def main() -> None:
    """Main entry point for the CLI (defaults to supervisor agent)."""
    main_supervisor()


if __name__ == "__main__":
    main()
