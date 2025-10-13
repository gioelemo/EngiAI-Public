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

                # Invoke the agent (simpler than streaming for multi-agent systems)
                result = self.agent.invoke(self.state, self.config)  # type: ignore[arg-type]

                # Update state with result
                self.state = result

                # Print new messages, showing tool calls
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

            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                # Remove the last user message on error
                if self.state["messages"]:
                    self.state["messages"].pop()


def main_search() -> None:
    """Main entry point for the search agent CLI."""
    agent = SearchAgent()
    cli = ChatCLI(agent)
    cli.run(
        agent_type="Research Assistant",
        capabilities="I can help you search the web for information!",
    )


def main_engineering() -> None:
    """Main entry point for the engineering agent CLI."""
    agent = EngineeringAgent()
    cli = ChatCLI(agent)
    cli.run(
        agent_type="Engineering Assistant",
        capabilities="I can help you with structural design and optimization using EngiBench!\n"
        "Capabilities: Beam optimization, design simulation, topology optimization.",
    )


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
