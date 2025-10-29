"""
Interactive chat CLI for different agent types.
"""

from typing import cast

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

    def _get_user_request(self) -> str:
        """Extract the user's original request from message history.

        Returns:
            User's request string, or empty string if not found
        """
        for msg in reversed(self.state["messages"]):
            if (
                hasattr(msg, "type")
                and msg.type == "human"
                and isinstance(msg.content, str)
            ):
                return msg.content
        return ""

    def _check_for_interrupt(self) -> tuple[bool, str]:
        """Check if the graph execution was interrupted.

        Returns:
            Tuple of (is_interrupted, user_request)
        """
        try:
            snapshot = self.agent.graph.get_state(self.config)  # type: ignore[union-attr]

            # Check for interrupts in subgraphs (like cli_agent)
            if hasattr(snapshot, "tasks") and snapshot.tasks:
                for task in snapshot.tasks:
                    # Check if any task has interrupts
                    if hasattr(task, "interrupts") and task.interrupts:
                        return True, self._get_user_request()

            # Check for top-level interrupts
            if hasattr(snapshot, "next") and snapshot.next:
                next_nodes = str(snapshot.next)

                if "cli_agent" in next_nodes or "tool_node" in next_nodes:
                    # Before showing confirmation, check if CLI agent will actually call tools
                    # If it's just an informational question, auto-resume without confirmation
                    command_info = self._extract_command_info()
                    if not command_info:
                        # No tool calls detected - this is just a question, not a command
                        # Auto-resume execution without confirmation
                        # The special marker "AUTO_RESUME" signals the caller to continue
                        return True, "__AUTO_RESUME__"

                    # Extract user's original request for context
                    return True, self._get_user_request()

        except Exception:
            # Silently handle errors in interrupt detection
            pass

        return False, ""

    def _extract_command_info(self) -> str:
        """Extract command information from pending tool calls.

        Returns:
            Formatted string with command details, or empty string if none found
        """
        try:
            # Get the snapshot to access state
            snapshot = self.agent.graph.get_state(self.config)  # type: ignore[union-attr]

            # Check if CLI agent is about to run
            if (
                hasattr(snapshot, "next")
                and snapshot.next
                and "cli_agent" in str(snapshot.next)
                and hasattr(snapshot, "values")
                and "messages" in snapshot.values
            ):
                messages = snapshot.values["messages"]

                # Temporarily invoke the CLI agent to get its plan
                # Use a separate thread ID so we don't affect the main conversation
                try:
                    # Check if agent has cli_agent attribute (SupervisorAgent)
                    if hasattr(self.agent, "cli_agent"):
                        cli_agent = self.agent.cli_agent  # type: ignore[attr-defined]
                        # Invoke CLI agent to get the plan (will be interrupted at tool_node)
                        cli_agent_state = cast(MessagesState, {"messages": messages})
                        cli_config = {"configurable": {"thread_id": "cli_preview"}}

                        # Invoke once - it will be interrupted before tool execution
                        _ = cli_agent.invoke(cli_agent_state, cli_config)

                        # Check the CLI agent's state for tool calls
                        cli_snapshot = cli_agent.agent.get_state(cli_config)  # type: ignore[union-attr]

                        if (
                            hasattr(cli_snapshot, "values")
                            and "messages" in cli_snapshot.values
                        ):
                            cli_messages = cli_snapshot.values["messages"]

                            # Look for tool calls in the most recent AI message
                            for msg in reversed(cli_messages):
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    return self._format_tool_calls(msg.tool_calls)

                except Exception:
                    # Silently fail and fall back to checking main state
                    pass

            # Fallback: Look for tool calls in main state messages
            for msg in reversed(self.state["messages"]):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    return self._format_tool_calls(msg.tool_calls)

        except Exception:
            # Silently fail - just return empty string
            pass

        return ""

    def _format_tool_calls(self, tool_calls: list) -> str:
        """Format tool calls into a readable string.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            Formatted string with command details
        """
        info_lines = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "Unknown")
            args = tool_call.get("args", {})

            if tool_name == "execute_cli_command":
                command = args.get("command", "N/A")
                working_dir = args.get("working_dir")
                timeout = args.get("timeout", 300)

                info_lines.append("Command to execute:")
                info_lines.append(f"  $ {command}")
                if working_dir:
                    info_lines.append(f"  Working directory: {working_dir}")
                info_lines.append(f"  Timeout: {timeout}s")

            elif tool_name == "check_cli_tool_available":
                tool = args.get("tool_name", "N/A")
                info_lines.append(f"Checking availability of tool: {tool}")

            elif tool_name == "list_directory_contents":
                directory = args.get("directory_path", "N/A")
                pattern = args.get("pattern")
                info_lines.append(f"Listing directory: {directory}")
                if pattern:
                    info_lines.append(f"  Pattern: {pattern}")

            else:
                info_lines.append(f"Tool: {tool_name}")
                if args:
                    info_lines.append(f"  Arguments: {args}")

        return "\n".join(info_lines)

    def _handle_confirmation(self, user_request: str) -> bool:
        """Show confirmation prompt and get user response.

        Args:
            user_request: The user's original request

        Returns:
            True if user confirmed, False if cancelled
        """
        print("\n⚠️  CLI Command Execution Pending")
        if user_request:
            print(f"\nBased on your request: {user_request}")

        # Extract and display the command that will be executed
        command_info = self._extract_command_info()
        if command_info:
            print(f"\n{command_info}")
        else:
            print("\nThe agent will execute a command-line tool.")

        confirmation = (
            input("\nType 'yes' to proceed or 'no' to cancel: ").strip().lower()
        )

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

                    # Check if this is an auto-resume (question, not command)
                    if user_request == "__AUTO_RESUME__":
                        # No tool calls detected - auto-resume without confirmation
                        result = self.agent.invoke(None, self.config)  # type: ignore[arg-type]
                        self.state = result
                        # Continue to display messages normally (keep original messages_before)
                    elif self._handle_confirmation(user_request):
                        # User confirmed - resume execution
                        # Resume the graph - CLI agent will now execute
                        result = self.agent.invoke(None, self.config)  # type: ignore[arg-type]

                        # Update state with final result
                        self.state = result
                        # Keep original messages_before to display only NEW messages
                    else:
                        # User cancelled - skip to next iteration
                        continue
                else:
                    # No interrupt - update state with result
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
