"""
CLI command confirmation handling for the Streamlit UI.

Handles the confirmation flow when the agent wants to execute CLI commands.
"""

from typing import cast

import streamlit as st

from src.models.state import MessagesState
from src.ui.message_processing import format_and_display_messages


def check_streamlit_interrupt() -> tuple[bool, str]:
    """Check if graph execution was interrupted for confirmation.

    Returns:
        Tuple of (is_interrupted, user_request)
    """
    try:
        snapshot = st.session_state.agent.graph.get_state(st.session_state.config)  # type: ignore[attr-defined]
        if hasattr(snapshot, "next") and snapshot.next:
            next_nodes = str(snapshot.next)

            if "cli_agent" in next_nodes:
                # Before showing confirmation, check if CLI agent will actually call tools
                # If it's just an informational question, auto-resume without confirmation
                command_info = extract_command_info()
                if not command_info:
                    # No tool calls detected - this is just a question, not a command
                    # Auto-resume execution without confirmation
                    # The special marker "AUTO_RESUME" signals the caller to continue
                    return True, "__AUTO_RESUME__"

                # Extract user's original request for context
                user_request = ""
                if st.session_state.agent_state.get("messages"):
                    for msg in reversed(st.session_state.agent_state["messages"]):
                        if hasattr(msg, "type") and msg.type == "human":
                            if isinstance(msg.content, str):
                                user_request = msg.content
                            break
                return True, user_request
    except Exception as e:
        st.error(f"[DEBUG] Exception: {e}")

    return False, ""


def extract_command_info() -> str:
    """Extract command information from pending CLI tool calls.

    Returns:
        Formatted string with command details, or empty string if none found
    """
    try:
        # Get the snapshot to access state
        snapshot = st.session_state.agent.graph.get_state(st.session_state.config)  # type: ignore[attr-defined]

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
                if hasattr(st.session_state.agent, "cli_agent"):
                    cli_agent = st.session_state.agent.cli_agent  # type: ignore[attr-defined]
                    # Invoke CLI agent to get the plan (will be interrupted at tool_node)
                    cli_agent_state = cast(MessagesState, {"messages": messages})
                    cli_config = {
                        "configurable": {"thread_id": "cli_preview_streamlit"}
                    }

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
                                return format_tool_calls_for_display(msg.tool_calls)

            except Exception:
                # Silently fail and fall back to checking main state
                pass

        # Fallback: Look for tool calls in main state messages
        for msg in reversed(st.session_state.agent_state.get("messages", [])):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return format_tool_calls_for_display(msg.tool_calls)

    except Exception:
        # Silently fail - just return empty string
        pass

    return ""


def format_tool_calls_for_display(tool_calls: list) -> str:
    """Format tool calls into a readable string for display.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        Formatted string with command details, or empty string if only safe tools
    """
    # Safe tools that don't require user confirmation
    safe_tools = {
        "open_gui_application",
        "check_cli_tool_available",
        "list_directory_contents",
        "get_prusa_slicer_path",
    }

    info_lines = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "Unknown")
        args = tool_call.get("args", {})

        # Skip safe tools - they don't need confirmation
        if tool_name in safe_tools:
            continue

        if tool_name == "execute_cli_command":
            command = args.get("command", "N/A")
            working_dir = args.get("working_dir")
            timeout = args.get("timeout", 300)

            info_lines.append("**Command to execute:**")
            info_lines.append(f"```bash\n$ {command}\n```")
            if working_dir:
                info_lines.append(f"**Working directory:** `{working_dir}`")
            info_lines.append(f"**Timeout:** {timeout}s")

        else:
            # Unknown tool - show it for safety
            info_lines.append(f"**Tool:** `{tool_name}`")
            if args:
                info_lines.append(f"**Arguments:** `{args}`")

    return "\n".join(info_lines)


def show_confirmation_prompt(user_request: str) -> None:
    """Show confirmation prompt in Streamlit UI.

    Args:
        user_request: The user's original request
    """
    st.session_state.waiting_for_confirmation = True

    # Build confirmation message
    msg_parts = ["⚠️ **CLI Command Execution Pending**\n"]

    if user_request:
        msg_parts.append(f"Based on your request:\n> *{user_request}*\n")

    # Extract and display the command information
    command_info = extract_command_info()
    if command_info:
        msg_parts.append(f"\n{command_info}\n")
    else:
        msg_parts.append("\nThe agent will execute a command-line tool.\n")

    msg_parts.append("\nType **'yes'** to proceed or **'no'** to cancel.")

    confirm_msg = "\n".join(msg_parts)
    st.warning(confirm_msg)
    st.session_state.messages.append({"role": "assistant", "content": confirm_msg})


def handle_confirmation_response(user_input: str) -> None:
    """Handle user's confirmation response (yes/no).

    Args:
        user_input: User's confirmation response
    """
    user_response = user_input.lower().strip()

    # Display the confirmation response
    with st.chat_message("user"):
        st.markdown(user_input)

    if user_response in ["yes", "y", "si", "sì", "ok", "proceed", "confermo", "certo"]:
        # User confirmed - resume execution
        with st.chat_message("assistant"), st.spinner("Executing command..."):
            try:
                result = st.session_state.agent.invoke(None, st.session_state.config)  # type: ignore[arg-type]
                st.session_state.waiting_for_confirmation = False
                st.session_state.agent_state = result

                # Use the stored messages_before count to display only new messages
                messages_before = st.session_state.get(
                    "messages_before_confirmation", 0
                )
                new_messages = st.session_state.agent_state["messages"][
                    messages_before:
                ]

                full_response, suggested_prompts = format_and_display_messages(
                    new_messages
                )
                if full_response:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "suggested_prompts": suggested_prompts,
                        }
                    )
            except Exception as e:
                st.error(f"Error executing command: {e}")
                st.session_state.waiting_for_confirmation = False
    else:
        # User cancelled
        st.session_state.waiting_for_confirmation = False
        with st.chat_message("assistant"):
            cancel_msg = "❌ Command execution cancelled."
            st.markdown(cancel_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": cancel_msg}
            )
