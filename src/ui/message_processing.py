"""
Message processing and formatting functions for the Streamlit UI.

Handles message formatting, display, and special content extraction.
"""

import base64
import re
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from src.ui.media_display import (
    display_image,
    display_log_file,
    display_response_media,
    display_slurm_file,
    display_stl,
    find_images_in_text,
    find_log_files_in_text,
    find_slurm_files_in_text,
    find_stl_files_in_text,
)


def fix_latex_delimiters(text: str) -> str:
    """Fix LaTeX delimiters to be Streamlit-compatible.

    Converts \\( \\) to $ $ and \\[ \\] to $$ $$ for proper rendering.

    Args:
        text: Text that may contain LaTeX with incorrect delimiters

    Returns:
        Text with fixed LaTeX delimiters
    """
    # Replace display math: \[ ... \] with $$ ... $$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # Replace inline math: \( ... \) with $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

    # Also handle single bracket/paren versions that might appear
    text = re.sub(
        r"\[\s*([^]]*?)\s*\](?=\s|$|[.,;!?])",
        lambda m: (
            f"${m.group(1)}$"
            if any(
                c in m.group(1)
                for c in ["\\frac", "\\sum", "\\int", "=", "+", "-", "*", "/", "^", "_"]
            )
            else m.group(0)
        ),
        text,
    )

    return text


def extract_suggested_prompts(response: str) -> tuple[str, list[str]]:
    """Extract suggested prompts from the response.

    Args:
        response: The full response text

    Returns:
        Tuple of (cleaned_response, list of suggested prompts)
    """
    # Pattern to match suggested prompts block with proper backticks
    pattern = r"```suggested_prompts\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    # Fallback: Check for malformed suggested_prompts without backticks
    # This handles cases where LLM outputs "suggested_prompts" without code fences
    if not match:
        # Pattern 1: Multi-line format (each suggestion on new line)
        fallback_pattern_multiline = r"suggested_prompts\s*\n((?:.*?\n)+?)(?=\n\n|\Z)"
        fallback_match = re.search(
            fallback_pattern_multiline, response, re.DOTALL | re.IGNORECASE
        )

        if fallback_match:
            # Extract suggestions from malformed block
            suggestions_text = fallback_match.group(1)
            # Try to parse line-by-line format (newline separated)
            lines = [
                line.strip() for line in suggestions_text.split("\n") if line.strip()
            ]

            # Filter out lines that look like markdown formatting or are too short
            min_length = 3
            suggestions = [
                line
                for line in lines
                if len(line) > min_length
                and not line.startswith("#")
                and not line.startswith("*")
            ]

            if suggestions:
                # Remove the malformed block from response
                cleaned_response = response.replace(fallback_match.group(0), "").strip()
                cleaned_response = re.sub(r"\n{3,}", "\n\n", cleaned_response)
                return cleaned_response, suggestions

        # Pattern 2: Single-line format (all on same line after "suggested_prompts")
        fallback_pattern_inline = r"suggested_prompts\s+(.*?)(?=\n\n|\Z)"
        fallback_match_inline = re.search(
            fallback_pattern_inline, response, re.DOTALL | re.IGNORECASE
        )

        if fallback_match_inline:
            suggestions_text = fallback_match_inline.group(1)
            # Split by newlines to get individual suggestions
            lines = [
                line.strip() for line in suggestions_text.split("\n") if line.strip()
            ]

            min_length = 3
            suggestions = [
                line
                for line in lines
                if len(line) > min_length
                and not line.startswith("#")
                and not line.startswith("*")
            ]

            if suggestions:
                # Remove the malformed block from response
                cleaned_response = response.replace(
                    fallback_match_inline.group(0), ""
                ).strip()
                cleaned_response = re.sub(r"\n{3,}", "\n\n", cleaned_response)
                return cleaned_response, suggestions

        # No suggestions found at all
        return response, []

    # Extract the suggestions
    suggestions_text = match.group(1)
    suggestions = [s.strip() for s in suggestions_text.split("---") if s.strip()]

    # Remove the suggestions block from the response
    cleaned_response = re.sub(pattern, "", response, flags=re.DOTALL | re.IGNORECASE)

    # Strategy: Look for any paragraph that contains bullet points matching our suggestions
    # This is more aggressive but effective
    for suggestion in suggestions:
        # Escape special regex characters in the suggestion
        escaped_suggestion = re.escape(suggestion)
        # Remove any lines that contain this exact suggestion
        # This catches bullet points like "- Visualize the beam" or "* Visualize the beam"
        cleaned_response = re.sub(
            rf"^\s*[-•*]\s*{escaped_suggestion}\s*$",
            "",
            cleaned_response,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # Also remove common header patterns that introduce suggestions
    patterns_to_remove = [
        # "Would you like to:" followed by newlines/bullets
        r"(?:Would you like to|Let me know (?:if you (?:want|would like) to|your next step)|Alternatively|You (?:can|could|may))[:\s]*\n+(?:\s*[-•*]\s*[^\n]*\n+)*",
        # "Next steps:" or similar followed by newlines/bullets
        r"(?:Next steps?|Suggested (?:actions?|steps?|next steps?)|What'?s next\??)[:\s]*\n+(?:\s*[-•*]\s*[^\n]*\n+)*",
        # Standalone paragraph with only bullets (after we removed the matching ones)
        r"\n{2,}\s*(?:[-•*]\s*\n+)*\s*\n{2,}",
    ]

    for pattern_to_remove in patterns_to_remove:
        cleaned_response = re.sub(
            pattern_to_remove,
            "\n\n",
            cleaned_response,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # Clean up excessive newlines
    cleaned_response = re.sub(r"\n{3,}", "\n\n", cleaned_response)
    cleaned_response = cleaned_response.strip()

    return cleaned_response, suggestions


def extract_and_display_validation_warnings(response_text: str) -> str:
    """Extract validation warnings from response and display them as Streamlit warnings.

    Args:
        response_text: The response text that may contain validation warnings

    Returns:
        Response text with validation warnings removed (they'll be shown separately)
    """

    # --- Pattern 1: Check for the highly structured "CRITICAL" block ---
    pattern_structured = (
        r"={60,}\n🚨 \*\*CRITICAL: Resource Allocation Review\*\*\n={60,}.*?={60,}"
    )
    match_structured = re.search(pattern_structured, response_text, re.DOTALL)

    if match_structured:
        validation_block = match_structured.group(0)

        # Extract the issues
        issues_match = re.search(
            r"\*\*Issues Found:\*\*\n(.*?)(?=\n\*\*💡|={60})",
            validation_block,
            re.DOTALL,
        )
        recommendations_match = re.search(
            r"\*\*💡 Recommendations:\*\*\n(.*?)(?=\n={60})",
            validation_block,
            re.DOTALL,
        )

        # Display as Streamlit error
        error_message = "### 🚨 Resource Allocation Warning\n\n"

        if issues_match:
            issues_text = issues_match.group(1).strip()
            error_message += "**Issues Found:**\n" + issues_text + "\n\n"

        if recommendations_match:
            rec_text = recommendations_match.group(1).strip()
            error_message += "**💡 Recommendations:**\n" + rec_text + "\n\n"

        error_message += (
            "⚠️ **The script was generated despite exceeding recommended limits.**\n"
        )
        error_message += "⚠️ **Please review and adjust resources before submitting.**"

        st.error(error_message)

        # Remove the validation block from response text
        response_text = response_text.replace(validation_block, "").strip()
        return re.sub(r"\n{3,}", "\n\n", response_text)  # Return cleaned text

    # --- UPDATED: Pattern 2: Check for various unstructured warning blocks ---

    # Define start markers (case-insensitive)
    start_markers = [
        r"Important Validation Notes:",  # <-- ADDED THIS
        r"Important Warnings:",
        r"However, there are important notes and validation warnings:",
        r"Resource Validation/Warnings:",
        r"Resource Allocation Warning",
    ]

    # Define end markers (lookahead, case-insensitive)
    end_markers = [
        r"Generated SLURM Script \(shortened for clarity",  # <-- ADDED THIS
        r"File Location:",  # <-- ADDED THIS
        r"SLURM Script File:",
        r"SLURM Script \(saved to",
        r"The script below is valid",
        r"Example SLURM Script \(view below\):",
        r"##",  # Next markdown heading
        r"\n\nLet me know if",
        r"\n\nIf you want to correct",
        r"\n\nPlease reduce the GPU count",  # Add another common follow-up
    ]

    pattern_unstructured = (
        r"((?:{}).*?)"  # Start: Match any of the start markers
        r"(?={})"  # End: Lookahead for any of the end markers
    ).format("|".join(start_markers), "|".join(end_markers))

    match_unstructured = re.search(
        pattern_unstructured, response_text, re.DOTALL | re.IGNORECASE
    )

    if match_unstructured:
        validation_block = match_unstructured.group(1).strip()

        # Clean up the extracted block for display
        # Remove the introductory line itself to avoid redundancy
        warning_content = re.sub(
            "|".join(start_markers),
            "",
            validation_block,
            flags=re.IGNORECASE,
        ).strip()

        # Format it nicely for the warning box
        warning_msg = f"⚠️ **Resource Allocation Warning**\n\n{warning_content}"

        # Display it as a Streamlit warning
        st.warning(warning_msg)

        # Remove the validation block from the original response text
        response_text = response_text.replace(validation_block, "").strip()

        # Clean up potential double newlines
        response_text = re.sub(r"\n{3,}", "\n\n", response_text)

        return response_text

    # If no patterns matched, return the original text
    return response_text


def display_suggested_prompts(suggestions: list[str]) -> None:
    """Display suggested prompts as clickable buttons.

    Args:
        suggestions: List of suggested prompt strings
    """
    if not suggestions:
        return

    st.markdown("---")
    st.markdown("**💡 Suggested next steps:**")

    # Create columns for suggestions (max 2 per row)
    num_cols = min(2, len(suggestions))
    cols = st.columns(num_cols)

    # Use message count to ensure unique keys for newly displayed suggestions
    msg_count = len(st.session_state.get("messages", []))

    for idx, suggestion in enumerate(suggestions):
        col_idx = idx % num_cols
        button_key = f"new_suggestion_{msg_count}_{idx}"
        with cols[col_idx]:
            # Create a button for each suggestion
            # Use "new_" prefix and message count to distinguish from history buttons
            button_clicked = st.button(
                suggestion,
                key=button_key,
                width="stretch",
            )
            if button_clicked:
                # Store the selected suggestion to process
                st.session_state.selected_suggestion = suggestion
                # Trigger rerun so chat.py can process the suggestion
                st.rerun()


def format_tool_call(tool_call: Any) -> str:
    """Format a tool call for display.

    Args:
        tool_call: Tool call object (dict-like)

    Returns:
        Formatted string describing the tool call
    """
    # Tool calls can be dict-like objects
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("name", "Unknown")
    else:
        tool_name = getattr(tool_call, "name", "Unknown")
    return f"🔧 **Using tool:** `{tool_name}`"


def format_ai_message(message: AIMessage | ToolMessage) -> str:
    """Format an AI or tool message for display.

    Args:
        message: The message to format

    Returns:
        Formatted message content
    """
    max_content_length = 1000

    if isinstance(message, ToolMessage):
        # Tool results
        content = str(message.content)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n... (truncated)"
        return f"```\n{content}\n```"

    # AI message
    content = str(message.content) if message.content else ""

    # Add tool calls if present
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_info = "\n\n".join(
            format_tool_call(tc)
            for tc in message.tool_calls  # type: ignore[arg-type]
        )
        if content:
            return f"{tool_info}\n\n{content}"
        return tool_info

    return content


def format_and_display_messages(new_messages: list) -> tuple[str, list[str]]:
    """Format and display new messages from the agent.

    Displays the final AI response and any relevant tool outputs (like CLI command results).
    This keeps the UI clean while showing important execution results.

    Args:
        new_messages: List of new messages to display

    Returns:
        Tuple of (formatted response string, list of suggested prompts)
    """
    # Find the last AIMessage that has content and no tool_calls (the final response)
    # Also collect all tool outputs from the messages
    final_message = None
    tool_outputs = []

    for message in new_messages:
        if isinstance(message, ToolMessage):
            # Collect all tool outputs
            tool_outputs.append(message)
        elif isinstance(message, AIMessage):
            # Check if this is a final response (has content, no tool_calls)
            has_content = message.content and str(message.content).strip()
            has_tool_calls = hasattr(message, "tool_calls") and message.tool_calls
            if has_content and not has_tool_calls:
                final_message = message

    suggested_prompts: list[str] = []
    response_parts = []

    # Only show raw CLI command outputs (shell commands like pip list, ls, etc.)
    # Skip tool outputs from agents that format their own responses (Prusa, engineering, etc.)
    # The AI will provide a formatted summary for those
    if tool_outputs:
        for tool_msg in tool_outputs:
            content = str(tool_msg.content)
            # Only show outputs that look like raw shell command output:
            # - Has multiple lines
            # - Looks like package list, file listing, or similar tabular data
            # - Doesn't contain structured data patterns
            is_shell_output = (
                "\n" in content
                and not content.strip().startswith("{")
                and "array([" not in content
                and "'problem_id':" not in content
                and "'success':" not in content
                # Skip Prusa/printer outputs (formatted by the agent)
                and "Printer Name:" not in content
                and "Printer UUID:" not in content
                and "Temperature (Nozzle)" not in content
                # Skip JSON responses (API outputs)
                and '"id":' not in content
                and '"state":' not in content
                and '"command":' not in content
                # Skip other agent-formatted outputs
                and "---" not in content[:100]  # Separators indicate formatted output
            )
            if is_shell_output:
                # Truncate very long outputs
                max_content_length = 3000
                if len(content) > max_content_length:
                    content = (
                        content[:max_content_length] + "\n\n... (output truncated)"
                    )
                response_parts.append(f"```\n{content}\n```")

    if final_message:
        full_response = str(final_message.content)

        # Extract suggested prompts first
        cleaned_response, suggested_prompts = extract_suggested_prompts(full_response)

        # Extract and display validation warnings as separate Streamlit components
        cleaned_response = extract_and_display_validation_warnings(cleaned_response)

        response_parts.append(cleaned_response)

    # Combine all parts
    combined_response = "\n\n".join(response_parts)

    if combined_response:
        # Display the combined response
        st.markdown(combined_response)

        # Display media files
        display_response_media(combined_response)

        return combined_response, suggested_prompts

    return "", suggested_prompts


def _display_uploaded_files(message: dict, message_idx: int) -> None:
    """Display uploaded files (images and PDFs) from user messages.

    Args:
        message: Message dictionary
        message_idx: Index of the message for unique keys
    """
    if not message.get("images"):
        return

    for file_data in message["images"]:
        file_type = file_data.get("type", "")
        file_name = file_data.get("name", "file")

        if file_type == "application/pdf":
            # Display PDF as a download link
            pdf_bytes = base64.b64decode(file_data["data"])
            st.download_button(
                label=f"📄 {file_name}",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
                key=f"pdf_{message_idx}_{file_name}",
            )
        else:
            # Display as image
            img_bytes = base64.b64decode(file_data["data"])
            st.image(img_bytes, width=400)


def _display_media_files(message: dict, message_idx: int) -> None:
    """Display media files referenced in message content.

    Args:
        message: Message dictionary
        message_idx: Index of the message for unique keys
    """
    # Display images referenced in text (file paths)
    images = find_images_in_text(message["content"])
    for img_path in images:
        display_image(img_path, button_key_prefix=f"msg_{message_idx}_img")

    # Display STL files
    stl_files = find_stl_files_in_text(message["content"])
    for idx, stl_path in enumerate(stl_files):
        display_stl(stl_path, idx, button_key_prefix=f"msg_{message_idx}_stl")

    # Display SLURM scripts
    slurm_files = find_slurm_files_in_text(message["content"])
    for slurm_path in slurm_files:
        display_slurm_file(slurm_path, button_key_prefix=f"msg_{message_idx}_slurm")

    # Display log files (.err and .out)
    log_files = find_log_files_in_text(message["content"])
    for log_path in log_files:
        display_log_file(log_path, button_key_prefix=f"msg_{message_idx}_log")


def _display_suggested_prompts(message: dict, message_idx: int) -> None:
    """Display suggested prompts for assistant messages.

    Args:
        message: Message dictionary
        message_idx: Index of the message for unique keys
    """
    if message["role"] != "assistant" or not message.get("suggested_prompts"):
        return

    suggestions = message["suggested_prompts"]
    if not suggestions:
        return

    st.markdown("---")
    st.markdown("**💡 Suggested next steps:**")

    # Create columns for suggestions (max 2 per row)
    num_cols = min(2, len(suggestions))
    cols = st.columns(num_cols)

    for idx, suggestion in enumerate(suggestions):
        col_idx = idx % num_cols
        with cols[col_idx]:
            # Create a button for each suggestion
            if st.button(
                suggestion,
                key=f"hist_suggestion_{message_idx}_{idx}",
                width="stretch",
            ):
                # Store the selected suggestion to process
                st.session_state.selected_suggestion = suggestion
                st.rerun()


def display_message(message: dict, message_idx: int = 0) -> None:
    """Display a single message in the chat interface.

    Args:
        message: Dictionary with 'role' and 'content' keys (and optionally 'suggested_prompts')
        message_idx: Index of the message in the chat history for unique keys
    """
    with st.chat_message(message["role"]):
        # Fix LaTeX delimiters before displaying
        content = fix_latex_delimiters(message["content"])

        # For assistant messages, also clean any suggestions block from content
        if message["role"] == "assistant":
            cleaned_content, _ = extract_suggested_prompts(content)
            content = cleaned_content

        # Display text content
        st.markdown(content)

        # Display all file types
        _display_uploaded_files(message, message_idx)
        _display_media_files(message, message_idx)
        _display_suggested_prompts(message, message_idx)
