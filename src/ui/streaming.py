"""
Streaming text functionality for Streamlit chat interface.

Provides generators and utilities for streaming text responses with typewriter effect.
"""

import time
from collections.abc import Generator


def stream_text(text: str, chunk_size: int = 3) -> Generator[str]:
    """Generate text chunks for streaming display with typewriter effect.

    This function intelligently splits text, handling markdown structures like
    lists, code blocks, and headings by yielding them as complete units to
    preserve formatting during streaming.

    Args:
        text: The text to stream
        chunk_size: Number of words to yield per chunk for regular text (default: 3)

    Yields:
        Text chunks to be displayed incrementally
    """
    lines = text.split("\n")
    in_code_block = False

    for line in lines:
        # Check for code block markers
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            yield line + "\n"
            time.sleep(0.02)
            continue

        # If inside code block, yield entire line
        if in_code_block:
            yield line + "\n"
            time.sleep(0.02)
            continue

        # Check if line is a markdown structure that should be yielded whole
        stripped = line.strip()
        min_numbered_list_length = 2
        is_special = (
            stripped.startswith(
                ("#", "-", "*", "+", ">", "|")
            )  # Headings, lists, blockquotes, tables
            or (
                len(stripped) > min_numbered_list_length
                and stripped[0].isdigit()
                and stripped[1:3] in (". ", ") ")
            )  # Numbered list
            or not stripped  # Empty line
        )

        if is_special:
            # Yield special markdown lines whole to preserve formatting
            yield line + "\n"
            time.sleep(0.05)
        else:
            # Regular text - stream word by word
            words = line.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield chunk
                time.sleep(0.05)
            # Add newline at end of line
            if words:
                yield "\n"
