"""
Streaming text functionality for Streamlit chat interface.

Provides generators and utilities for streaming text responses with typewriter effect.
"""

import time
from collections.abc import Generator


def stream_text(text: str, chunk_size: int = 3) -> Generator[str, None, None]:
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


def stream_text_by_char(text: str, delay: float = 0.01) -> Generator[str, None, None]:
    """Generate text character by character for streaming display.

    This provides a more granular streaming effect, displaying text
    one character at a time.

    Args:
        text: The text to stream
        delay: Delay between characters in seconds (default: 0.01)

    Yields:
        Individual characters to be displayed incrementally
    """
    for char in text:
        yield char
        if delay > 0:
            time.sleep(delay)


def stream_markdown_sections(text: str) -> Generator[str, None, None]:
    """Stream markdown text by sections (paragraphs and code blocks).

    This is optimized for markdown content, yielding complete sections
    to avoid breaking markdown formatting during streaming.

    Args:
        text: Markdown text to stream

    Yields:
        Complete markdown sections (paragraphs, code blocks, etc.)
    """
    # Split by double newlines (paragraphs) but keep code blocks intact
    in_code_block = False
    current_section = []
    lines = text.split("\n")

    for line in lines:
        # Check for code block markers
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        current_section.append(line)

        # Yield on paragraph breaks (empty lines) but not inside code blocks
        if not line.strip() and not in_code_block and current_section:
            section_text = "\n".join(current_section)
            if section_text.strip():
                yield section_text + "\n"
            current_section = []

    # Yield remaining content
    if current_section:
        section_text = "\n".join(current_section)
        if section_text.strip():
            yield section_text
