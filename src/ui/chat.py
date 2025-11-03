"""Chat page for the Engineer Assistant Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.streamlit_app import display_message, process_user_input  # noqa: E402


def render() -> None:
    """Render the chat page."""
    # Small logo at top when chat has messages
    if st.session_state.messages:
        # Display chat history
        for message_idx, message in enumerate(st.session_state.messages):
            display_message(message, message_idx)
    else:
        # Welcome message for empty chat
        st.markdown("<br>" * 2, unsafe_allow_html=True)

        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.markdown(
                '<h2 style="text-align: center;">💬 Start a conversation</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="text-align: center; color: #666;">Ask me anything about engineering design, optimization, or 3D modeling</p>',
                unsafe_allow_html=True,
            )

    # Chat input
    if prompt := st.chat_input("Ask me anything about engineering design..."):
        process_user_input(prompt)


if __name__ == "__main__":
    render()
