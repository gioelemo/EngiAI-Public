"""Chat page for the Engineer Assistant Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.message_processing import display_message  # noqa: E402
from src.ui.streamlit_app import (  # noqa: E402
    add_job_to_monitor,
    process_user_input,
    render_job_monitor_compact,
)


def _render_job_monitoring_prompt() -> None:
    """Render a prompt asking user if they want to monitor submitted jobs."""
    pending_jobs = st.session_state.get("pending_job_monitor", [])
    if not pending_jobs:
        return

    # Show prompt for each pending job
    for job_id in pending_jobs:
        with st.container(border=True):
            st.markdown(f"### 🚀 Job {job_id} Submitted")
            st.markdown("Would you like to monitor this job's status?")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button(
                    "✅ Yes, Monitor",
                    key=f"monitor_yes_{job_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    add_job_to_monitor(job_id, "SUBMITTED")
                    st.session_state.pending_job_monitor.remove(job_id)
                    st.success(f"✅ Now monitoring job {job_id}")
                    st.rerun()

            with col2:
                if st.button(
                    "❌ No Thanks",
                    key=f"monitor_no_{job_id}",
                    type="secondary",
                    use_container_width=True,
                ):
                    st.session_state.pending_job_monitor.remove(job_id)
                    st.info("You can still add it manually from the job monitor widget")
                    st.rerun()

            with col3:
                st.caption("💡 You can change this preference in Settings")


def render() -> None:
    """Render the chat page."""
    # IMPORTANT: Check for pending suggestion FIRST, before any rendering
    # This handles the case where a button was clicked and stored a suggestion
    pending_suggestion = None
    if (
        hasattr(st.session_state, "selected_suggestion")
        and st.session_state.selected_suggestion
    ):
        pending_suggestion = st.session_state.selected_suggestion
        st.session_state.selected_suggestion = (
            None  # Clear immediately to prevent double-processing
        )

    # Small logo at top when chat has messages
    if st.session_state.messages:
        # Display chat history
        for message_idx, message in enumerate(st.session_state.messages):
            display_message(message, message_idx)

        # Show job monitoring prompt if there are pending jobs
        if st.session_state.get("pending_job_monitor"):
            _render_job_monitoring_prompt()
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

    # Chat input with image upload support - always show it so user can respond to follow-up questions
    message = st.chat_input(
        "Ask me anything about engineering design...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "gif", "webp", "pdf"],
    )

    # Determine what to process: pending suggestion takes priority, then chat input
    input_to_process = pending_suggestion if pending_suggestion else message

    # Process the input if we have any
    if input_to_process:
        process_user_input(input_to_process)

    # Render compact job monitor at the bottom if there are jobs
    if st.session_state.get("monitored_jobs"):
        st.markdown("---")
        render_job_monitor_compact()


if __name__ == "__main__":
    render()
