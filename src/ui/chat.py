"""Chat page for the Engineer Assistant Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.canvas_bridge import (  # noqa: E402
    get_excalidraw_whiteboard,
    process_canvas_export,
)
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


def _render_canvas_column() -> None:
    """Render the canvas/whiteboard column."""
    st.subheader("🎨 Whiteboard", text_alignment="center")

    # Check if export was triggered
    trigger_export = st.session_state.get("trigger_canvas_export", False)
    if trigger_export:
        st.session_state.trigger_canvas_export = False

    # Render the Excalidraw component
    export_data = get_excalidraw_whiteboard(
        height=650, key="excalidraw_main", trigger_export=trigger_export
    )

    # If we received data and it's different from what we last processed, store it
    if export_data:
        last_processed = st.session_state.get("last_processed_export")
        if export_data != last_processed:
            st.session_state.pending_canvas_export = export_data
            st.session_state.last_processed_export = export_data
    elif trigger_export:
        # Canvas was empty when export was triggered
        st.warning("Canvas is empty! Draw something first.")

    # Add Streamlit button below canvas
    if st.button(
        "📤 Send to Chat",
        key="canvas_send_button",
        use_container_width=True,
        type="primary",
    ):
        # Trigger export by setting a flag
        st.session_state.trigger_canvas_export = True
        st.rerun()


def _get_pending_inputs() -> tuple[str | dict | None, str | None]:
    """Get pending canvas export and suggestion inputs.

    Returns:
        Tuple of (pending_canvas_input, pending_suggestion)
    """
    pending_canvas_input = None
    if st.session_state.get("pending_canvas_export"):
        try:
            # Process the export
            pending_canvas_input = process_canvas_export(
                st.session_state.pending_canvas_export
            )
            # Clear the pending export
            del st.session_state.pending_canvas_export
        except Exception as e:
            st.error(f"Failed to process canvas export: {e}")
            if "pending_canvas_export" in st.session_state:
                del st.session_state.pending_canvas_export

    pending_suggestion = None
    if (
        hasattr(st.session_state, "selected_suggestion")
        and st.session_state.selected_suggestion
    ):
        pending_suggestion = st.session_state.selected_suggestion
        st.session_state.selected_suggestion = None

    return pending_canvas_input, pending_suggestion


def _render_chat_history() -> None:
    """Render chat message history or welcome message."""
    if st.session_state.messages:
        # Display chat history
        for message_idx, message in enumerate(st.session_state.messages):
            display_message(message, message_idx)

        # Show job monitoring prompt if there are pending jobs
        if st.session_state.get("pending_job_monitor"):
            _render_job_monitoring_prompt()
    else:
        # Welcome message for empty chat
        st.markdown("")
        st.markdown("")
        st.header("💬 Start a conversation", text_alignment="center")
        st.caption(
            "Ask me anything about engineering design, optimization, or 3D modeling",
            text_alignment="center",
        )


def render() -> None:
    """Render the chat page with Excalidraw canvas."""
    # Create two columns: 2/3 for chat, 1/3 for canvas
    chat_col, canvas_col = st.columns([2, 1], vertical_alignment="bottom")

    # Render canvas column
    with canvas_col:
        _render_canvas_column()

    # Render chat column
    with chat_col:
        # Get pending inputs (canvas export and suggestions)
        pending_canvas_input, pending_suggestion = _get_pending_inputs()

        # Display chat history or welcome message
        _render_chat_history()

        # Create a placeholder for new messages BEFORE the chat input
        new_message_placeholder = st.empty()

        # Chat input with image upload support
        message = st.chat_input(
            "Ask me anything about engineering design...",
            accept_file=True,
            file_type=["png", "jpg", "jpeg", "gif", "webp", "pdf"],
        )

        # Determine what to process: canvas export > pending suggestion > chat input
        input_to_process = pending_canvas_input or pending_suggestion or message

        # Process the input if we have any
        if input_to_process:
            with new_message_placeholder.container():
                process_user_input(input_to_process)
            st.rerun()

        # Render compact job monitor at the bottom if there are jobs
        if st.session_state.get("monitored_jobs"):
            st.markdown("---")
            render_job_monitor_compact()


if __name__ == "__main__":
    render()
