"""Chat page for the Engineer Assistant Streamlit app."""

import sys
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html as components_html

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
    """Render the chat page with Excalidraw canvas."""
    # Create two columns: 2/3 for chat, 1/3 for canvas
    chat_col, canvas_col = st.columns([2, 1])

    with canvas_col:
        # Center the whiteboard section
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <h3 style="text-align: center; margin-bottom: 1rem;">🎨 Whiteboard</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Excalidraw HTML for iframe with export functionality
        excalidraw_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.css"/>
    <style>
        body { margin: 0; padding: 0; background: #f5f5f5; font-family: sans-serif; }
        #app { height: calc(100vh - 60px); width: 100%; }
        #export-btn {
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            background: #6965db;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 1000;
        }
        #export-btn:hover { background: #5753c5; }
        #export-btn:active { transform: translateX(-50%) scale(0.98); }
    </style>
    <script>window.EXCALIDRAW_ASSET_PATH = "https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/prod/";</script>
    <script type="importmap">
    {
        "imports": {
            "react": "https://esm.sh/react@19.0.0",
            "react/jsx-runtime": "https://esm.sh/react@19.0.0/jsx-runtime",
            "react-dom": "https://esm.sh/react-dom@19.0.0",
            "react-dom/client": "https://esm.sh/react-dom@19.0.0/client"
        }
    }
    </script>
</head>
<body>
    <div id="app"></div>
    <button id="export-btn">📤 Send to Chat</button>
    <script type="module">
        (async () => {
            try {
                const React = await import("react");
                const ReactDOM = await import("react-dom/client");
                const ExcalidrawLib = await import('https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.js?external=react,react-dom');

                let excalidrawAPI = null;

                const App = () => {
                    return React.createElement(
                        'div',
                        { style: { height: 'calc(100vh - 60px)' } },
                        React.createElement(ExcalidrawLib.Excalidraw, {
                            excalidrawAPI: (api) => { excalidrawAPI = api; }
                        })
                    );
                };

                const root = ReactDOM.createRoot(document.getElementById("app"));
                root.render(React.createElement(App));

                // Export button handler
                document.getElementById('export-btn').addEventListener('click', async () => {
                    if (!excalidrawAPI) {
                        alert('Excalidraw not ready yet!');
                        return;
                    }

                    const btn = document.getElementById('export-btn');
                    btn.textContent = '⏳ Exporting...';
                    btn.disabled = true;

                    try {
                        const elements = excalidrawAPI.getSceneElements();
                        if (!elements || elements.length === 0) {
                            alert('Canvas is empty! Draw something first.');
                            btn.textContent = '📤 Send to Chat';
                            btn.disabled = false;
                            return;
                        }

                        // Export to blob
                        const blob = await ExcalidrawLib.exportToBlob({
                            elements: elements,
                            appState: excalidrawAPI.getAppState(),
                            files: excalidrawAPI.getFiles(),
                            mimeType: 'image/png',
                            quality: 0.95
                        });

                        // Convert blob to base64
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            const base64data = reader.result;

                            // Download the image automatically
                            const link = document.createElement('a');
                            link.href = base64data;
                            link.download = 'excalidraw-' + Date.now() + '.png';
                            link.click();

                            // Also send via postMessage for potential auto-upload
                            window.parent.postMessage({
                                type: 'excalidraw-export',
                                data: base64data,
                                timestamp: Date.now()
                            }, '*');

                            btn.textContent = '✅ Sent!';
                            setTimeout(() => {
                                btn.textContent = '📤 Send to Chat';
                                btn.disabled = false;
                            }, 2000);
                        };
                        reader.readAsDataURL(blob);

                    } catch (error) {
                        alert('Export failed: ' + error.message);
                        btn.textContent = '📤 Send to Chat';
                        btn.disabled = false;
                    }
                });

            } catch (error) {
                document.getElementById('app').innerHTML = '<div style="padding:20px;color:red;">Error loading Excalidraw: ' + error.message + '</div>';
            }
        })();
    </script>
</body>
</html>
"""
        components_html(excalidraw_html, height=700)

        st.markdown("---")
        st.info(
            "💡 Click 'Send to Chat' button inside the whiteboard to download your drawing, then upload it in the chat below!"
        )

    with chat_col:
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
