"""Home page for the Engineer Assistant Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def render() -> None:
    """Render the home page."""
    # Add vertical spacing
    st.markdown("")
    st.markdown("")

    # Center the logo and text
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        logo_path = project_root / "assets" / "logo.png"
        if logo_path.exists():
            try:
                # Use file path directly instead of PIL to avoid caching issues
                st.image(str(logo_path), width="stretch")
            except Exception:
                st.title("💬 EngiAI", text_alignment="center")
        else:
            st.title("💬 EngiAI", text_alignment="center")

        st.caption(
            "Your AI-powered engineering design assistant",
            text_alignment="center",
        )

    st.markdown("")

    # Feature highlights
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔧 Engineering Agent")
        st.markdown(
            "Specialized in optimization tasks, design analysis, and engineering calculations."
        )

    with col2:
        st.markdown("### 🏗️ CAD Agent")
        st.markdown(
            "Handles STL conversion, 3D printing preparation, and CAD-related tasks."
        )

    with col3:
        st.markdown("### 🔍 Search Agent")
        st.markdown(
            "Research capabilities and information gathering for your projects."
        )

    st.markdown("---")

    # Getting started section
    st.markdown("## 🚀 Getting Started")
    st.markdown(
        """
    Navigate to the **Chat** page to start interacting with the multi-agent system. The assistant
    will automatically route your requests to the most appropriate specialized agent.

    **Example queries:**
    - "Optimize this design for minimum weight"
    - "Convert this heatmap to an STL file for 3D printing"
    - "Research the latest materials for aerospace applications"
    - "Create a SLURM job for parallel simulation"

    Check the **W&B Report** page to view your training metrics and experiment tracking.
    """
    )

    st.markdown("---")

    # System information
    st.markdown("## System Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Multi-Agent Architecture**")
        st.markdown(
            """
        - Supervisor coordinates specialized agents
        - Dynamic routing based on task requirements
        - Supports HPC/SLURM integration
        - Real-time visualization of results
        """
        )

    with info_col2:
        st.markdown("**Capabilities**")
        st.markdown(
            """
        - Engineering optimization workflows
        - 3D model generation and visualization
        - Research and information retrieval
        - Command-line tool integration
        """
        )


if __name__ == "__main__":
    render()
