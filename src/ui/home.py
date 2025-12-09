"""Home page for the Engineer Assistant Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _render_header() -> None:
    """Render the header section with logo and title."""
    st.markdown("")
    st.markdown("")

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        logo_path = project_root / "assets" / "logo.png"
        if logo_path.exists():
            try:
                st.image(str(logo_path), width="stretch")
            except Exception:
                st.title("💬 EngiAI", text_alignment="center")
        else:
            st.title("💬 EngiAI", text_alignment="center")

        st.caption(
            "Your AI-powered engineering design assistant",
            text_alignment="center",
        )


def _render_agent_grid() -> None:
    """Render the agent feature grid."""
    st.markdown("")
    st.markdown("### 🤖 Intelligent LLM-Based Agent Routing")
    st.markdown(
        "The supervisor uses **structured LLM output** to intelligently analyze your request "
        "and route it to the most appropriate specialized agent. No hardcoded keywords!"
    )
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 🔧 Engineering")
        st.markdown(
            "Structural optimization, topology design, STL generation, WandB models"
        )
        st.markdown("#### 📚 RAG")
        st.markdown("Document Q&A with MMORE multimodal RAG, HPC documentation")

    with col2:
        st.markdown("#### 🔍 Search")
        st.markdown("Web research and information gathering")
        st.markdown("#### 📄 ArXiv")
        st.markdown("Scientific paper search, download, and analysis")

    with col3:
        st.markdown("#### 🖨️ Prusa")
        st.markdown("3D printer management via Prusa Connect")
        st.markdown("#### 🖥️ HPC")
        st.markdown("SLURM job submission and cluster monitoring")

    with col4:
        st.markdown("#### ⚙️ CLI")
        st.markdown("Execute commands and open GUI apps (PrusaSlicer, Blender)")
        st.markdown("#### 🎯 Supervisor")
        st.markdown("Answers capability questions directly")


def _render_getting_started() -> None:
    """Render the getting started section with example queries."""
    st.markdown("## 🚀 Getting Started")
    st.markdown(
        """
    Navigate to the **Chat** page to start interacting with the multi-agent system. The intelligent
    supervisor will automatically analyze your request and route it to the most appropriate agent.

    **Example queries:**

    **Engineering & Optimization:**
    - "Optimize a 2D beam with 35% volume fraction" → Engineering Agent
    - "Convert this topology optimization to STL for 3D printing" → Engineering Agent

    **Documentation & Learning:**
    - "How do I submit a job on Euler cluster?" → RAG Agent (documentation query)
    - "What are SLURM commands?" → RAG Agent (HPC documentation)
    - "Explain beam optimization techniques" → RAG Agent

    **Actions & Execution:**
    - "Submit job.slurm to Euler cluster" → HPC Agent (action)
    - "Check status of job 12345" → HPC Agent
    - "Open PrusaSlicer" → CLI Agent

    **Research & Papers:**
    - "Search ArXiv for topology optimization papers" → ArXiv Agent
    - "Research latest materials for aerospace" → Search Agent

    **3D Printing:**
    - "Check my Prusa printer status" → Prusa Agent
    - "Slice this STL file" → CLI Agent

    The system intelligently distinguishes between **documentation questions** ("how do I...")
    and **action requests** ("do this"), routing them to the appropriate agent!
    """
    )


def _render_system_info() -> None:
    """Render the system information section."""
    st.markdown("## System Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Intelligent Multi-Agent Architecture**")
        st.markdown(
            """
        - **LLM-based routing** with structured Pydantic output
        - **Intent understanding**: Distinguishes questions from actions
        - **7 specialized agents** for different domains
        - **No hardcoded keywords**: Pure LLM decision-making
        - Logging with reasoning for routing decisions
        """
        )

    with info_col2:
        st.markdown("**Key Capabilities**")
        st.markdown(
            """
        - **MMORE RAG**: Multimodal document Q&A (text, images, tables)
        - **HPC Integration**: SLURM job management via SSH
        - **3D Printing**: Prusa Connect + PrusaSlicer integration
        - **Engineering**: EngiBench topology optimization
        - **Research**: ArXiv papers + web search
        - **GUI Apps**: Open PrusaSlicer, Blender, VS Code, etc.
        """
        )


def render() -> None:
    """Render the home page."""
    _render_header()
    _render_agent_grid()
    st.markdown("---")
    _render_getting_started()
    st.markdown("---")
    _render_system_info()


if __name__ == "__main__":
    render()
