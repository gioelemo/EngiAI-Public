"""W&B Report page for the EngiAI Streamlit app."""

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config  # noqa: E402


def render() -> None:
    """Render the Weights & Biases report page."""
    st.markdown("# 📊 Weights & Biases Training Report")

    # Check if W&B report URL is configured
    if not config.wandb_report_url:
        st.warning(
            "⚠️ **W&B Report URL not configured**\n\n"
            "To view your training reports here, add your W&B report URL to the `.env` file:\n\n"
            "```bash\n"
            'WANDB_REPORT_URL="https://wandb.ai/your-entity/your-project/reports/Your-Report--VmlldzoxMjM0NTY"\n'
            "```\n\n"
            "You can create reports in your W&B workspace and copy the URL."
        )
        return

    st.markdown(
        f"View your training metrics and experiment tracking: "
        f"[Open in W&B ↗]({config.wandb_report_url})"
    )

    # Embed the W&B report using iframe
    iframe_html = f"""
    <iframe
        src="{config.wandb_report_url}"
        style="border:none; width:100%; height:1024px; border-radius: 8px;"
        title="Weights & Biases Training Report">
    </iframe>
    """

    components.html(iframe_html, height=1050, scrolling=True)

    st.markdown("---")
    st.markdown(
        "💡 **Tip:** You can update the report URL in your `.env` file to display different reports."
    )


if __name__ == "__main__":
    render()
