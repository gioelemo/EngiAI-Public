"""Human clarification tool for engineering agent.

Provides a tool that allows the LLM to request clarification from the user
when design parameters are ambiguous or missing from the prompt.
"""

import json

from langchain_core.tools import tool


@tool
def ask_human_for_clarification(clarification_request: str) -> str:
    """Ask the user for clarification when design parameters are ambiguous or missing.

    Use this tool when the user's request does NOT specify exact numerical values
    for required design parameters (e.g., volume fraction, force distance,
    filter radius). Do NOT guess or use default values — ask the user instead.

    Args:
        clarification_request: A clear question asking the user for the specific
            missing design parameters. Include what parameters are needed and their
            valid ranges if known. Example: "What volume fraction should I use for
            this beam design? Please specify a value between 0.1 and 0.9."

    Returns:
        A JSON string with structured clarification data.
    """
    result = {
        "success": True,
        "question": clarification_request,
        "message": (
            "Clarification requested. Awaiting user response. "
            "Do not proceed with design tools until the user provides the required parameters."
        ),
    }
    return json.dumps(result)
