"""Human clarification tool for engineering agent.

Provides a tool that allows the LLM to request clarification from the user
when design parameters are ambiguous or missing from the prompt.
"""

from langchain_core.tools import tool


@tool
def ask_human_for_clarification(question: str) -> str:
    """Ask the user for clarification when design parameters are ambiguous or missing.

    Use this tool when the user's request does NOT specify exact numerical values
    for required design parameters (e.g., volume fraction, force distribution,
    filter radius). Do NOT guess or use default values — ask the user instead.

    After calling this tool, do NOT call any further tools. End your response
    and wait for the user's reply.

    Args:
        question: A clear question asking the user for the specific missing
            parameters. Include what parameters are needed and their valid ranges
            if known.

    Returns:
        A confirmation message indicating the clarification was requested.
    """
    return (
        f"Clarification requested: {question}\n\n"
        "Awaiting user response. Do not proceed with design tools "
        "until the user provides the required parameters."
    )
