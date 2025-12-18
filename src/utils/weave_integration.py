"""
Weave integration utilities for LLM tracing and benchmarking.

This module provides utilities for integrating Weights & Biases Weave
for tracking LLM calls. After calling weave.init(), LangChain components
are automatically traced - no decorators needed.

See: https://docs.wandb.ai/weave/guides/integrations/langchain
"""

import logging

from config import config

logger = logging.getLogger(__name__)


def init_weave() -> bool:
    """
    Initialize Weave tracing for the project.

    This automatically enables tracing for LangChain components.
    No decorators or manual instrumentation needed.

    Returns:
        True if Weave was successfully initialized, False otherwise.
    """
    return config.setup_weave_tracing()


def is_weave_enabled() -> bool:
    """
    Check if Weave tracing is enabled.

    Returns:
        True if Weave is enabled and available, False otherwise.
    """
    if not config.use_weave:
        return False

    try:
        import weave  # noqa: PLC0415, F401
    except ImportError:
        return False
    else:
        return True
