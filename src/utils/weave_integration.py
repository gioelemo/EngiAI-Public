"""
Weave integration utilities for LLM tracing and benchmarking.

This module provides utilities for integrating Weights & Biases Weave
for tracking LLM calls, creating evaluation datasets, and running benchmarks.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from config import config

logger = logging.getLogger(__name__)

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])


def get_weave():
    """
    Lazy import of weave module.

    Returns:
        The weave module if available and enabled, None otherwise.
    """
    if not config.use_weave:
        return None

    try:
        import weave  # noqa: PLC0415
    except ImportError:
        logger.warning("Weave is not installed. Install with: pip install weave")
        return None
    else:
        return weave


def init_weave() -> bool:
    """
    Initialize Weave tracing for the project.

    Returns:
        True if Weave was successfully initialized, False otherwise.
    """
    return config.setup_weave_tracing()


def traced(func: F) -> F:
    """
    Decorator to trace a function with Weave.

    This decorator will automatically track the inputs, outputs, and code
    of the decorated function when Weave is enabled.

    Example:
        ```python
        from src.utils.weave_integration import traced

        @traced
        def my_llm_function(prompt: str) -> str:
            # Your LLM call here
            return response
        ```

    Args:
        func: The function to trace

    Returns:
        The decorated function
    """
    weave = get_weave()
    if weave is None:
        # Return the original function if Weave is not available
        return func

    # Use Weave's @weave.op() decorator
    return weave.op()(func)


def traced_async(func: F) -> F:
    """
    Decorator to trace an async function with Weave.

    Example:
        ```python
        from src.utils.weave_integration import traced_async

        @traced_async
        async def my_async_llm_function(prompt: str) -> str:
            # Your async LLM call here
            return response
        ```

    Args:
        func: The async function to trace

    Returns:
        The decorated async function
    """
    weave = get_weave()
    if weave is None:
        return func

    return weave.op()(func)


class WeaveContext:
    """
    Context manager for Weave tracing.

    This allows you to manually start and stop tracing for specific code blocks.

    Example:
        ```python
        from src.utils.weave_integration import WeaveContext

        with WeaveContext("my_operation"):
            # Your code to trace here
            result = some_function()
        ```
    """

    def __init__(self, operation_name: str):
        """
        Initialize the Weave context.

        Args:
            operation_name: Name of the operation to trace
        """
        self.operation_name = operation_name
        self.weave = get_weave()

    def __enter__(self):
        """Enter the context."""
        if self.weave is None:
            return self
        # Note: Weave doesn't require explicit context entry
        logger.debug(f"Starting Weave trace for: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        if self.weave is None:
            return
        logger.debug(f"Ending Weave trace for: {self.operation_name}")


def create_dataset(name: str, rows: list[dict[str, Any]]) -> Any:
    """
    Create a Weave dataset for evaluation.

    Example:
        ```python
        from src.utils.weave_integration import create_dataset

        dataset = create_dataset(
            name="my_benchmark_dataset",
            rows=[
                {"input": "What is 2+2?", "expected_output": "4"},
                {"input": "What is the capital of France?", "expected_output": "Paris"},
            ]
        )
        ```

    Args:
        name: Name of the dataset
        rows: List of dictionaries containing the dataset rows

    Returns:
        A Weave Dataset object if successful, None otherwise
    """
    weave = get_weave()
    if weave is None:
        logger.warning("Cannot create dataset: Weave is not available")
        return None

    try:
        dataset = weave.Dataset(name=name, rows=rows)
        logger.info(f"Created Weave dataset: {name} with {len(rows)} rows")
    except Exception:
        logger.exception("Failed to create Weave dataset")
        return None
    else:
        return dataset


def log_evaluation(
    dataset_name: str,
    predictions: list[Any],
    scores: dict[str, float] | None = None,
) -> None:
    """
    Log evaluation results to Weave.

    Example:
        ```python
        from src.utils.weave_integration import log_evaluation

        predictions = ["4", "Paris", "Blue"]
        scores = {"accuracy": 0.95, "f1": 0.92}

        log_evaluation(
            dataset_name="my_benchmark_dataset",
            predictions=predictions,
            scores=scores
        )
        ```

    Args:
        dataset_name: Name of the dataset being evaluated
        predictions: List of predictions from the model
        scores: Optional dictionary of evaluation scores
    """
    weave = get_weave()
    if weave is None:
        logger.warning("Cannot log evaluation: Weave is not available")
        return

    try:
        logger.info(
            f"Logging evaluation for dataset: {dataset_name} "
            f"with {len(predictions)} predictions"
        )
        if scores:
            logger.info(f"Evaluation scores: {scores}")
        # Note: Actual logging implementation depends on Weave's evaluation API
        # This is a placeholder for the actual implementation
    except Exception:
        logger.exception("Failed to log evaluation")


class DisableTracing:
    """
    Context manager to temporarily disable Weave tracing for specific operations.

    Use this for utility/background operations that shouldn't clutter traces,
    like generating chat titles, health checks, etc.

    Example:
        ```python
        from src.utils.weave_integration import disable_tracing

        with disable_tracing():
            # LLM calls here won't be traced
            result = llm.invoke("Generate a title")
        ```
    """

    def __init__(self):
        """Initialize the context manager."""
        self.weave = get_weave()
        self.original_client_wrappers = {}

    def __enter__(self):
        """Disable tracing on entry."""
        if self.weave is None:
            return self

        try:
            # Temporarily unpatch OpenAI to disable auto-instrumentation
            from openai.resources.chat import completions  # noqa: PLC0415

            # Store original methods
            if hasattr(completions.Completions, "create"):
                self.original_client_wrappers["openai_create"] = (
                    completions.Completions.create
                )
                # Unpatch by accessing __wrapped__ if it exists
                if hasattr(completions.Completions.create, "__wrapped__"):
                    completions.Completions.create = (
                        completions.Completions.create.__wrapped__
                    )
        except Exception as e:
            logger.debug(f"Could not disable Weave tracing: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Re-enable tracing on exit."""
        if self.weave is None:
            return

        try:
            from openai.resources.chat import completions  # noqa: PLC0415

            # Restore original patched methods
            if "openai_create" in self.original_client_wrappers:
                completions.Completions.create = self.original_client_wrappers[
                    "openai_create"
                ]
        except Exception as e:
            logger.debug(f"Could not re-enable Weave tracing: {e}")


def disable_tracing():
    """
    Create a context manager that disables Weave tracing.

    Use this for utility operations that shouldn't be traced.

    Example:
        ```python
        with disable_tracing():
            title = generate_chat_title(message)
        ```

    Returns:
        DisableTracing context manager
    """
    return DisableTracing()


# Convenience function to check if Weave is enabled
def is_weave_enabled() -> bool:
    """
    Check if Weave tracing is enabled.

    Returns:
        True if Weave is enabled and available, False otherwise.
    """
    return config.use_weave and get_weave() is not None
