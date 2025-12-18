#!/usr/bin/env python3
"""
Example script demonstrating Weave integration for LLM tracing and benchmarking.

This script shows how to:
1. Initialize Weave tracing
2. Create traced functions
3. Create evaluation datasets
4. Run benchmarks
5. Log evaluation results

Usage:
    python scripts/weave_example.py
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.utils.weave_integration import (
    create_dataset,
    init_weave,
    is_weave_enabled,
    log_evaluation,
    traced,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@traced
def simple_llm_call(prompt: str) -> str:
    """
    Example of a simple LLM call with Weave tracing.

    This function will be automatically traced by Weave when enabled.

    Args:
        prompt: The input prompt

    Returns:
        A simulated response (in real usage, this would call an actual LLM)
    """
    # Simulate an LLM call (in real usage, this would call an actual LLM)
    # For demonstration purposes, return a simple response
    if "stress" in prompt.lower():
        return "Stress is force per unit area, measured in Pascals (Pa)"
    elif "young" in prompt.lower() or "modulus" in prompt.lower():
        return "Young's modulus is a measure of material stiffness"
    else:
        return f"Response to: {prompt}"


@traced
def engineering_qa(question: str, context: str | None = None) -> dict:
    """
    Example of a more complex function with multiple inputs and structured output.

    Args:
        question: The engineering question
        context: Optional context for the question

    Returns:
        Dictionary with answer and metadata
    """
    # Simulate a more complex LLM interaction
    answer = simple_llm_call(question)

    return {
        "question": question,
        "answer": answer,
        "context_used": context is not None,
        "confidence": 0.95,
    }


def run_simple_example():
    """Run a simple tracing example."""
    logger.info("=== Simple Tracing Example ===")

    # Make some traced calls
    result1 = simple_llm_call("What is stress in materials science?")
    logger.info(f"Result 1: {result1}")

    result2 = simple_llm_call("Explain Young's modulus")
    logger.info(f"Result 2: {result2}")

    result3 = engineering_qa(
        "What is the difference between stress and strain?",
        context="Material mechanics",
    )
    logger.info(f"Result 3: {result3}")


def run_benchmark_example():
    """Run a benchmark evaluation example."""
    logger.info("\n=== Benchmark Evaluation Example ===")

    # Create a benchmark dataset
    benchmark_data = [
        {
            "id": 1,
            "input": "What is stress?",
            "expected_output": "Force per unit area",
            "category": "definitions",
        },
        {
            "id": 2,
            "input": "What is Young's modulus?",
            "expected_output": "Measure of stiffness",
            "category": "definitions",
        },
        {
            "id": 3,
            "input": "Calculate stress for 1000N on 10mm²",
            "expected_output": "100 MPa",
            "category": "calculations",
        },
    ]

    # Create the dataset in Weave
    dataset = create_dataset(name="engineering_qa_example_v1", rows=benchmark_data)

    if dataset is None:
        logger.warning("Could not create dataset (Weave may be disabled)")
        return

    # Run evaluation on the dataset
    predictions = []
    for item in benchmark_data:
        response = simple_llm_call(item["input"])
        predictions.append(response)
        logger.info(f"Question: {item['input']}")
        logger.info(f"Expected: {item['expected_output']}")
        logger.info(f"Predicted: {response}\n")

    # Calculate some simple metrics
    # In real usage, you would use proper evaluation metrics
    scores = {
        "num_predictions": len(predictions),
        "avg_response_length": sum(len(p) for p in predictions) / len(predictions),
        "example_metric": 0.85,  # Placeholder for actual metrics
    }

    # Log the evaluation results
    log_evaluation(
        dataset_name="engineering_qa_example_v1", predictions=predictions, scores=scores
    )

    logger.info(f"Evaluation scores: {scores}")


def main():
    """Main function."""
    logger.info("Starting Weave integration example")
    logger.info(f"Weave configuration: USE_WEAVE={config.use_weave}")
    logger.info(f"Weave project: {config.weave_project}")

    # Initialize Weave
    if init_weave():
        logger.info("✓ Weave tracing initialized successfully")
    else:
        logger.warning(
            "✗ Weave tracing is not active. "
            "Set USE_WEAVE=true in .env to enable tracing."
        )
        logger.info("Continuing without tracing (functions will still work)")

    # Run examples
    try:
        run_simple_example()
        run_benchmark_example()

        if is_weave_enabled():
            logger.info("\n✓ Examples completed! Check the Weave UI for traces:")
            logger.info(
                f"   https://wandb.ai/{config.weave_project.replace('/', '/').split('/')[0]}/"
                f"{config.weave_project.replace('/', '/').split('/')[1]}/weave"
            )
        else:
            logger.info("\n✓ Examples completed (without Weave tracing)")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
