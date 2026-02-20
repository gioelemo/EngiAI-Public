"""HPC workflow scorer for hpc_train_beams2d evaluations.

Scores the agent's ability to orchestrate end-to-end cGAN training on HPC:
1. generate_training_command — correct seed/epochs/algorithm
2. submit_slurm_job — job submitted to Euler
3. monitor_job_until_complete — job monitored to completion
4. evaluate_model — run EngiOpt evaluation script to compute metrics

Primary metric: step_completion_rate (completed steps / total steps).
Secondary: training config correctness, evaluation metrics extraction.
"""

import contextlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Expected workflow steps in order
WORKFLOW_STEPS = [
    "generate_training_command",
    "submit_slurm_job",
    "monitor_job_until_complete",
    "evaluate_model",
]


def _check_training_config(
    tool_calls_info: list[dict[str, Any]],
    training_config: dict[str, Any],
) -> bool:
    """Check if generate_training_command was called with correct config."""
    for tc in tool_calls_info:
        if tc.get("name") != "generate_training_command":
            continue
        args = tc.get("args", {})
        # The tool takes a TrainingConfig dataclass, but LLMs may pass it as a dict
        # Check top-level args or nested 'cfg' dict
        cfg = args.get("cfg", args)
        if isinstance(cfg, dict):
            seed_match = cfg.get("seed") == training_config.get("seed")
            epochs_match = cfg.get("epochs") == training_config.get("epochs")
            algo_match = cfg.get("algorithm", "cgan_cnn_2d") == training_config.get(
                "algorithm", "cgan_cnn_2d"
            )
            if seed_match and epochs_match and algo_match:
                return True
    return False


def _check_tool_called(tool_calls_info: list[dict[str, Any]], tool_name: str) -> bool:
    """Check if a tool was called at least once."""
    return any(tc.get("name") == tool_name for tc in tool_calls_info)


def _check_evaluate_model(
    tool_calls_info: list[dict[str, Any]],
    messages: list,  # noqa: ARG001
) -> bool:
    """Check if the evaluate_model tool was called.

    Looks for the evaluate_model tool call in tool_calls_info.
    Also accepts execute_cli_command with an EngiOpt evaluate command as fallback.
    """
    for tc in tool_calls_info:
        name = tc.get("name", "")
        # Primary: the dedicated engineering tool
        if name == "evaluate_model":
            return True
        # Fallback: CLI execution of the evaluation script
        if name == "execute_cli_command":
            command = tc.get("args", {}).get("command", "")
            if "evaluate_" in command and "engiopt" in command:
                return True

    return False


def _extract_evaluation_metrics(messages: list) -> dict[str, float]:
    """Extract evaluation metrics from CLI output in messages.

    Looks for metric values (IOG, COG, FOG, MMD, DPP, viol) in
    execute_cli_command results or agent responses.
    """
    metrics: dict[str, float] = {}
    metric_names = ["IOG", "COG", "FOG", "MMD", "DPP", "viol"]

    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue

        for metric_name in metric_names:
            if metric_name in content and metric_name not in metrics:
                # Try to extract numeric value after metric name
                # Patterns: "IOG: 0.123", "IOG=0.123", "'IOG': 0.123"
                patterns = [
                    rf"{metric_name}\s*[:=]\s*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)",
                    rf"'{metric_name}'\s*:\s*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)",
                    rf'"{metric_name}"\s*:\s*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, content)
                    if match:
                        with contextlib.suppress(ValueError, TypeError):
                            metrics[metric_name] = float(match.group(1))
                        break

    return metrics


def score_hpc_workflow(
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score HPC cGAN training workflow execution.

    Args:
        output: Agent output with tool_calls_info, messages, response
        target: Ground truth (empty for this problem type)
        metadata: Problem metadata including training_config, expected_workflow_steps

    Returns:
        Dict with workflow completion metrics and extracted evaluation data.
    """
    tool_calls_info = output.get("tool_calls_info", [])
    messages = output.get("messages", [])
    example_id = metadata.get("example_id", -1)
    training_config = metadata.get("training_config", {})

    # --- Step completion scoring ---
    steps_completed: dict[str, bool] = {}

    # 1. generate_training_command called (config correctness scored separately)
    steps_completed["generate_training_command"] = _check_tool_called(
        tool_calls_info, "generate_training_command"
    )

    # 2. submit_slurm_job called
    steps_completed["submit_slurm_job"] = _check_tool_called(
        tool_calls_info, "submit_slurm_job"
    )

    # 3. monitor_job_until_complete called
    steps_completed["monitor_job_until_complete"] = _check_tool_called(
        tool_calls_info, "monitor_job_until_complete"
    )

    # 4. evaluate_model via dedicated tool or CLI fallback
    steps_completed["evaluate_model"] = _check_evaluate_model(tool_calls_info, messages)

    completed_count = sum(1 for v in steps_completed.values() if v)
    total_steps = len(steps_completed)
    step_completion_rate = completed_count / total_steps if total_steps > 0 else 0.0

    # --- Evaluation metrics extraction (for analysis) ---
    eval_metrics = _extract_evaluation_metrics(messages)

    # Evaluation score: did the script run and produce metrics?
    eval_metrics_score = min(1.0, len(eval_metrics) / 6) if eval_metrics else 0.0

    # --- Composite score ---
    # Weighted: workflow completion (85%) + evaluation quality (15%)
    # Tool efficiency is scored separately by tool_use_scorer.
    hpc_workflow_score = 0.85 * step_completion_rate + 0.15 * eval_metrics_score

    logger.info(
        "Example %d: step_completion=%d/%d (%.2f), eval_metrics=%d, score=%.3f",
        example_id,
        completed_count,
        total_steps,
        step_completion_rate,
        len(eval_metrics),
        hpc_workflow_score,
    )

    return {
        # Primary metric
        "hpc_workflow_score": hpc_workflow_score,
        # Workflow completion details
        "step_completion_rate": step_completion_rate,
        "steps_completed_count": completed_count,
        "steps_total": total_steps,
        **{f"step_{k}": v for k, v in steps_completed.items()},
        # Evaluation metrics (from EngiOpt evaluation script output)
        "eval_metrics": eval_metrics,
        "eval_metrics_count": len(eval_metrics),
        "eval_metrics_score": eval_metrics_score,
        **{f"eval_{k}": v for k, v in eval_metrics.items()},
        # Metadata
        "training_config_correct": _check_training_config(
            tool_calls_info, training_config
        ),
        "example_id": example_id,
    }
