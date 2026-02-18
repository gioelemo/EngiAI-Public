"""HPC workflow scorer for hpc_train_beams2d evaluations.

Scores the agent's ability to orchestrate end-to-end cGAN training on HPC:
1. generate_training_command — correct seed/epochs/algorithm
2. submit_slurm_job — job submitted to Euler
3. monitor_job_until_complete — job monitored to completion
4. download_wandb_model — trained model downloaded
5. sample_designs_from_model — designs generated from trained model
6. simulate_design — designs simulated (>= expected count)

Primary metric: step_completion_rate (completed steps / total steps).
Secondary: design count validation, compliance extraction for offline comparison.
"""

import ast
import contextlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Expected workflow steps in order
WORKFLOW_STEPS = [
    "generate_training_command",
    "submit_slurm_job",
    "monitor_job_until_complete",
    "download_wandb_model",
    "sample_designs_from_model",
    "simulate_design",
]


def _parse_tool_result(content: str) -> dict[str, Any] | None:
    """Parse tool result from message content string."""
    if not isinstance(content, str):
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        json_content = content.replace("'", '"')
        json_content = json_content.replace("True", "true")
        json_content = json_content.replace("False", "false")
        json_content = json_content.replace("None", "null")
        json_content = re.sub(r"array\((.*?)\)", r"\1", json_content)
        return json.loads(json_content)
    except (json.JSONDecodeError, Exception):
        pass

    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        pass

    return None


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


def _count_tool_calls(tool_calls_info: list[dict[str, Any]], tool_name: str) -> int:
    """Count how many times a tool was called."""
    return sum(1 for tc in tool_calls_info if tc.get("name") == tool_name)


def _extract_compliance_values(messages: list) -> list[float]:
    """Extract compliance values from simulate_design tool results in messages."""
    compliance_values = []
    for msg in messages:
        # Tool messages have a 'name' attribute
        msg_name = getattr(msg, "name", None)
        if msg_name != "simulate_design":
            continue
        content = getattr(msg, "content", "")
        parsed = _parse_tool_result(content)
        if parsed is None:
            continue
        # Look for compliance in the result
        for key in ("compliance", "final_compliance", "c", "final_c"):
            val = parsed.get(key)
            if val is not None:
                with contextlib.suppress(TypeError, ValueError):
                    compliance_values.append(float(val))
                break
    return compliance_values


def _extract_design_file_paths(messages: list) -> list[str]:
    """Extract design .npy file paths from sample_designs_from_model results."""
    paths = []
    for msg in messages:
        msg_name = getattr(msg, "name", None)
        if msg_name != "sample_designs_from_model":
            continue
        content = getattr(msg, "content", "")
        parsed = _parse_tool_result(content)
        if parsed is None:
            continue
        design_files = parsed.get("design_files", [])
        if isinstance(design_files, list):
            paths.extend(design_files)
    return paths


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
        Dict with workflow completion metrics and extracted data for offline comparison.
    """
    tool_calls_info = output.get("tool_calls_info", [])
    messages = output.get("messages", [])
    example_id = metadata.get("example_id", -1)
    training_config = metadata.get("training_config", {})
    n_expected_designs = metadata.get("n_design_conditions", 5)

    # --- Step completion scoring ---
    steps_completed: dict[str, bool] = {}

    # 1. generate_training_command with correct config
    steps_completed["generate_training_command"] = _check_training_config(
        tool_calls_info, training_config
    )

    # 2. submit_slurm_job called
    steps_completed["submit_slurm_job"] = _check_tool_called(
        tool_calls_info, "submit_slurm_job"
    )

    # 3. monitor_job_until_complete called
    steps_completed["monitor_job_until_complete"] = _check_tool_called(
        tool_calls_info, "monitor_job_until_complete"
    )

    # 4. download_wandb_model called
    steps_completed["download_wandb_model"] = _check_tool_called(
        tool_calls_info, "download_wandb_model"
    )

    # 5. sample_designs_from_model called
    steps_completed["sample_designs_from_model"] = _check_tool_called(
        tool_calls_info, "sample_designs_from_model"
    )

    # 6. simulate_design called >= n_expected_designs times
    sim_count = _count_tool_calls(tool_calls_info, "simulate_design")
    steps_completed["simulate_design"] = sim_count >= n_expected_designs

    completed_count = sum(1 for v in steps_completed.values() if v)
    total_steps = len(steps_completed)
    step_completion_rate = completed_count / total_steps if total_steps > 0 else 0.0

    # --- Design data extraction (for offline comparison) ---
    compliance_values = _extract_compliance_values(messages)
    design_file_paths = _extract_design_file_paths(messages)

    n_designs_generated = len(design_file_paths)
    n_simulations_completed = len(compliance_values)

    # Design generation score (binary: did the agent generate enough designs?)
    designs_generated_score = (
        1.0
        if n_designs_generated >= n_expected_designs
        else (
            n_designs_generated / n_expected_designs if n_expected_designs > 0 else 0.0
        )
    )

    # Simulations score (binary: did the agent simulate enough designs?)
    simulations_completed_score = (
        1.0
        if n_simulations_completed >= n_expected_designs
        else (
            n_simulations_completed / n_expected_designs
            if n_expected_designs > 0
            else 0.0
        )
    )

    # --- Composite score ---
    # Weighted combination per score_categories in problem_registry
    hpc_workflow_score = (
        0.50 * step_completion_rate
        + 0.35 * (0.30 * designs_generated_score + 0.70 * simulations_completed_score)
        + 0.15 * 1.0  # Tool efficiency scored by tool_use_scorer separately
    )

    logger.info(
        "Example %d: step_completion=%d/%d (%.2f), designs=%d, simulations=%d, score=%.3f",
        example_id,
        completed_count,
        total_steps,
        step_completion_rate,
        n_designs_generated,
        n_simulations_completed,
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
        # Design quality data (for offline comparison)
        "compliance_values": compliance_values,
        "design_file_paths": design_file_paths,
        "n_designs_generated": n_designs_generated,
        "n_simulations_completed": n_simulations_completed,
        "designs_generated_score": designs_generated_score,
        "simulations_completed_score": simulations_completed_score,
        # Metadata
        "training_config_correct": steps_completed.get(
            "generate_training_command", False
        ),
        "example_id": example_id,
    }
