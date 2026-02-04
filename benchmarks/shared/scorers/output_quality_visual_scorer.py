"""Generic scorer that works for any topology optimization problem.

This scorer uses problem configuration to extract objectives, check constraints,
and compute scores without problem-specific code.

Design Quality Metrics:
-----------------------
The scorer tracks two complementary types of connectivity/printability metrics:

1. **2D Connectivity** (num_components, connected_design):
   - Checks if the design array has disconnected regions at the density level
   - Uses scipy.ndimage.label with 8-connectivity on the numpy array
   - This is a PRECURSOR check before STL extrusion
   - Computed from the design array directly

2. **3D Watertightness** (is_watertight, volume_mm3, surface_area_mm2):
   - Checks if the extruded mesh forms a closed manifold solid (no holes/gaps)
   - Uses trimesh validation on the generated STL file
   - This is the DEFINITIVE 3D printability check
   - Extracted from STL export tool results

**Important**: num_components==1 is necessary but NOT sufficient for watertightness.
A design can be connected in 2D but non-watertight in 3D due to mesh generation issues.
Both metrics provide valuable complementary information about design quality.
"""

import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import label

from benchmarks.shared.objective_extractor import (
    calculate_objective_score,
    extract_objectives_from_tool_messages,
)
from benchmarks.shared.problem_registry import get_problem_config
from benchmarks.shared.utils import (
    create_design_comparison,
    extract_design_from_tool_messages,
    extract_optimization_history_from_tool_messages,
    get_hf_dataset,
)

logger = logging.getLogger(__name__)

# Quality thresholds for topology optimization
BINARY_THRESHOLD = 0.5  # Threshold for converting density to binary (material vs void)


def _get_design_array(
    output: dict[str, Any],
    _metadata: dict[str, Any],
    _problem_config: Any,
    example_id: int,
) -> np.ndarray | None:
    """Extract design array from messages.

    Args:
        output: Agent output with messages
        metadata: Metadata dict
        problem_config: Problem configuration
        example_id: Example ID for logging

    Returns:
        Design array or None if not found
    """
    messages = output.get("messages", [])
    design_array = extract_design_from_tool_messages(messages, example_id)

    if design_array is None:
        logger.warning(f"Example {example_id}: No design array found in messages")
    else:
        logger.debug(
            f"Example {example_id}: Extracted design with shape {design_array.shape}"
        )

    return design_array


def _calculate_design_metrics(
    design_array: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Calculate similarity metrics between agent and ground truth designs.

    These metrics are universal across all topology optimization problems.

    Args:
        design_array: Agent's optimized design
        ground_truth: Ground truth optimal design

    Returns:
        Dictionary with IoU, pixel accuracy, MSE metrics
    """
    agent_binary = (design_array > BINARY_THRESHOLD).astype(int)
    gt_binary = (ground_truth > BINARY_THRESHOLD).astype(int)

    intersection = np.logical_and(agent_binary, gt_binary).sum()
    union = np.logical_or(agent_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else 0.0

    pixel_accuracy = np.mean(agent_binary == gt_binary)
    mse = np.mean((design_array - ground_truth) ** 2)

    return {
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
        "mse": float(mse),
    }


def _check_design_connectivity(design_array: np.ndarray) -> dict[str, Any]:
    """Check if design forms a single connected component.

    Uses 8-connectivity (diagonals count) which is appropriate for 3D printing
    since diagonal voxels share edges when extruded.

    This is a 2D precursor check before STL extrusion - it checks if the design
    array has disconnected regions at the density level.

    Args:
        design_array: Design array (continuous densities 0-1)

    Returns:
        Dictionary with connectivity info:
        - connected_design: True if single connected component
        - num_components: Number of separate material regions
    """
    binary = (design_array > BINARY_THRESHOLD).astype(int)
    structure = np.ones((3, 3))  # 8-connectivity
    _, num_components = label(binary, structure=structure)
    return {
        "connected_design": num_components == 1,
        "num_components": int(num_components),
    }


def _extract_watertightness_from_messages(
    messages: list, example_id: int
) -> dict[str, Any]:
    """Extract watertightness metrics from STL export tool calls.

    Searches for convert_design_to_stl tool results in messages and
    extracts 3D mesh validation metrics for printability assessment.

    This provides true 3D mesh validation (watertightness) which is
    complementary to 2D connectivity checks. A design can be connected
    in 2D but non-watertight in 3D due to mesh generation issues.

    Args:
        messages: LangChain messages from agent execution
        example_id: Example ID for logging

    Returns:
        Dictionary with watertightness metrics or empty dict if not found:
        - is_watertight: bool | None
        - volume_mm3: float | None
        - surface_area_mm2: float | None
        - num_vertices: int | None
        - num_faces: int | None
        - watertight_check_available: bool
        - mesh_validation_time: float
    """
    watertightness_metrics = {}

    for msg in messages:
        # Check if this is a tool message (has tool_call_id)
        if not hasattr(msg, "tool_call_id"):
            continue

        content = msg.content
        if not isinstance(content, str):
            continue

        # Check if this is a convert_design_to_stl tool result
        if (
            "is_watertight" not in content
            and "watertight_check_available" not in content
        ):
            continue

        logger.debug(
            f"Example {example_id}: Found watertightness info in STL export tool result"
        )

        # Try to parse as JSON first (most reliable)
        try:
            # Try direct JSON parse
            result = json.loads(content)
            if isinstance(result, dict) and "is_watertight" in result:
                watertightness_metrics = {
                    "is_watertight": result.get("is_watertight"),
                    "volume_mm3": result.get("volume_mm3"),
                    "surface_area_mm2": result.get("surface_area_mm2"),
                    "num_vertices": result.get("num_vertices"),
                    "num_faces": result.get("num_faces"),
                    "watertight_check_available": result.get(
                        "watertight_check_available", False
                    ),
                    "mesh_validation_time": result.get("mesh_validation_time", 0.0),
                }
                # Continue to get the LAST occurrence (most recent STL export)
                continue
        except json.JSONDecodeError:
            pass

        # Fallback: regex extraction
        try:
            # Extract is_watertight
            wt_match = re.search(r"['\"]is_watertight['\"]\s*:\s*(\w+)", content)
            if wt_match:
                wt_value = wt_match.group(1)
                is_wt = (
                    True
                    if wt_value == "True"
                    else (False if wt_value == "False" else None)
                )

                watertightness_metrics = {
                    "is_watertight": is_wt,
                    "volume_mm3": _extract_float_field(content, "volume_mm3"),
                    "surface_area_mm2": _extract_float_field(
                        content, "surface_area_mm2"
                    ),
                    "num_vertices": _extract_int_field(content, "num_vertices"),
                    "num_faces": _extract_int_field(content, "num_faces"),
                    "watertight_check_available": _extract_bool_field(
                        content, "watertight_check_available"
                    ),
                    "mesh_validation_time": _extract_float_field(
                        content, "mesh_validation_time"
                    )
                    or 0.0,
                }
        except Exception as e:
            logger.debug(
                f"Example {example_id}: Failed to extract watertightness via regex: {e}"
            )

    if not watertightness_metrics:
        logger.debug(
            f"Example {example_id}: No watertightness metrics found (STL not generated)"
        )

    return watertightness_metrics


def _extract_float_field(content: str, field: str) -> float | None:
    """Extract float field from content string."""
    match = re.search(rf"['\"]?{field}['\"]?\s*:([\d.\s]+)", content)
    if match:
        try:
            return float(match.group(1).strip())
        except ValueError:
            return None
    return None


def _extract_int_field(content: str, field: str) -> int | None:
    """Extract int field from content string."""
    match = re.search(rf"['\"]?{field}['\"]?\s*:(\d+)", content)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_bool_field(content: str, field: str) -> bool:
    """Extract bool field from content string."""
    match = re.search(rf"['\"]?{field}['\"]?\s*:\s*(\w+)", content)
    if match:
        value = match.group(1)
        return value == "True"
    return False


def _calculate_constraint_score(
    design_array: np.ndarray,
    conditions: dict[str, Any],
    problem_config: Any,
    example_id: int,
) -> tuple[float, dict[str, Any]]:
    """Calculate constraint matching score based on problem config.

    Args:
        design_array: Agent's design
        conditions: Target conditions from dataset
        problem_config: Problem configuration
        example_id: Example ID for logging

    Returns:
        Tuple of (constraint_score, detailed_metrics)
    """
    constraint_conditions = problem_config.get_constraint_conditions()

    if not constraint_conditions:
        logger.debug(f"Example {example_id}: No constraints to check")
        return 1.0, {}

    violations = 0
    metrics = {}

    for cond_config in constraint_conditions:
        # Get target value from conditions
        target_value = conditions.get(cond_config.field_name)
        if target_value is None:
            # Try aliases
            for alias in cond_config.aliases:
                target_value = conditions.get(alias)
                if target_value is not None:
                    break

        if target_value is None:
            logger.debug(
                f"Example {example_id}: No target value for {cond_config.name}"
            )
            continue

        # Check constraint based on type
        if cond_config.constraint_type == "equality":
            # For equality constraints on design array (e.g., volume fraction)
            if (
                cond_config.name == "volume_fraction"
                or "volume" in cond_config.name.lower()
            ):
                actual_value = np.mean(design_array)
                error = abs(actual_value - target_value)

                metrics[f"{cond_config.name}_actual"] = float(actual_value)
                metrics[f"{cond_config.name}_target"] = float(target_value)
                metrics[f"{cond_config.name}_error"] = float(error)

                # Check if within tolerance
                if error >= cond_config.tolerance:
                    violations += 1
                    logger.debug(
                        f"Example {example_id}: {cond_config.name} violated - "
                        f"actual={actual_value:.4f}, target={target_value:.4f}, "
                        f"error={error:.4f} >= tolerance={cond_config.tolerance}"
                    )

        elif cond_config.constraint_type == "inequality":
            # Placeholder for inequality constraints (can be extended)
            logger.debug(
                f"Example {example_id}: Inequality constraints not yet implemented"
            )

    # Calculate score: 1.0 if no violations, 0.0 if any violations
    # (Can be made more nuanced with partial credit)
    constraint_score = 1.0 if violations == 0 else 0.0
    metrics["constraint_violations"] = violations

    return constraint_score, metrics


def _save_comparison_image(
    comparison_image: Image.Image,
    output: dict[str, Any],
    metadata: dict[str, Any],
    example_id: int,
) -> dict[str, str]:
    """Save comparison image and return paths/encodings."""
    problem_type = metadata.get("problem_type", "unknown")
    model_name = output.get("model", "unknown")
    seed = metadata.get("seed")
    prompt_style = metadata.get("prompt_style", "full")
    rag_status = metadata.get("rag_status", "no_rag")

    # Sanitize model name for filesystem (replace / and : with _)
    model_safe = model_name.replace("/", "_").replace(":", "_")

    # Build output directory path, including seed subfolder if seed is provided
    # Use results/models/{model}/{problem}/{prompt_style}/{rag_status}/ structure
    base_dir = (
        Path(__file__).parent.parent.parent
        / "evaluations"
        / "results"
        / "models"
        / model_safe
        / problem_type
        / prompt_style
        / rag_status
        / "comparisons"
    )
    output_dir = base_dir / f"seed_{seed}" if seed is not None else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"comparison_example_{example_id}.png"
    comparison_image.save(image_path)

    buffered = io.BytesIO()
    comparison_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "comparison_image_path": str(image_path),
        "comparison_image_base64": f"data:image/png;base64,{img_str}",
    }


def _create_error_result(error_info: dict[str, Any]) -> dict[str, Any]:
    """Create a standardized error result with score=0.0."""
    return {"score": 0.0, **error_info}


def _validate_inputs(
    metadata: dict[str, Any],
) -> tuple[str, int, Any] | dict[str, Any]:
    """Validate metadata and get problem config.

    Returns (problem_name, example_id, problem_config) or error dict.
    """
    example_id = metadata.get("example_id", 0)
    problem_name = metadata.get("problem_type")

    if not problem_name:
        return _create_error_result({"error": "No problem_type in metadata"})

    try:
        problem_config = get_problem_config(problem_name)
    except ValueError as e:
        return _create_error_result({"error": str(e)})

    return problem_name, example_id, problem_config


def _load_dataset_and_ground_truth(
    metadata: dict[str, Any],
    problem_config: Any,
    example_id: int,
    design_array: np.ndarray,
) -> tuple[Any, np.ndarray] | dict[str, Any]:
    """Load dataset and ground truth, validate shape.

    Returns (hf_dataset, ground_truth) or error dict.
    """
    dataset_name = metadata.get("dataset_name", problem_config.dataset_name)
    dataset_split = metadata.get("dataset_split", "test")

    try:
        hf_dataset = get_hf_dataset(dataset_name, split=dataset_split)
    except Exception as e:
        return _create_error_result(
            {
                "design_found": True,
                "error": f"Failed to load dataset: {e}",
            }
        )

    if example_id >= len(hf_dataset):
        return _create_error_result(
            {
                "design_found": True,
                "reason": f"Invalid example_id: {example_id} for split: {dataset_split}",
            }
        )

    ground_truth = np.array(hf_dataset[example_id][problem_config.design_field])

    if design_array.shape != ground_truth.shape:
        return _create_error_result(
            {
                "design_found": True,
                "reason": f"Shape mismatch: agent={design_array.shape}, ground_truth={ground_truth.shape}",
                "agent_shape": str(design_array.shape),
                "gt_shape": str(ground_truth.shape),
            }
        )

    return hf_dataset, ground_truth


def _compute_hierarchical_score(
    problem_config: Any,
    metrics_data: dict[str, Any],
    design_found: bool,
    metadata: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Compute hierarchical composite score from multiple categories.

    Categories:
    1. Design Quality: IoU, pixel accuracy, constraint match, objective match
    2. Tool Efficiency: efficiency ratio, sequence score (from metadata)
    3. Task Completion: whether the task was completed successfully
    4. Printability: connectivity and watertightness

    Args:
        problem_config: Problem configuration with score_categories
        metrics_data: Dictionary containing all metrics:
            - design_metrics: dict with iou, pixel_accuracy
            - constraint_score: float
            - objective_score: float
            - connectivity_metrics: dict with connected_design, num_components
            - watertightness_metrics: dict with is_watertight, etc.
        design_found: Whether design was successfully found
        metadata: Metadata dict (may contain tool_calls_info from tool_use_scorer)

    Returns:
        Tuple of (overall_score, category_scores_dict)
    """
    # Extract metrics from metrics_data
    design_metrics = metrics_data.get("design_metrics", {})
    constraint_score = metrics_data.get("constraint_score", 0.0)
    objective_score = metrics_data.get("objective_score", 0.0)
    connectivity_metrics = metrics_data.get("connectivity_metrics", {})
    watertightness_metrics = metrics_data.get("watertightness_metrics", {})

    category_scores = {}

    # 1. Design Quality Category
    dq_weights = problem_config.get_metric_weights("design_quality")
    if dq_weights:
        design_quality_score = (
            dq_weights.get("iou", 0.0) * design_metrics.get("iou", 0.0)
            + dq_weights.get("pixel_accuracy", 0.0)
            * design_metrics.get("pixel_accuracy", 0.0)
            + dq_weights.get("constraint_match", 0.0) * constraint_score
            + dq_weights.get("objective_match", 0.0) * objective_score
        )
        category_scores["design_quality"] = float(design_quality_score)

    # 2. Tool Efficiency Category (extract from metadata if available)
    tool_efficiency_score = 0.0
    te_weights = problem_config.get_metric_weights("tool_efficiency")
    if te_weights:
        # Check if tool usage metrics are available in metadata
        # (they would be if tool_use_scorer ran and passed results)
        efficiency_ratio = metadata.get("efficiency_ratio", 1.0)  # Default to perfect
        sequence_score = metadata.get("sequence_score", 1.0)  # Default to perfect

        tool_efficiency_score = (
            te_weights.get("efficiency_ratio", 0.0) * efficiency_ratio
            + te_weights.get("sequence_score", 0.0) * sequence_score
        )
        category_scores["tool_efficiency"] = float(tool_efficiency_score)

    # 3. Task Completion Category
    tc_weights = problem_config.get_metric_weights("task_completion")
    if tc_weights:
        # Task completion: 1.0 if design found, 0.0 otherwise
        task_completion_score = 1.0 if design_found else 0.0
        category_scores["task_completion"] = float(task_completion_score)

    # 4. Printability Category
    pr_weights = problem_config.get_metric_weights("printability")
    if pr_weights:
        # Convert boolean metrics to 0/1
        connected = 1.0 if connectivity_metrics.get("connected_design", False) else 0.0
        watertight = 1.0 if watertightness_metrics.get("is_watertight", False) else 0.0

        printability_score = (
            pr_weights.get("connectivity", 0.0) * connected
            + pr_weights.get("watertightness", 0.0) * watertight
        )
        category_scores["printability"] = float(printability_score)

    # Compute overall score as weighted sum of category scores
    overall_score = 0.0
    for category_name, category_score in category_scores.items():
        category_weight = problem_config.get_category_weight(category_name)
        overall_score += category_weight * category_score

    return float(overall_score), category_scores


def score_output_quality_visual(
    output: dict[str, Any],
    target: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Visual quality scorer for topology optimization problems.

    Uses problem configuration to:
    - Extract design from messages
    - Calculate design similarity metrics (IoU, pixel accuracy, MSE)
    - Check constraints (volume fraction, etc.)
    - Extract and score objectives (compliance, overlap, etc.)
    - Compute weighted overall score

    Args:
        output: Agent output with messages and model info
        target: Target/ground truth data from dataset
        metadata: Metadata including problem_type, dataset_name, example_id

    Returns:
        Dictionary with score and detailed metrics
    """
    # Validate inputs and get problem config
    validation_result = _validate_inputs(metadata)
    if isinstance(validation_result, dict):
        return validation_result

    problem_name, example_id, problem_config = validation_result

    # Extract design array
    design_array = _get_design_array(output, metadata, problem_config, example_id)
    if design_array is None:
        return _create_error_result(
            {
                "design_found": False,
                "reason": "No design array found in tool results",
                "num_messages": len(output.get("messages", [])),
            }
        )

    # Load dataset and ground truth
    dataset_result = _load_dataset_and_ground_truth(
        metadata, problem_config, example_id, design_array
    )
    if isinstance(dataset_result, dict):
        return dataset_result

    hf_dataset, ground_truth = dataset_result

    # Calculate design metrics (universal)
    design_metrics = _calculate_design_metrics(design_array, ground_truth)

    # Check design connectivity (2D precursor check)
    connectivity_metrics = _check_design_connectivity(design_array)

    # Extract watertightness metrics from STL export (3D printability check)
    messages = output.get("messages", [])
    watertightness_metrics = _extract_watertightness_from_messages(messages, example_id)

    # Get conditions from target or dataset row
    conditions = target if isinstance(target, dict) else {}
    if not conditions and "conditions" in hf_dataset[example_id]:
        conditions = hf_dataset[example_id]["conditions"]

    # Calculate constraint matching score
    constraint_score, constraint_metrics = _calculate_constraint_score(
        design_array, conditions, problem_config, example_id
    )

    # Extract objectives from messages (reuse messages already extracted above)
    agent_objectives = extract_objectives_from_tool_messages(
        messages, problem_config, example_id
    )

    # Get target objectives from dataset
    target_objectives = {}
    for obj_config in problem_config.objectives:
        # Try to get from target dict
        value = (
            target.get(obj_config.target_field) if isinstance(target, dict) else None
        )

        # If not in target, try dataset row directly
        if value is None:
            value = hf_dataset[example_id].get(obj_config.target_field)

        target_objectives[obj_config.name] = value

    # Calculate objective matching score
    objective_score, objective_metrics = calculate_objective_score(
        agent_objectives, target_objectives, problem_config, example_id
    )

    # Compute hierarchical composite score
    overall_score, category_scores = _compute_hierarchical_score(
        problem_config=problem_config,
        metrics_data={
            "design_metrics": design_metrics,
            "constraint_score": constraint_score,
            "objective_score": objective_score,
            "connectivity_metrics": connectivity_metrics,
            "watertightness_metrics": watertightness_metrics,
        },
        design_found=True,
        metadata=metadata,
    )

    # Extract optimization history from messages for global metrics
    optimization_history = extract_optimization_history_from_tool_messages(
        messages, example_id
    )

    # Build result dictionary
    result: dict[str, Any] = {
        "score": float(overall_score),  # Hierarchical composite score
        "design_found": True,
        "problem_type": problem_name,
        "constraint_score": float(constraint_score),
        "objective_score": float(objective_score),
        "example_id": example_id,  # Store for correct mapping in global metrics
        "design": design_array.tolist(),  # Store design for global metrics computation
        "optimization_history": optimization_history,  # Store for optimality gap metrics
        # Category scores
        **{f"{cat}_score": score for cat, score in category_scores.items()},
        **design_metrics,
        **connectivity_metrics,
        **watertightness_metrics,  # 3D printability metrics
        **constraint_metrics,
        **objective_metrics,
    }

    # Save comparison image if possible
    try:
        # Pass problem_type and conditions for physics-based visualizations
        comparison_image = create_design_comparison(
            design_array,
            ground_truth,
            example_id,
            problem_type=problem_name,
            conditions=conditions,
        )
        if comparison_image is not None:
            image_data = _save_comparison_image(
                comparison_image, output, metadata, example_id
            )
            result.update(image_data)
            result["comparison_image_generated"] = True
    except Exception as e:
        logger.debug(f"Example {example_id}: Failed to create comparison image: {e}")
        result["comparison_image_generated"] = False

    return result
