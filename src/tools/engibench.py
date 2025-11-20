"""
EngiBench tools for engineering design optimization.

EngiBench (https://engibench.ethz.ch) is a library for benchmarking
engineering design problems. It provides access to various optimization
problems like beam design, airfoils, truss structures, etc.

These tools allow LLM agents to interact with EngiBench simulators.
"""

import datetime
import random
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from engibench.problems.beams2d.v0 import Beams2D  # type: ignore[import-untyped]
from engibench.problems.thermoelastic2d.v0 import (
    ThermoElastic2D,  # type: ignore[import-untyped]
)
from langchain_core.tools import tool

matplotlib.use("Agg")  # Use non-interactive backend

# Constants
EXPECTED_ARRAY_DIMENSIONS = 2  # For 2D beam design arrays

# Problem registry - maps problem type names to their classes
PROBLEM_REGISTRY: dict[str, type] = {
    "beams2d": Beams2D,
    "thermoelastic2d": ThermoElastic2D,
}

# Unified state management for all problem types
# Each problem type has its own instance and last_design
_problem_states: dict[str, dict[str, Any]] = {
    "beams2d": {"problem_instance": None, "last_design": None},
    "thermoelastic2d": {"problem_instance": None, "last_design": None},
}


# Unified helper functions for problem management
def get_problem_class(problem_type: str) -> type:
    """Get the problem class for a given problem type."""
    problem_key = problem_type.lower()
    if problem_key not in PROBLEM_REGISTRY:
        msg = f"Unknown problem type: {problem_type}. Available types: {', '.join(PROBLEM_REGISTRY.keys())}"
        raise ValueError(msg)
    return PROBLEM_REGISTRY[problem_key]


def get_problem_state(problem_type: str) -> dict[str, Any]:
    """Get the state dictionary for a given problem type."""
    problem_key = problem_type.lower()
    if problem_key not in _problem_states:
        _problem_states[problem_key] = {
            "problem_instance": None,
            "last_design": None,
            "initial_design": None,
        }
    return _problem_states[problem_key]


def get_unified_problem_instance(problem_type: str) -> Beams2D | ThermoElastic2D:
    """Get or create a problem instance for the given problem type."""
    state = get_problem_state(problem_type)
    if state["problem_instance"] is None:
        problem_class = get_problem_class(problem_type)
        state["problem_instance"] = problem_class()
    return state["problem_instance"]  # type: ignore[return-value]


def set_unified_problem_instance(problem_type: str, problem: Any) -> None:
    """Set the problem instance for a given problem type."""
    state = get_problem_state(problem_type)
    state["problem_instance"] = problem


def get_unified_last_design(problem_type: str) -> np.ndarray | None:
    """Get the last design for a given problem type."""
    state = get_problem_state(problem_type)
    return state["last_design"]


def set_unified_last_design(problem_type: str, design: np.ndarray) -> None:
    """Store the last design for a given problem type."""
    state = get_problem_state(problem_type)
    state["last_design"] = design


def get_initial_design(problem_type: str) -> np.ndarray | None:
    """Get the initial design (before optimization) for a given problem type."""
    state = get_problem_state(problem_type)
    return state.get("initial_design")


def set_initial_design(problem_type: str, design: np.ndarray) -> None:
    """Store the initial design (before optimization) for a given problem type."""
    state = get_problem_state(problem_type)
    state["initial_design"] = design


# ============================================================================
# UNIFIED TOOLS - These work with any problem type
# ============================================================================


@tool
def create_problem(
    problem_type: str = "beams2d",
    seed: int = 0,
) -> dict[str, Any]:
    """
    Create an optimization problem using EngiBench.

    This unified tool works with any problem type available in EngiBench.
    Currently supported: 'beams2d', 'thermoelastic2d'.

    Args:
        problem_type: Type of problem ('beams2d', 'thermoelastic2d', etc.)
        seed: Random seed for reproducibility (default: 0)

    Returns:
        dict with problem info:
        - problem_id: str identifier
        - problem_type: str
        - design_space: description of design space
        - objectives: list of objectives to optimize
        - conditions: problem conditions/constraints
        - dataset_id: HuggingFace dataset identifier
        - success: bool
        - message: str

    Example:
        >>> info = create_problem(problem_type="beams2d", seed=42)
        >>> print(info['objectives'])
        >>> # [('compliance', 'MINIMIZE')]
    """
    try:
        problem_class = get_problem_class(problem_type)
        problem = problem_class()
        problem.reset(seed=seed)

        # Store the problem instance for reuse by other tools
        set_unified_problem_instance(problem_type, problem)

        return {
            "problem_id": problem_type.lower(),
            "problem_type": problem_type.lower(),
            "design_space": str(problem.design_space),
            "objectives": [
                (name, str(direction)) for name, direction in problem.objectives
            ],
            "conditions": problem.conditions,
            "conditions_keys": problem.conditions_keys,
            "dataset_id": problem.dataset_id,
            "success": True,
            "message": f"Created {problem_type} problem with seed {seed}",
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to create problem: {e!s}"}


@tool
def simulate_design(
    problem_type: str = "beams2d",
    design_description: str = "random design",
    config: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Simulate a design and return its performance metrics.

    This unified tool works with any problem type available in EngiBench.
    Currently supported: 'beams2d', 'thermoelastic2d'.

    Args:
        problem_type: Type of problem ('beams2d', 'thermoelastic2d', etc.)
        design_description: Description of the design approach (e.g., "random design",
            "optimized topology", "last design"). The tool will generate or retrieve
            an appropriate design based on this description.
        config: Problem-specific configuration parameters (optional)
            For beams2d: {"volume_fraction": 0.35, "force_distribution": 0.0}
            For thermoelastic2d: {"volfrac": 0.3, "weight": 0.5, "rmin": 1.1}
        seed: Random seed for reproducibility

    Returns:
        dict with simulation results (structure depends on problem type):
        - success: bool
        - For beams2d: compliance, volume_fraction_used, design_valid
        - For thermoelastic2d: structural_compliance, thermal_compliance, volume_fraction_used
        - message: str

    Example:
        >>> result = simulate_design(
        ...     problem_type="beams2d",
        ...     design_description="random design",
        ...     config={"volume_fraction": 0.4},
        ...     seed=42
        ... )
        >>> print(f"Compliance: {result['compliance']}")
    """
    try:
        if config is None:
            config = {}

        # Get problem instance
        problem = get_unified_problem_instance(problem_type)
        problem.reset(seed=seed)

        # Check if we have a stored design from previous operations
        last_design = get_unified_last_design(problem_type)

        # Determine which design to simulate
        if (
            "last" in design_description.lower()
            or "previous" in design_description.lower()
            or "current" in design_description.lower()
            or "optimized" in design_description.lower()
        ) and last_design is not None:
            design = last_design
        elif "random" in design_description.lower():
            design, _ = problem.random_design()
            set_unified_last_design(problem_type, design)
        else:
            design, _ = problem.random_design()
            set_unified_last_design(problem_type, design)

        # Run simulation
        objectives = problem.simulate(design=design, config=config if config else None)

        # Format results based on problem type
        problem_key = problem_type.lower()
        if problem_key == "beams2d":
            compliance = float(objectives[0])
            volume_fraction_actual = float(design.mean())
            return {
                "success": True,
                "problem_type": problem_key,
                "compliance": compliance,
                "volume_fraction_used": volume_fraction_actual,
                "design_valid": True,
                "message": f"Simulated {design_description} with compliance {compliance:.6f}",
            }
        elif problem_key == "thermoelastic2d":
            structural_compliance = float(objectives[0])
            thermal_compliance = float(objectives[1])
            volume_fraction_actual = float(objectives[2])
            return {
                "success": True,
                "problem_type": problem_key,
                "structural_compliance": structural_compliance,
                "thermal_compliance": thermal_compliance,
                "volume_fraction_used": volume_fraction_actual,
                "design_valid": True,
                "message": f"Simulated {design_description}: Structural={structural_compliance:.6f}, Thermal={thermal_compliance:.6f}",
            }
        else:
            return {
                "success": False,
                "error": f"Unknown problem type result format: {problem_type}",
            }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Simulation failed: {e!s}"}


@tool
def optimize_design(
    problem_type: str = "beams2d",
    starting_point: str = "random",
    config: dict[str, Any] | None = None,
    seed: int = 0,
    save_result: bool = True,
) -> dict[str, Any]:
    """
    Optimize a design using gradient-based optimization.

    This unified tool works with any problem type available in EngiBench.
    Currently supported: 'beams2d', 'thermoelastic2d'.

    Args:
        problem_type: Type of problem ('beams2d', 'thermoelastic2d', etc.)
        starting_point: Initial design approach ("random", "uniform", or "sparse")
        config: Problem-specific configuration parameters (optional)
            For beams2d: {"volfrac": 0.35, "forcedist": 0.0}
            For thermoelastic2d: {"volfrac": 0.3, "weight": 0.5, "rmin": 1.1}
        seed: Random seed for optimization (default: 0 for reproducibility)
        save_result: Whether to save the optimized design to outputs/ directory

    Returns:
        dict with optimization results (structure depends on problem type):
        - success: bool
        - For beams2d: initial_compliance, final_compliance, improvement
        - For thermoelastic2d: initial/final structural/thermal compliance, improvement
        - design_shape: tuple
        - save_path: str (if save_result=True)
        - message: str

    Example:
        >>> result = optimize_design(
        ...     problem_type="beams2d",
        ...     config={"volfrac": 0.4},
        ...     seed=42
        ... )
        >>> print(f"Improvement: {result['improvement']:.1f}%")
    """
    try:
        if config is None:
            config = {}

        # Get problem instance
        problem = get_unified_problem_instance(problem_type)
        problem.reset(seed=seed)

        # Get starting design
        if starting_point.lower() == "random":
            design, _ = problem.random_design()
        elif starting_point.lower() == "uniform":
            volfrac = config.get("volfrac", config.get("volume_fraction", 0.3))
            shape = problem.design_space.shape
            design = np.full(shape, volfrac, dtype=np.float32)
        elif starting_point.lower() == "sparse":
            volfrac = config.get("volfrac", config.get("volume_fraction", 0.3))
            shape = problem.design_space.shape
            design = np.full(shape, volfrac * 0.5, dtype=np.float32)
        else:
            design, _ = problem.random_design()

        # Store initial design separately for later visualization
        set_initial_design(problem_type, design)
        # Also set as last_design (will be overwritten after optimization)
        set_unified_last_design(problem_type, design)

        # Simulate initial design
        initial_objectives = problem.simulate(
            design=design, config=config if config else None
        )

        # Run optimization (design is a positional argument, not keyword)
        optimized_design, optimization_info = problem.optimize(
            design, config=config if config else None
        )

        # Store optimized design as the new last_design
        set_unified_last_design(problem_type, optimized_design)

        # Simulate final design
        final_objectives = problem.simulate(
            design=optimized_design, config=config if config else None
        )

        # Format results based on problem type
        result = _format_optimization_result(
            problem_type,
            initial_objectives,
            final_objectives,
            optimized_design,
            optimization_info,
        )

        # Save if requested
        if save_result:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / f"{problem_type}_design_optimized.npy"
            np.save(str(save_path), optimized_design)
            result["save_path"] = str(save_path)
            result["message"] += f" Saved to {save_path}"
            return result
        else:
            return result
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Optimization failed: {e!s}"}


def _extract_figure_and_axis(render_result):
    """Helper to extract figure and axis from different render return types."""
    if isinstance(render_result, tuple):
        # Beams2D returns a tuple
        return render_result
    # ThermoElastic2D returns a single figure
    fig = render_result
    return fig, fig.gca()


def _get_design_suffix(design_description: str, design_type: str) -> str:
    """Helper to get filename suffix based on design description."""
    desc_lower = design_description.lower()
    suffix_map = {
        "initial": "_initial",
        "before": "_initial",
        "final": "_final",
        "after": "_final",
        "optimized": "_optimized",
        "optimal": "_optimized",
        "random": "_random",
    }
    return next(
        (suf for keyword, suf in suffix_map.items() if keyword in desc_lower),
        "_" + design_type.replace(" ", "_").replace("(", "").replace(")", ""),
    )


def _build_versioned_path(base_path: Path, problem_type: str, suffix: str) -> Path:
    """Helper to build a versioned file path with problem type prefix."""
    stem = base_path.stem
    extension = base_path.suffix

    # Add problem type prefix if not already present
    if not stem.startswith(problem_type):
        stem = f"{problem_type}_{stem}"

    # Add timestamp for versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path.parent / f"{stem}{suffix}_{timestamp}{extension}"


def _should_use_last_design(design_description: str) -> bool:
    """Helper to determine if we should use the last design."""
    desc_lower = design_description.lower()
    keywords = ["last", "previous", "current", "optimized", "final"]
    return any(keyword in desc_lower for keyword in keywords)


def _get_design_to_render(
    problem_type: str, design_description: str, problem: Any
) -> tuple[np.ndarray, str]:
    """Get the design to render based on the description.

    Returns:
        Tuple of (design array, design type string)
    """
    desc_lower = design_description.lower()

    # Check if user wants the initial design (before optimization)
    if "initial" in desc_lower or "before" in desc_lower or "starting" in desc_lower:
        initial_design = get_initial_design(problem_type)
        if initial_design is not None:
            return initial_design, "initial design"
        # No initial design stored, create a random one
        design, _ = problem.random_design()
        return design, "random design"

    # Check if user wants optimized/final design
    if _should_use_last_design(design_description):
        last_design = get_unified_last_design(problem_type)
        if last_design is not None:
            return last_design, design_description
        # No last design stored, create a random one
        design, _ = problem.random_design()
        return design, "random design"

    # Default to random design
    design, _ = problem.random_design()
    return design, "random design"


def _format_optimization_result(
    problem_type: str,
    initial_objectives: Any,
    final_objectives: Any,
    optimized_design: np.ndarray,
    optimization_info: dict,
) -> dict[str, Any]:
    """Format optimization results based on problem type.

    Returns:
        Dictionary with formatted results
    """
    problem_key = problem_type.lower()
    result: dict[str, Any] = {
        "success": True,
        "problem_type": problem_key,
        "design_shape": optimized_design.shape,
        "optimization_info": optimization_info,
    }

    if problem_key == "beams2d":
        initial_compliance = float(initial_objectives[0])
        final_compliance = float(final_objectives[0])
        improvement = (
            (initial_compliance - final_compliance) / initial_compliance
        ) * 100

        result.update(
            {
                "initial_compliance": initial_compliance,
                "final_compliance": final_compliance,
                "improvement": improvement,
                "message": f"Optimized {problem_type} design: {initial_compliance:.6f} → {final_compliance:.6f} ({improvement:.1f}% improvement)",
            }
        )

    elif problem_key == "thermoelastic2d":
        initial_structural = float(initial_objectives[0])
        initial_thermal = float(initial_objectives[1])
        final_structural = float(final_objectives[0])
        final_thermal = float(final_objectives[1])
        structural_improvement = (
            (initial_structural - final_structural) / initial_structural
        ) * 100
        thermal_improvement = (
            (initial_thermal - final_thermal) / initial_thermal
        ) * 100

        result.update(
            {
                "initial_structural_compliance": initial_structural,
                "initial_thermal_compliance": initial_thermal,
                "final_structural_compliance": final_structural,
                "final_thermal_compliance": final_thermal,
                "structural_improvement": structural_improvement,
                "thermal_improvement": thermal_improvement,
                "message": f"Optimized {problem_type} design: Structural {initial_structural:.6f}→{final_structural:.6f} ({structural_improvement:.1f}%), Thermal {initial_thermal:.6f}→{final_thermal:.6f} ({thermal_improvement:.1f}%)",
            }
        )

    return result


@tool
def render_design(
    problem_type: str = "beams2d",
    design_description: str = "random design",
    config: dict[str, Any] | None = None,
    save_path: str = "design.png",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Render a design as a visual heatmap and save it as an image file.

    This unified tool works with any problem type available in EngiBench.
    Currently supported: 'beams2d', 'thermoelastic2d'.

    Args:
        problem_type: Type of problem ('beams2d', 'thermoelastic2d', etc.)
        design_description: Description of the design to render (e.g., "random design",
            "optimized topology", "initial design", "final design")
        config: Problem-specific configuration parameters (optional)
            For beams2d: {"volume_fraction": 0.35, "force_distribution": 0.0}
            For thermoelastic2d: {"volfrac": 0.3, "weight": 0.5, "rmin": 1.1}
        save_path: Base filename for the image (will be saved in outputs/ directory)
        seed: Random seed for reproducibility (None = use random seed)

    Returns:
        dict with rendering results:
        - success: bool
        - save_path: str (full path where PNG image was saved)
        - npy_path: str (full path where numpy array was saved)
        - design_shape: tuple
        - message: str

    Example:
        >>> result = render_design(
        ...     problem_type="beams2d",
        ...     design_description="optimized design"
        ... )
        >>> print(result['save_path'])
    """
    try:
        if config is None:
            config = {}

        # Create outputs directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Ensure save_path is in outputs directory
        save_path_obj = Path(save_path)
        if save_path_obj.parent.name != "outputs":
            full_save_path = output_dir / save_path_obj.name
        else:
            full_save_path = save_path_obj

        # Get problem instance
        problem: Beams2D | ThermoElastic2D = get_unified_problem_instance(problem_type)

        # Verify we have the correct problem type
        problem_class = get_problem_class(problem_type)
        if not isinstance(problem, problem_class):
            msg = (
                f"Error: Problem instance mismatch! "
                f"Expected {problem_class.__name__}, got {type(problem).__name__}. "
                f"Please create a {problem_type} problem first using create_problem()."
            )
            return {"success": False, "message": msg}

        # Use random seed if none provided
        if seed is None:
            seed = random.randint(0, 999999)
        elif seed != 0:
            problem.reset(seed=seed)  # type: ignore[attr-defined]

        # Get the design to render based on description
        design, design_type = _get_design_to_render(
            problem_type, design_description, problem
        )

        # Build versioned file path
        suffix = _get_design_suffix(design_description, design_type)
        full_save_path = _build_versioned_path(full_save_path, problem_type, suffix)

        # Render the design
        # Note: Different EngiBench problems return different types:
        # Beams2D returns a tuple, ThermoElastic2D returns a single figure
        render_result = problem.render(design, open_window=False)  # type: ignore[attr-defined]
        fig, ax = _extract_figure_and_axis(render_result)

        # Add title
        title = f"{problem_type} - {design_type.title()}"
        ax.set_title(title, fontsize=10)

        # Save figure
        fig.savefig(str(full_save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save design array
        npy_path = full_save_path.with_suffix(".npy")
        np.save(str(npy_path), design)

        return {
            "success": True,
            "problem_type": problem_type.lower(),
            "save_path": str(full_save_path),
            "npy_path": str(npy_path),
            "design_shape": design.shape,
            "design_type": design_type,
            "seed": seed,
            "message": f"Successfully rendered {design_type} and saved to {full_save_path}. Design array saved to {npy_path}.",
        }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Rendering failed: {e!s}"}


@tool
def get_problem_info(problem_type: str = "beams2d") -> dict[str, Any]:
    """
    Get information about available EngiBench problems.

    Args:
        problem_type: Type of problem ("beams2d", "thermoelastic2d", etc.)
            Default: "beams2d"

    Returns:
        dict with problem information:
        - available_problems: list of supported problem types
        - selected_problem: str
        - description: str (problem description)
        - objectives: list
        - typical_conditions: dict

    Example:
        >>> info = get_problem_info("beams2d")
        >>> print(info['description'])
    """
    problems_info = {
        "beams2d": {
            "description": "2D beam topology optimization problem. Minimize compliance "
            "(maximize stiffness) while satisfying volume constraints. Based on the classic "
            "88-line topology optimization code using SIMP (Solid Isotropic Material with "
            "Penalization) method.",
            "objectives": [("compliance", "MINIMIZE")],
            "typical_conditions": {
                "volfrac": "Volume fraction constraint (0-1, default: 0.35)",
                "rmin": "Filter radius for density filtering (>0, default: 2.0)",
                "forcedist": "Force distribution parameter (0-1, default: 0.0)",
                "overhang_constraint": "Enable overhang constraint (bool, default: False)",
            },
            "design_space": "2D grid (50x100) with material density [0,1] at each point",
        },
        "thermoelastic2d": {
            "description": "2D thermoelastic topology optimization problem. Multi-physics problem "
            "that couples structural (mechanical) and thermal domains. Minimize total compliance "
            "(structural + thermal) subject to volume constraints. Features one-way coupling from "
            "thermal to elastic domain through thermal expansion.",
            "objectives": [
                ("structural_compliance", "MINIMIZE"),
                ("thermal_compliance", "MINIMIZE"),
                ("volume_fraction", "MINIMIZE"),
            ],
            "typical_conditions": {
                "volfrac": "Volume fraction constraint (0-1, default: 0.3)",
                "rmin": "Filter radius for density filtering (>0, default: 1.1)",
                "weight": "Domain weighting: 1.0=pure structural, 0.0=pure thermal (0-1, default: 0.5)",
                "fixed_elements": "Binary NxN matrix of structurally fixed elements",
                "force_elements_x": "Binary NxN matrix of x-direction structural loads",
                "force_elements_y": "Binary NxN matrix of y-direction structural loads",
                "heatsink_elements": "Binary NxN matrix of heatsink elements",
            },
            "design_space": "2D grid (64x64) with material density [0,1] at each point",
        },
    }

    problem_key = problem_type.lower()

    if problem_key in problems_info:
        info = problems_info[problem_key]
        return {
            "success": True,
            "available_problems": list(problems_info.keys()),
            "selected_problem": problem_key,
            **info,
        }
    else:
        return {
            "success": True,
            "available_problems": list(problems_info.keys()),
            "selected_problem": None,
            "message": f"Problem type '{problem_type}' not found. Available: {list(problems_info.keys())}",
        }


@tool
def get_problem_details(problem_type: str = "beams2d") -> dict[str, Any]:
    """
    Get detailed information directly from the problem object's attributes.

    This tool creates a problem instance and extracts information from its
    design_space, objectives, and conditions attributes - the authoritative
    source of problem specifications.

    Args:
        problem_type: Type of problem ("beams2d", "thermoelastic2d", etc.)
            Default: "beams2d"

    Returns:
        dict with problem details:
        - success: bool
        - problem_type: str
        - design_space: str (from problem.design_space)
        - objectives: list of tuples (name, direction) from problem.objectives
        - conditions: dict (from problem.conditions)
        - conditions_keys: list (from problem.conditions_keys)
        - dataset_id: str (from problem.dataset_id)

    Example:
        >>> details = get_problem_details("beams2d")
        >>> print(details['design_space'])
        >>> # Box(0.0, 1.0, (50, 100), float32)
        >>> print(details['objectives'])
        >>> # [('compliance', 'MINIMIZE')]
    """
    try:
        problem_key = problem_type.lower()

        # Check if problem type is supported
        if problem_key not in PROBLEM_REGISTRY:
            return {
                "success": False,
                "error": f"Problem type '{problem_type}' not supported. Available: {list(PROBLEM_REGISTRY.keys())}",
            }

        # Create problem instance using the registry
        problem_class = PROBLEM_REGISTRY[problem_key]
        problem = problem_class()

        return {
            "success": True,
            "problem_type": problem_key,
            "design_space": str(problem.design_space),
            "design_space_shape": problem.design_space.shape,
            "design_space_bounds": {
                "low": float(problem.design_space.low.min()),
                "high": float(problem.design_space.high.max()),
            },
            "objectives": [
                (name, str(direction)) for name, direction in problem.objectives
            ],
            "conditions": problem.conditions,
            "conditions_keys": problem.conditions_keys,
            "dataset_id": problem.dataset_id,
            "description": f"Information from the EngiBench {problem_key} problem object. "
            f"The design_space defines where designs can exist (shape {problem.design_space.shape}). "
            f"The objectives specify what to optimize. "
            f"The conditions are the parameters that can be varied for different problem instances.",
        }
    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get problem details: {e!s}"}


@tool
def get_dataset_info(problem_type: str = "beams2d") -> dict[str, Any]:
    """
    Get information about the EngiBench dataset for a problem.

    The dataset contains pre-computed optimal designs from the EngiBench benchmark.
    Each dataset includes training, validation, and test splits with optimal designs
    and their corresponding parameters.

    Args:
        problem_type: Type of problem ("beams2d", "thermoelastic2d", etc.)
            Default: "beams2d"

    Returns:
        dict with dataset information:
        - success: bool (whether the operation succeeded)
        - dataset_id: str (HuggingFace dataset identifier)
        - splits: dict with split names and their sizes
        - features: list of feature names in the dataset
        - total_samples: int (total number of samples across all splits)
        - description: str (description of dataset contents)

    Example:
        >>> info = get_dataset_info("beams2d")
        >>> print(f"Dataset has {info['total_samples']} total samples")
        >>> print(f"Features: {info['features']}")
    """
    try:
        problem_key = problem_type.lower()

        # Check if problem type is supported
        if problem_key not in PROBLEM_REGISTRY:
            return {
                "success": False,
                "error": f"Problem type '{problem_type}' not supported. Available: {list(PROBLEM_REGISTRY.keys())}",
            }

        # Create problem instance using the registry
        problem_class = PROBLEM_REGISTRY[problem_key]
        problem = problem_class()
        dataset = problem.dataset

        # Get information about each split
        splits = {}
        total_samples = 0
        features: list[str] = []

        for split_name in dataset:
            split = dataset[split_name]
            splits[split_name] = {
                "num_rows": len(split),
                "num_columns": len(split.column_names),
            }
            total_samples += len(split)
            if not features and split.column_names:
                features = split.column_names

        # Problem-specific descriptions
        descriptions = {
            "beams2d": "HuggingFace dataset containing pre-computed optimal beam designs. "
            "Each sample includes the optimal design array, optimization parameters "
            "(volfrac, rmin, forcedist, overhang_constraint), compliance value (c), "
            "and optimization history.",
            "thermoelastic2d": "HuggingFace dataset containing pre-computed optimal thermoelastic designs. "
            "Each sample includes the optimal design array, optimization parameters "
            "(volfrac, temp_load, mech_load), compliance and thermal expansion values, "
            "and optimization history.",
        }

        return {
            "success": True,
            "problem_type": problem_key,
            "dataset_id": problem.dataset_id,
            "splits": splits,
            "split_names": list(dataset.keys()),
            "features": features,
            "total_samples": total_samples,
            "description": descriptions.get(
                problem_key, f"HuggingFace dataset for {problem_key} problem."
            ),
        }
    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get dataset info: {e!s}"}
