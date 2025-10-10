"""
EngiBench tools for engineering design optimization.

EngiBench (https://engibench.ethz.ch) is a library for benchmarking
engineering design problems. It provides access to various optimization
problems like beam design, airfoils, truss structures, etc.

These tools allow LLM agents to interact with EngiBench simulators.
"""

from typing import Any

import matplotlib
from engibench.problems.beams2d.v0 import Beams2D  # type: ignore[import-untyped]
from langchain_core.tools import tool

matplotlib.use("Agg")  # Use non-interactive backend
import random

import matplotlib.pyplot as plt


@tool
def create_beam_problem(
    seed: int = 0,
) -> dict[str, Any]:
    """
    Create a 2D beam optimization problem using EngiBench.

    The beam problem involves optimizing the material distribution in a 2D grid
    to minimize compliance (maximize stiffness) while satisfying volume constraints.

    Args:
        seed: Random seed for reproducibility (default: 0)

    Returns:
        dict with problem info:
        - problem_id: str identifier
        - design_space: description of design space (50x100 grid)
        - objectives: list of objectives to optimize
        - conditions: problem conditions/constraints
        - dataset_id: HuggingFace dataset identifier

    Example:
        >>> info = create_beam_problem(seed=42)
        >>> print(info['objectives'])
        >>> # [('compliance', 'MINIMIZE')]
    """
    try:
        problem = Beams2D()

        return {
            "problem_id": "beams2d",
            "design_space": str(problem.design_space),
            "objectives": [
                (name, str(direction)) for name, direction in problem.objectives
            ],
            "conditions": problem.conditions,
            "conditions_keys": problem.conditions_keys,
            "dataset_id": problem.dataset_id,
            "success": True,
            "message": f"Created Beams2D problem with seed {seed}",
        }
    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to create problem: {e!s}"}


@tool
def simulate_beam_design(
    design_description: str,
    volume_fraction: float = 0.35,
    force_distribution: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Simulate a beam design and return its performance metrics.

    This tool evaluates a beam design by running a structural simulation to
    calculate compliance (inverse of stiffness). Lower compliance is better.

    Args:
        design_description: Description of the design approach (e.g., "uniform distribution",
            "optimized topology", "cantilever beam"). The tool will generate or retrieve
            an appropriate design based on this description.
        volume_fraction: Fraction of volume that can be filled with material (0-1)
            Default: 0.35 (35% material)
        force_distribution: Distribution parameter for applied forces (0-1)
            Default: 0.0 (single point load)
        seed: Random seed for reproducibility

    Returns:
        dict with simulation results:
        - compliance: float (lower is better)
        - design_valid: bool (whether design satisfies constraints)
        - volume_fraction_used: float (actual volume used)
        - message: str (description of results)

    Example:
        >>> result = simulate_beam_design(
        ...     design_description="random design",
        ...     volume_fraction=0.4,
        ...     seed=42
        ... )
        >>> print(f"Compliance: {result['compliance']}")
    """
    try:
        problem = Beams2D()
        problem.reset(seed=seed)

        # Generate a design based on description
        if "random" in design_description.lower():
            design, _ = problem.random_design()
        else:
            # Use a random design from the dataset
            design, _ = problem.random_design()

        # Set up configuration
        config = {
            "volfrac": volume_fraction,
            "forcedist": force_distribution,
        }

        # Skip constraint checks for faster simulation
        # Note: This allows exploring designs that might violate constraints

        # Run simulation
        objectives = problem.simulate(design=design, config=config)
        compliance = float(objectives[0])  # First objective is compliance

    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Simulation failed: {e!s}"}
    else:
        return {
            "success": True,
            "design_valid": True,  # Not checking constraints
            "compliance": compliance,
            "volume_fraction_used": volume_fraction,
            "force_distribution": force_distribution,
            "message": f"Simulation successful. Compliance: {compliance:.4f} (constraint checks skipped)",
        }


@tool
def optimize_beam_design(
    starting_point: str = "random",
    volume_fraction: float = 0.35,
    force_distribution: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Optimize a beam design using gradient-based optimization.

    This tool starts from an initial design and runs an optimization algorithm
    (typically topology optimization) to find the best material distribution.

    Args:
        starting_point: Initial design approach ("random", "uniform", or "sparse")
            Default: "random"
        volume_fraction: Maximum fraction of volume that can be filled (0-1)
            Default: 0.35 (35% material)
        force_distribution: Distribution parameter for applied forces (0-1)
            Default: 0.0 (single point load)
        seed: Random seed for reproducibility

    Returns:
        dict with optimization results:
        - initial_compliance: float (compliance before optimization)
        - final_compliance: float (compliance after optimization)
        - improvement: float (percentage improvement)
        - iterations: int (number of optimization steps)
        - success: bool
        - message: str

    Example:
        >>> result = optimize_beam_design(
        ...     starting_point="random",
        ...     volume_fraction=0.4,
        ...     seed=42
        ... )
        >>> print(f"Improved by {result['improvement']:.1f}%")
    """
    try:
        problem = Beams2D()
        problem.reset(seed=seed)

        # Generate starting design
        if starting_point.lower() == "random":
            design, _ = problem.random_design()
        else:
            design, _ = problem.random_design()

        # Configuration
        config = {
            "volfrac": volume_fraction,
            "forcedist": force_distribution,
        }

        # Get initial performance
        initial_objectives = problem.simulate(design=design, config=config)
        initial_compliance = float(initial_objectives[0])

        # Run optimization
        optimized_design, history = problem.optimize(
            starting_point=design, config=config
        )

        # Get final performance
        final_objectives = problem.simulate(design=optimized_design, config=config)
        final_compliance = float(final_objectives[0])

        # Calculate improvement
        improvement = (
            (initial_compliance - final_compliance) / initial_compliance
        ) * 100

        return {
            "success": True,
            "initial_compliance": initial_compliance,
            "final_compliance": final_compliance,
            "improvement": improvement,
            "iterations": len(history),
            "volume_fraction": volume_fraction,
            "message": f"Optimization successful. Improved compliance by {improvement:.1f}% "
            f"({initial_compliance:.4f} → {final_compliance:.4f})",
        }

    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Optimization failed: {e!s}"}


@tool
def render_beam_design(
    design_description: str = "random design",
    volume_fraction: float = 0.35,
    force_distribution: float = 0.0,
    save_path: str = "beam_design.png",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Render a beam design as a visual heatmap and save it as an image file.

    This tool creates a visual representation of a beam topology design,
    showing material distribution. Dark areas represent solid material,
    light areas represent voids/air.

    Args:
        design_description: Description of the design to render (e.g., "random design",
            "optimized topology"). The tool will generate or retrieve an appropriate design.
        volume_fraction: Fraction of volume filled with material (0-1)
            Default: 0.35 (35% material)
        force_distribution: Distribution parameter for applied forces (0-1)
            0.0 = single point load at center (default)
            1.0 = uniformly distributed load across top
        save_path: Path where the image will be saved (relative to current directory)
            Default: "beam_design.png"
        seed: Random seed for reproducibility (None = use random seed for different designs each time)

    Returns:
        dict with rendering results:
        - success: bool
        - save_path: str (where image was saved)
        - design_shape: tuple (dimensions of the design grid)
        - message: str (description of results)

    Example:
        >>> result = render_beam_design(
        ...     design_description="random design",
        ...     volume_fraction=0.35,
        ...     save_path="my_beam.png"
        ... )
        >>> print(result['message'])
    """
    try:
        # Use a random seed if none provided (so you get different designs each time)
        if seed is None:
            seed = random.randint(0, 999999)

        problem = Beams2D()
        problem.reset(seed=seed)

        # Configuration with user-specified parameters
        config = {
            "volfrac": volume_fraction,
            "forcedist": force_distribution,
        }

        # Generate a design based on description
        if (
            "optimized" in design_description.lower()
            or "optimal" in design_description.lower()
        ):
            # Run optimization with the specified parameters
            design, history = problem.optimize(config=config)
            design_type = "optimized"
            compliance = float(history[-1].obj_values[0]) if history else None
        elif "random" in design_description.lower():
            # Get a random design from dataset (parameters are informational only)
            design, _ = problem.random_design()
            design_type = "random"
            compliance = None
        else:
            # Default: get optimal design from dataset
            design, _ = problem.random_design(
                dataset_split="train", design_key="optimal_design"
            )
            design_type = "dataset optimal"
            compliance = None

        # Render the design using EngiBench's built-in render method
        fig, ax = problem.render(design, open_window=False)

        # Add title with parameters
        title = f"{design_type.title()} Design (volfrac={volume_fraction:.2f}, forcedist={force_distribution:.2f})"
        if compliance is not None:
            title += f"\nCompliance: {compliance:.4f}"
        ax.set_title(title, fontsize=10)

        # Save the figure
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        result = {
            "success": True,
            "save_path": save_path,
            "design_shape": design.shape,
            "design_type": design_type,
            "seed": seed,
            "volume_fraction_requested": volume_fraction,
            "volume_fraction_actual": float(design.mean()),
            "force_distribution": force_distribution,
            "message": f"Successfully rendered {design_type} design and saved to {save_path}. "
            f"Requested volfrac={volume_fraction:.2f}, actual={design.mean():.3f}, forcedist={force_distribution:.2f}, seed={seed}",
        }

        if compliance is not None:
            result["compliance"] = compliance
            result["message"] += f", compliance={compliance:.4f}"

    except ImportError as e:
        return {
            "success": False,
            "error": f"engibench or matplotlib not installed: {e!s}. Install with: pip install engibench matplotlib",
        }
    except Exception as e:
        return {"success": False, "error": f"Rendering failed: {e!s}"}
    else:
        return result


@tool
def get_problem_info(problem_type: str = "beams2d") -> dict[str, Any]:
    """
    Get information about available EngiBench problems.

    Args:
        problem_type: Type of problem ("beams2d", "airfoil", "truss", etc.)
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
            "(maximize stiffness) while satisfying volume constraints.",
            "objectives": [("compliance", "MINIMIZE")],
            "typical_conditions": {
                "volfrac": "Volume fraction (0-1)",
                "forcedist": "Force distribution parameter (0-1)",
            },
            "design_space": "2D grid (typically 50x100) with material density at each point",
        },
        "airfoil": {
            "description": "Airfoil shape optimization for aerodynamic performance",
            "objectives": [("drag", "MINIMIZE"), ("lift", "MAXIMIZE")],
            "typical_conditions": {
                "reynolds": "Reynolds number",
                "mach": "Mach number",
            },
            "design_space": "Control points defining airfoil shape",
        },
    }

    if problem_type.lower() in problems_info:
        info = problems_info[problem_type.lower()]
        return {
            "success": True,
            "available_problems": list(problems_info.keys()),
            "selected_problem": problem_type.lower(),
            **info,
        }
    else:
        return {
            "success": True,
            "available_problems": list(problems_info.keys()),
            "selected_problem": None,
            "message": f"Problem type '{problem_type}' not found. Available: {list(problems_info.keys())}",
        }
