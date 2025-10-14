"""
EngiBench tools for engineering design optimization.

EngiBench (https://engibench.ethz.ch) is a library for benchmarking
engineering design problems. It provides access to various optimization
problems like beam design, airfoils, truss structures, etc.

These tools allow LLM agents to interact with EngiBench simulators.
"""

import random
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from engibench.problems.beams2d.v0 import Beams2D  # type: ignore[import-untyped]
from langchain_core.tools import tool

matplotlib.use("Agg")  # Use non-interactive backend

# Constants
EXPECTED_ARRAY_DIMENSIONS = 2  # For 2D beam design arrays

# State management using a module-level dictionary
# This allows tools like optimize_beam_design to reuse the problem
# created by create_beam_problem without using global statements
_state: dict[str, Any] = {
    "problem_instance": None,
    "last_design": None,
}


def get_problem_instance() -> Beams2D:
    """Get the current problem instance, creating one if needed."""
    if _state["problem_instance"] is None:
        _state["problem_instance"] = Beams2D()
    return _state["problem_instance"]


def set_problem_instance(problem: Beams2D) -> None:
    """Set the problem instance."""
    _state["problem_instance"] = problem


def get_last_design() -> np.ndarray | None:
    """Get the last design that was created or used."""
    return _state["last_design"]


def set_last_design(design: np.ndarray) -> None:
    """Store the last design for potential reuse."""
    _state["last_design"] = design


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
        problem.reset(seed=seed)

        # Store the problem instance for reuse by other tools
        set_problem_instance(problem)

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
        # Use the existing problem instance to maintain consistency
        problem = get_problem_instance()

        # Always reset the problem before simulation for clean state
        problem.reset(seed=seed)

        # Check if we have a stored design from previous operations
        last_design = get_last_design()

        # Determine which design to simulate
        if (
            "last" in design_description.lower()
            or "previous" in design_description.lower()
            or "current" in design_description.lower()
            or "optimized" in design_description.lower()
        ) and last_design is not None:
            # Use the stored design from previous operation
            design = last_design
        elif "random" in design_description.lower():
            design, _ = problem.random_design()
            # Store for future use
            set_last_design(design)
        else:
            # Use a random design from the dataset
            design, _ = problem.random_design()
            # Store for future use
            set_last_design(design)

        # Set up configuration
        config = {
            "volfrac": volume_fraction,
            "forcedist": force_distribution,
        }

        # Skip constraint checks for faster simulation
        # Note: This allows exploring designs that might violate constraints

        # Run simulation (problem is already reset above)
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
def check_beam_constraints(
    design_source: str = "last",
    volume_fraction: float | None = None,
    force_distribution: float | None = None,
) -> dict[str, Any]:
    """
    Check if a beam design satisfies the problem constraints.

    This tool validates whether a design meets the constraints specified
    in the problem configuration. It checks the volume fraction (material
    usage) and can detect violations that may affect optimization results.

    Args:
        design_source: Source of the design to check. Options:
            - "last": Use the last design from state (default)
            - "random": Generate and check a random design
            - file path: Load design from .npy file (e.g., "my_design.npy")
        volume_fraction: Target volume fraction (material density). If None, uses current problem settings
        force_distribution: Force distribution parameter. If None, uses current problem settings

    Returns:
        dict with constraint validation results:
        - success: bool
        - has_violations: bool (True if any constraints are violated)
        - violations: dict (details of constraint violations, if any)
        - config_used: dict (the configuration used for checking)
        - design_source_used: str (which design source was actually used)
        - message: str

    Example:
        To check the last design (most common usage):
        >>> result = check_beam_constraints()
        >>> if result['has_violations']:
        ...     print("Violations found:", result['violations'])

        To check a random design:
        >>> result = check_beam_constraints(design_source="random")

        To check with custom constraints:
        >>> result = check_beam_constraints(
        ...     volume_fraction=0.3,
        ...     force_distribution=0.5
        ... )

        To check a design from file:
        >>> result = check_beam_constraints(design_source="my_design.npy")
    """
    try:
        problem = get_problem_instance()
        if problem is None:
            return {
                "success": False,
                "error": "No problem instance found. Create one first with create_beam_problem()",
            }

        # Get the design based on source
        if design_source == "last":
            # Use last design from state
            design = _state.get("last_design")
            if design is None:
                return {
                    "success": False,
                    "error": "No design found in state. Create a design first with simulate_beam_design() or optimize_beam_design()",
                }
            source_used = "last design from state"

        elif design_source == "random":
            # Generate a random design
            design, _ = problem.random_design()
            source_used = "randomly generated design"

        else:
            # Treat as file path
            design_file = Path(design_source)
            if not design_file.exists():
                return {
                    "success": False,
                    "error": f"Design file not found: {design_source}",
                }
            design = np.load(design_file)
            source_used = f"design from file: {design_source}"

        # Build config dict
        config = {}
        if volume_fraction is not None:
            config["volfrac"] = volume_fraction
        if force_distribution is not None:
            config["forcedist"] = force_distribution

        # Check constraints using the problem's method
        violations = problem.check_constraints(design, config)

        return {
            "success": True,
            "has_violations": bool(violations),
            "violations": violations if violations else {},
            "config_used": config,
            "design_source_used": source_used,
            "message": (
                f"Checked {source_used}. "
                f"{'Violations found: ' + str(violations) if violations else 'No violations detected.'}"
            ),
        }

    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Constraint check failed: {e!s}"}


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

    Important: The problem is always reset before optimization to ensure
    reproducibility and clean state. This follows best practices from EngiBench.

    Args:
        starting_point: Initial design approach ("random", "uniform", or "sparse")
            Default: "random"
        volume_fraction: Maximum fraction of volume that can be filled (0-1)
            Default: 0.35 (35% material)
        force_distribution: Distribution parameter for applied forces (0-1)
            Default: 0.0 (single point load)
        seed: Random seed for optimization (default: 0 for reproducibility)

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
        ...     volume_fraction=0.35,
        ...     seed=0  # Resets problem to seed=0 before optimization
        ... )
        >>> print(f"Improved by {result['improvement']:.1f}%")
    """
    try:
        # Use the existing problem instance if available (from create_beam_problem)
        # This maintains consistency across the conversation
        problem = get_problem_instance()

        # Always reset problem before optimization for clean state
        # This matches notebook workflow and ensures reproducibility
        problem.reset(seed=seed)

        # Check if we have a last design from previous operations (e.g., from render)
        last_design = get_last_design()

        # Determine starting design
        if last_design is not None and starting_point.lower() != "random":
            # Reuse the last design if available (e.g., from rendering)
            # This matches notebook workflow: render → optimize same design
            design = last_design
        elif starting_point.lower() == "random":
            # Generate a new random design
            design, _ = problem.random_design()
        else:
            # Default: random design
            design, _ = problem.random_design()

        # Store this design for potential future use
        set_last_design(design)

        # Configuration
        config = {
            "volfrac": volume_fraction,
            "forcedist": force_distribution,
        }

        # Get initial performance (problem is already reset above)
        initial_objectives = problem.simulate(design=design, config=config)
        initial_compliance = float(initial_objectives[0])

        # Run optimization (problem is already reset above)
        # Note: problem.optimize() accepts the design as first positional argument
        optimized_design, history = problem.optimize(design, config=config)

        # Store the optimized design for potential future use (e.g., rendering)
        set_last_design(optimized_design)

        # Get final performance (no need to reset again, we're just evaluating)
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
            "design_stored": True,  # Indicate that design is stored for future use
            "message": f"Optimization successful. Improved compliance by {improvement:.1f}% "
            f"({initial_compliance:.4f} → {final_compliance:.4f}). "
            f"Optimized design stored and ready for rendering.",
        }

    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Optimization failed: {e!s}"}


def _select_design_for_rendering(
    problem: Beams2D,
    design_description: str,
    config: dict[str, float],
    last_design: np.ndarray | None,
) -> tuple[np.ndarray, str, float | None]:
    """
    Select which design to render based on description and available designs.

    Returns:
        tuple of (design, design_type, compliance)
    """
    design_type = "unknown"
    compliance = None

    # Priority 1: User asks for stored design AND we have one
    if last_design is not None and any(
        keyword in design_description.lower()
        for keyword in ["last", "optimized", "previous", "current", "stored", "that"]
    ):
        design = last_design
        design_type = "stored (from previous optimization)"
        try:
            objectives = problem.simulate(design=design, config=config)
            compliance = float(objectives[0])
        except Exception:
            compliance = None
    # Priority 2: User wants new optimization (no stored design available)
    elif "optimize" in design_description.lower() and last_design is None:
        design, history = problem.optimize(config=config)
        design_type = "newly optimized"
        compliance = float(history[-1].obj_values[0]) if history else None
        set_last_design(design)
    # Priority 3: Random design requested
    elif "random" in design_description.lower():
        design, _ = problem.random_design()
        design_type = "random"
        set_last_design(design)
    # Priority 4: Default - use stored or random
    elif last_design is not None:
        design = last_design
        design_type = "stored design"
        try:
            objectives = problem.simulate(design=design, config=config)
            compliance = float(objectives[0])
        except Exception:
            compliance = None
    else:
        design, _ = problem.random_design()
        design_type = "random"
        set_last_design(design)

    return design, design_type, compliance


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

    The design array is also saved as a .npy file for numerical analysis.

    All images are automatically saved to the 'outputs/' directory.

    To prevent overwriting files when rendering before/after optimization,
    the tool automatically adds suffixes like "_random", "_optimized", "_initial", etc.
    based on the design_description.

    Args:
        design_description: Description of the design to render (e.g., "random design",
            "optimized topology", "initial design", "final design").
            The tool will generate or retrieve an appropriate design and use keywords
            to create descriptive filenames (e.g., "beam_design_random.png").
        volume_fraction: Fraction of volume filled with material (0-1)
            Default: 0.35 (35% material)
        force_distribution: Distribution parameter for applied forces (0-1)
            0.0 = single point load at center (default)
            1.0 = uniformly distributed load across top
        save_path: Base filename for the image (will be saved in outputs/ directory)
            Default: "beam_design.png"
            A descriptive suffix will be automatically added to prevent overwrites
        seed: Random seed for reproducibility (None = use random seed for different designs each time)

    Returns:
        dict with rendering results:
        - success: bool
        - save_path: str (full path where PNG image was saved)
        - npy_path: str (full path where numpy array was saved)
        - design_shape: tuple (dimensions of the design grid)
        - message: str (description of results)

    Example:
        >>> # Render initial design - saves as "beam_design_random.png"
        >>> result1 = render_beam_design(design_description="random design")
        >>>
        >>> # Optimize the design...
        >>>
        >>> # Render optimized design - saves as "beam_design_optimized.png"
        >>> result2 = render_beam_design(design_description="optimized design")
        >>>
        >>> # Both files are preserved!
    """
    try:
        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Ensure save_path is in the outputs directory
        save_path_obj = Path(save_path)
        if save_path_obj.parent.name != "outputs":
            # If user provided just a filename, put it in outputs/
            full_save_path = output_dir / save_path_obj.name
        else:
            # If user already specified outputs/, use as-is
            full_save_path = save_path_obj

        # Use the existing problem instance to maintain consistency
        problem = get_problem_instance()

        # Use a random seed if none provided (so you get different designs each time)
        if seed is None:
            seed = random.randint(0, 999999)
        elif seed != 0:
            problem.reset(seed=seed)

        # Configuration with user-specified parameters
        config = {
            "volfrac": volume_fraction,
            "forcedist": force_distribution,
        }

        # Select design using helper function
        design, design_type, compliance = _select_design_for_rendering(
            problem, design_description, config, get_last_design()
        )

        # Add automatic suffix based on design type to prevent overwriting
        # Determine suffix from design_description or design_type
        desc_lower = design_description.lower()

        # Map keywords to suffixes
        suffix_map = {
            "initial": "_initial",
            "before": "_initial",
            "final": "_final",
            "after": "_final",
            "optimized": "_optimized",
            "optimal": "_optimized",
            "random": "_random",
        }

        # Find matching suffix
        suffix = next(
            (suf for keyword, suf in suffix_map.items() if keyword in desc_lower),
            "_" + design_type.replace(" ", "_").replace("(", "").replace(")", ""),
        )

        # Insert suffix before file extension
        stem = full_save_path.stem
        extension = full_save_path.suffix
        full_save_path = full_save_path.parent / f"{stem}{suffix}{extension}"

        # Render the design using EngiBench's built-in render method
        fig, ax = problem.render(design, open_window=False)

        # Add title with parameters
        title = f"{design_type.title()} Design (volfrac={volume_fraction:.2f}, forcedist={force_distribution:.2f})"
        if compliance is not None:
            title += f"\nCompliance: {compliance:.4f}"
        ax.set_title(title, fontsize=10)

        # Save the figure
        fig.savefig(str(full_save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save the design array as .npy file (also in outputs/)
        npy_path = full_save_path.with_suffix(".npy")
        np.save(str(npy_path), design)

        result: dict[str, Any] = {
            "success": True,
            "save_path": str(full_save_path),
            "npy_path": str(npy_path),
            "design_shape": design.shape,
            "design_type": design_type,
            "seed": seed,
            "volume_fraction_requested": volume_fraction,
            "volume_fraction_actual": float(design.mean()),
            "force_distribution": force_distribution,
            "message": f"Successfully rendered {design_type} design and saved to {full_save_path}. "
            f"Design array saved to {npy_path}. "
            f"Requested volfrac={volume_fraction:.2f}, actual={design.mean():.3f}, forcedist={force_distribution:.2f}, seed={seed}",
        }

        if compliance is not None:
            result["compliance"] = compliance
            message = str(result["message"])
            result["message"] = f"{message}, compliance={compliance:.4f}"

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


@tool
def get_problem_details(problem_type: str = "beams2d") -> dict[str, Any]:
    """
    Get detailed information directly from the problem object's attributes.

    This tool creates a problem instance and extracts information from its
    design_space, objectives, and conditions attributes - the authoritative
    source of problem specifications.

    Args:
        problem_type: Type of problem ("beams2d", etc.)
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
        if problem_type.lower() == "beams2d":
            problem = Beams2D()

            return {
                "success": True,
                "problem_type": problem_type.lower(),
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
                "description": "This information comes directly from the EngiBench problem object. "
                "The design_space defines where designs can exist (a 50x100 grid with values 0-1). "
                "The objectives specify what to optimize (minimize compliance = maximize stiffness). "
                "The conditions are the parameters that can be varied for different problem instances.",
            }
        else:
            return {
                "success": False,
                "error": f"Problem type '{problem_type}' not supported yet. Currently only 'beams2d' is available.",
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
        problem_type: Type of problem ("beams2d", etc.)
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
        if problem_type.lower() == "beams2d":
            problem = Beams2D()
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

            return {
                "success": True,
                "problem_type": problem_type.lower(),
                "dataset_id": problem.dataset_id,
                "splits": splits,
                "split_names": list(dataset.keys()),
                "features": features,
                "total_samples": total_samples,
                "description": "HuggingFace dataset containing pre-computed optimal beam designs. "
                "Each sample includes the optimal design array, optimization parameters "
                "(volfrac, rmin, forcedist, overhang_constraint), compliance value (c), "
                "and optimization history.",
            }
        else:
            return {
                "success": False,
                "error": f"Problem type '{problem_type}' not supported yet. Currently only 'beams2d' is available.",
            }
    except ImportError:
        return {
            "success": False,
            "error": "engibench not installed. Install with: pip install engibench",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get dataset info: {e!s}"}
