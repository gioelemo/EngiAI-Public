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
from stl import mesh

matplotlib.use("Agg")  # Use non-interactive backend

# Constants
EXPECTED_ARRAY_DIMENSIONS = 2  # For 2D beam design arrays


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

    The design array is also saved as a .npy file for numerical analysis.

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
        - save_path: str (where PNG image was saved)
        - npy_path: str (where numpy array was saved)
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

        # Save the design array as .npy file
        npy_path = save_path.rsplit(".", 1)[0] + ".npy"
        np.save(npy_path, design)

        result = {
            "success": True,
            "save_path": save_path,
            "npy_path": npy_path,
            "design_shape": design.shape,
            "design_type": design_type,
            "seed": seed,
            "volume_fraction_requested": volume_fraction,
            "volume_fraction_actual": float(design.mean()),
            "force_distribution": force_distribution,
            "message": f"Successfully rendered {design_type} design and saved to {save_path}. "
            f"Design array saved to {npy_path}. "
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


def _create_surface_vertices(
    data: np.ndarray, scale_xy: float, scale_z: float, base_thickness: float
) -> list[list[float]]:
    """Create vertices for top and bottom surfaces of the mesh."""
    height, width = data.shape
    vertices = []

    # Top surface vertices (flipped Y coordinate)
    for i in range(height):
        for j in range(width):
            x = j * scale_xy
            y = (height - 1 - i) * scale_xy
            z = data[i, j] * scale_z + base_thickness
            vertices.append([x, y, z])

    # Bottom surface vertices
    for i in range(height):
        for j in range(width):
            x = j * scale_xy
            y = (height - 1 - i) * scale_xy
            z = 0
            vertices.append([x, y, z])

    return vertices


def _create_top_bottom_faces(height: int, width: int) -> list[list[int]]:
    """Create faces for top and bottom surfaces."""
    faces = []
    offset = height * width

    # Top surface faces
    for i in range(height - 1):
        for j in range(width - 1):
            v1 = i * width + j
            v2 = i * width + (j + 1)
            v3 = (i + 1) * width + j
            v4 = (i + 1) * width + (j + 1)
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    # Bottom surface faces
    for i in range(height - 1):
        for j in range(width - 1):
            v1 = offset + i * width + j
            v2 = offset + i * width + (j + 1)
            v3 = offset + (i + 1) * width + j
            v4 = offset + (i + 1) * width + (j + 1)
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])

    return faces


def _create_side_wall_faces(height: int, width: int) -> list[list[int]]:
    """Create faces for side walls (left, right, front, back edges)."""
    faces = []
    offset = height * width

    # Left edge
    for i in range(height - 1):
        v1 = i * width
        v2 = (i + 1) * width
        v3 = offset + i * width
        v4 = offset + (i + 1) * width
        faces.append([v1, v3, v2])
        faces.append([v2, v3, v4])

    # Right edge
    for i in range(height - 1):
        v1 = i * width + (width - 1)
        v2 = (i + 1) * width + (width - 1)
        v3 = offset + i * width + (width - 1)
        v4 = offset + (i + 1) * width + (width - 1)
        faces.append([v1, v2, v3])
        faces.append([v2, v4, v3])

    # Front edge
    for j in range(width - 1):
        v1 = j
        v2 = j + 1
        v3 = offset + j
        v4 = offset + j + 1
        faces.append([v1, v2, v3])
        faces.append([v2, v4, v3])

    # Back edge
    for j in range(width - 1):
        v1 = (height - 1) * width + j
        v2 = (height - 1) * width + j + 1
        v3 = offset + (height - 1) * width + j
        v4 = offset + (height - 1) * width + j + 1
        faces.append([v1, v3, v2])
        faces.append([v2, v3, v4])

    return faces


@tool
def convert_design_to_stl(
    npy_file_path: str,
    stl_file_path: str | None = None,
    scale_z: float = 10.0,
    scale_xy: float = 1.0,
    base_thickness: float = 1.0,
) -> dict[str, Any]:
    """
    Convert a 2D beam design array (.npy file) to a 3D STL file for 3D printing or CAD.

    This tool takes a numpy array file containing a 2D beam topology design
    and converts it into a 3D mesh (STL format). The height of each point
    corresponds to the material density at that location.

    Args:
        npy_file_path: Path to the .npy file containing the 2D design array
        stl_file_path: Output path for the STL file (default: same name as npy with .stl extension)
        scale_z: Height scale factor - how much to extrude the design vertically (default: 10)
        scale_xy: Horizontal scale factor for X and Y dimensions (default: 1)
        base_thickness: Thickness of the base layer in the same units as scale_z (default: 1)

    Returns:
        dict with conversion results:
        - success: bool
        - stl_path: str (where STL file was saved)
        - input_shape: tuple (dimensions of input array)
        - num_triangles: int (number of triangles in the mesh)
        - message: str (description of results)

    Example:
        To convert beam_design.npy to beam_design.stl with default settings:
        {"npy_file_path": "beam_design.npy"}

        To customize the 3D extrusion:
        {"npy_file_path": "beam_design.npy", "scale_z": 20, "scale_xy": 2, "base_thickness": 2}
    """
    try:
        # Validate input file exists
        input_path = Path(npy_file_path)
        if not input_path.exists():
            return {
                "success": False,
                "message": f"Input file '{npy_file_path}' not found",
            }

        # Determine output path
        if stl_file_path is None:
            stl_file_path = str(input_path.with_suffix(".stl"))
        output_path = Path(stl_file_path)

        # Load the numpy array
        data = np.load(input_path)

        # Validate data
        if data.ndim != EXPECTED_ARRAY_DIMENSIONS:
            return {
                "success": False,
                "message": f"Expected 2D array, got {data.ndim}D array. Shape: {data.shape}",
            }

        # Create mesh geometry
        height, width = data.shape
        vertices = _create_surface_vertices(data, scale_xy, scale_z, base_thickness)
        faces = _create_top_bottom_faces(height, width)
        faces.extend(_create_side_wall_faces(height, width))

        faces_array = np.array(faces)

        # Create the mesh
        beam_mesh = mesh.Mesh(np.zeros(faces_array.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces_array):
            for j in range(3):
                beam_mesh.vectors[i][j] = vertices[face[j]]

        # Save the STL file
        beam_mesh.save(str(output_path))

        return {
            "success": True,
            "stl_path": str(output_path),
            "input_shape": data.shape,
            "num_triangles": len(beam_mesh.vectors),
            "message": f"Successfully converted {npy_file_path} to {output_path}. "
            f"Created 3D mesh with {len(beam_mesh.vectors)} triangles. "
            f"Input shape: {data.shape}, scale_z={scale_z}, scale_xy={scale_xy}, base_thickness={base_thickness}",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error converting to STL: {e}",
        }
