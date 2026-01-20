"""
STL export utilities for beam design arrays.

This module provides functions to convert 2D beam topology designs
into 3D STL files suitable for 3D printing or CAD software.
"""

from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.tools import tool
from scipy.ndimage import label
from stl import mesh

# Threshold for considering a cell as non-zero (for binary designs)
CELL_THRESHOLD = 0.5


def _check_design_connectivity(data: np.ndarray) -> tuple[bool, int]:
    """Check if design forms a single connected component.

    Uses 8-connectivity (diagonals count) which is appropriate for 3D printing
    since diagonal voxels share edges when extruded.

    Args:
        data: Design array (will be thresholded at CELL_THRESHOLD)

    Returns:
        Tuple of (is_connected, num_components)
    """
    binary = (data >= CELL_THRESHOLD).astype(int)
    structure = np.ones((3, 3))  # 8-connectivity
    _, num_components = label(binary, structure=structure)
    return num_components == 1, int(num_components)


def _get_versioned_filename(base_path: Path) -> Path:
    """
    Generate a unique filename by adding version number if file exists.

    Args:
        base_path: The desired file path

    Returns:
        A unique file path with version number if needed (e.g., file_v1.stl, file_v2.stl)
    """
    if not base_path.exists():
        return base_path

    # File exists, create versioned name
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    # Check if the stem already has a version
    version = 1
    if "_v" in stem:
        parts = stem.rsplit("_v", 1)
        if len(parts) == 2 and parts[1].isdigit():  # noqa: PLR2004
            stem = parts[0]
            version = int(parts[1])

    # Find next available version
    while True:
        versioned_path = parent / f"{stem}_v{version}{suffix}"
        if not versioned_path.exists():
            return versioned_path
        version += 1


def _mirror_beam_along_y(data: np.ndarray) -> np.ndarray:
    """
    Mirror a half-beam design along the Y-axis to create a full symmetric beam.

    The function takes the half beam and mirrors it to create a complete beam.
    This is useful when optimization is done on half the beam due to symmetry.

    Args:
        data: 2D numpy array representing half of the beam design

    Returns:
        2D numpy array with the beam mirrored along Y-axis
    """
    # Flip the array along the first axis (rows/Y-axis)
    mirrored = np.flip(data, axis=1)

    # Concatenate original and mirrored parts
    # Stack vertically to extend along Y-axis
    full_beam = np.hstack([mirrored, data])

    return full_beam


def _create_stl_from_heatmap_extruded(
    data: np.ndarray, scale_z: float = 10.0, scale_xy: float = 1.0
) -> mesh.Mesh:
    """
    Convert 2D binary heatmap to 3D STL file with extrusion.
    Creates solid blocks for cells with non-zero values.

    Args:
        data: 2D numpy array with values (typically 0 or 1)
        scale_z: Extrusion height for non-zero cells
        scale_xy: Scaling factor for x and y dimensions

    Returns:
        mesh.Mesh object containing the 3D model
    """
    height, width = data.shape
    faces: list[list[list[float]]] = []

    # Process each cell
    for i in range(height):
        for j in range(width):
            # Skip cells with zero or near-zero values
            if data[i, j] < CELL_THRESHOLD:
                continue

            # Define the 8 vertices of a cube for this cell
            x = j * scale_xy
            y = (height - 1 - i) * scale_xy
            x1 = (j + 1) * scale_xy
            y1 = (height - 2 - i) * scale_xy
            z_top = scale_z
            z_bottom = 0

            # 8 vertices of the cube
            vertices = [
                [x, y, z_top],  # 0: top-left-front
                [x1, y, z_top],  # 1: top-right-front
                [x1, y1, z_top],  # 2: top-right-back
                [x, y1, z_top],  # 3: top-left-back
                [x, y, z_bottom],  # 4: bottom-left-front
                [x1, y, z_bottom],  # 5: bottom-right-front
                [x1, y1, z_bottom],  # 6: bottom-right-back
                [x, y1, z_bottom],  # 7: bottom-left-back
            ]

            # Create the 12 triangles (2 per face, 6 faces)
            cube_faces = [
                # Top face
                [0, 1, 2],
                [0, 2, 3],
                # Bottom face (reversed winding)
                [4, 6, 5],
                [4, 7, 6],
                # Front face
                [0, 4, 5],
                [0, 5, 1],
                # Back face
                [2, 6, 7],
                [2, 7, 3],
                # Left face
                [0, 3, 7],
                [0, 7, 4],
                # Right face
                [1, 5, 6],
                [1, 6, 2],
            ]

            # Add these faces to our list (with absolute vertex positions)
            faces.extend(
                [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
                for face in cube_faces
            )

    if len(faces) == 0:
        raise ValueError

    # Create the mesh
    beam_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            beam_mesh.vectors[i][j] = face[j]

    return beam_mesh


@tool
def convert_design_to_stl(  # noqa: PLR0913, PLR0912
    npy_file_path: str,
    stl_file_path: str | None = None,
    scale_xy: float = 1.0,
    scale_z: float = 10.0,
    mirror_y: bool = False,
    problem_type: str = "beams2d",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Convert a 2D topology optimization design (.npy file) to a 3D STL file for 3D printing or CAD.

    This tool takes a numpy array file containing a 2D topology design
    and converts it into a 3D mesh (STL format) using extrusion.

    For discrete designs (beams2d): Rounds values to binary (0 or 1) and extrudes solid cells.
    For continuous designs (thermoelastic2d): Uses threshold to determine solid regions,
    preserving the continuous density distribution in the Z-height.

    All STL files are automatically saved to the 'outputs/' directory.
    If the output file already exists, a version number is automatically added
    (e.g., design_v1.stl, design_v2.stl) to preserve the history.

    Args:
        npy_file_path: Path to the input .npy file containing the design array
        stl_file_path: Output filename for the STL file (will be saved in outputs/ directory)
            Default: same name as npy with .stl extension
            Note: If file exists, version number will be auto-added to preserve history
        scale_xy: Scaling factor for X and Y dimensions (default: 1.0)
        scale_z: Extrusion height for cells (default: 10.0)
        mirror_y: If True, mirror the design along Y-axis to create full symmetric structure
            from half-structure simulation (default: False)
        problem_type: Type of problem ("beams2d" or "thermoelastic2d", default: "beams2d")
            Determines whether to use binary rounding or continuous density
        threshold: For continuous problems, density threshold for solid vs void (default: 0.5)
            Only cells with density >= threshold will be included in the STL

    Returns:
        dict with conversion results:
        - success: bool
        - stl_path: str (where STL file was saved)
        - npy_path: str (input file path)
        - num_triangles: int (number of triangles in the mesh)
        - message: str

    Example:
        To convert beam_design.npy to beam_design.stl with default settings:
        >>> result = convert_design_to_stl("beam_design.npy")
        >>> print(result['message'])

        To convert a thermoelastic design (continuous densities):
        >>> result = convert_design_to_stl(
        ...     npy_file_path="thermoelastic_design.npy",
        ...     problem_type="thermoelastic2d",
        ...     threshold=0.3
        ... )

        To convert a half-beam and mirror it to create a full beam:
        >>> result = convert_design_to_stl(
        ...     npy_file_path="half_beam.npy",
        ...     problem_type="beams2d",
        ...     mirror_y=True
        ... )

        To specify custom output path and scaling:
        >>> result = convert_design_to_stl(
        ...     npy_file_path="optimized_beam.npy",
        ...     stl_file_path="my_beam.stl",
        ...     scale_z=15.0,
        ...     mirror_y=True
        ... )
    """
    try:
        # Load the design array
        input_path = Path(npy_file_path)
        if not input_path.exists():
            return {
                "success": False,
                "error": f"File not found: {npy_file_path}",
            }

        data = np.load(input_path)

        # Create outputs directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Determine output path (always in outputs/)
        if stl_file_path is None:
            # Default: same name as input but with .stl extension, in outputs/
            stl_file_path = str(output_dir / input_path.with_suffix(".stl").name)
        else:
            # Ensure output is in outputs/ directory
            stl_path_obj = Path(stl_file_path)
            if stl_path_obj.parent.name != "outputs":
                stl_file_path = str(output_dir / stl_path_obj.name)

        output_path = Path(stl_file_path)

        # Create versioned filename to preserve history (avoid overwriting)
        output_path = _get_versioned_filename(output_path)

        # Validate data dimensions
        if data.ndim != 2:  # noqa: PLR2004
            return {
                "success": False,
                "error": f"Expected 2D array, got {data.ndim}D array",
            }

        height, width = data.shape

        # Apply problem-specific processing
        if problem_type.lower() == "beams2d":
            # For beams: round to binary (0 or 1)
            data = np.round(data)
        else:
            # For continuous problems (thermoelastic2d): threshold densities
            # Keep cells above threshold, zero out below threshold
            data = np.where(data >= threshold, data, 0.0)

        # Mirror along Y-axis if requested (for half-structure to full-structure conversion)
        if mirror_y:
            if problem_type.lower() == "beams2d":
                data = _mirror_beam_along_y(data)
            else:
                # For non-beam problems, use generic mirroring
                data = np.concatenate([data, np.fliplr(data)], axis=1)
            height, width = data.shape

        # Count non-zero cells
        non_zero_count = np.count_nonzero(data)

        # Check design connectivity
        connected_design, num_components = _check_design_connectivity(data)

        # Create mesh using extruded blocks method
        beam_mesh = _create_stl_from_heatmap_extruded(data, scale_z, scale_xy)

        # Save to STL file
        beam_mesh.save(str(output_path))

        num_triangles = len(beam_mesh.vectors)

        return {
            "success": True,
            "stl_path": str(output_path),
            "npy_path": str(input_path),
            "design_shape": data.shape,
            "num_triangles": num_triangles,
            "non_zero_cells": int(non_zero_count),
            "connected_design": connected_design,
            "num_components": num_components,
            "scale_xy": scale_xy,
            "scale_z": scale_z,
            "mirrored": mirror_y,
            "message": f"Successfully converted {input_path.name} to outputs/{output_path.name}"
            + (" (mirrored along Y-axis)" if mirror_y else "")
            + f". Created 3D extruded mesh with {num_triangles} triangles from {non_zero_count} cells. "
            + f"Connected: {connected_design} ({num_components} component{'s' if num_components != 1 else ''}). "
            + f"Dimensions: {width}x{height} grid, scaled by {scale_xy}x{scale_xy}x{scale_z}",
        }
    except ValueError as e:
        return {
            "success": False,
            "error": f"No valid cells to convert: {e!s}",
        }
    except ImportError:
        return {
            "success": False,
            "error": "numpy-stl not installed. Install with: pip install numpy-stl",
        }
    except Exception as e:
        return {"success": False, "error": f"STL conversion failed: {e!s}"}
