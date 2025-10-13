"""
STL export utilities for beam design arrays.

This module provides functions to convert 2D beam topology designs
into 3D STL files suitable for 3D printing or CAD software.
"""

from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.tools import tool
from stl import mesh


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
    scale_xy: float = 1.0,
    scale_z: float = 10.0,
    base_thickness: float = 1.0,
) -> dict[str, Any]:
    """
    Convert a 2D beam design array (.npy file) to a 3D STL file for 3D printing or CAD.

    This tool takes a numpy array file containing a 2D topology design
    and converts it into a 3D mesh (STL format). The height of each point
    in the mesh is determined by the density value in the design array,
    scaled by scale_z parameter.

    Args:
        npy_file_path: Path to the input .npy file containing the design array
        stl_file_path: Output path for the STL file (default: same name as npy with .stl extension)
        scale_xy: Scaling factor for X and Y dimensions (default: 1.0)
        scale_z: Scaling factor for Z dimension (height) (default: 10.0)
        base_thickness: Thickness of the base plate in Z units (default: 1.0)

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

        To specify custom output path and scaling:
        >>> result = convert_design_to_stl(
        ...     npy_file_path="optimized_beam.npy",
        ...     stl_file_path="my_beam.stl",
        ...     scale_z=15.0
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

        # Determine output path
        if stl_file_path is None:
            stl_file_path = str(input_path.with_suffix(".stl"))
        output_path = Path(stl_file_path)

        # Validate data dimensions
        if data.ndim != 2:  # noqa: PLR2004
            return {
                "success": False,
                "error": f"Expected 2D array, got {data.ndim}D array",
            }

        height, width = data.shape

        # Create mesh geometry
        vertices = _create_surface_vertices(data, scale_xy, scale_z, base_thickness)
        faces = []
        faces.extend(_create_top_bottom_faces(height, width))
        faces.extend(_create_side_wall_faces(height, width))

        # Create the mesh
        beam_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                beam_mesh.vectors[i][j] = vertices[face[j]]

        # Save to STL file
        beam_mesh.save(str(output_path))

        return {
            "success": True,
            "stl_path": str(output_path),
            "npy_path": str(input_path),
            "design_shape": data.shape,
            "num_triangles": len(faces),
            "scale_xy": scale_xy,
            "scale_z": scale_z,
            "base_thickness": base_thickness,
            "message": f"Successfully converted {input_path.name} to {output_path.name}. "
            f"Created 3D mesh with {len(faces)} triangles. "
            f"Dimensions: {width}x{height} grid, scaled by {scale_xy}x{scale_xy}x{scale_z}",
        }
    except ImportError:
        return {
            "success": False,
            "error": "numpy-stl not installed. Install with: pip install numpy-stl",
        }
    except Exception as e:
        return {"success": False, "error": f"STL conversion failed: {e!s}"}
