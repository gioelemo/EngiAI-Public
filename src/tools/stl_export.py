"""
STL export utilities for beam design arrays.

This module provides functions to convert 2D beam topology designs
into 3D STL files suitable for 3D printing or CAD software.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.tools import tool
from scipy.ndimage import label
from stl import mesh

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

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


def _check_mesh_watertightness(
    stl_mesh: mesh.Mesh,
    scale_xy: float = 1.0,  # noqa: ARG001
    scale_z: float = 10.0,  # noqa: ARG001
    attempt_repair: bool = True,
) -> dict[str, Any]:
    """
    Check if STL mesh is watertight and extract 3D printability metrics.

    Uses trimesh library to validate mesh topology and compute geometric properties.
    If trimesh is not available, returns minimal info with warning.

    A watertight mesh is a closed manifold solid with no holes, gaps, or non-manifold edges.
    This is a critical requirement for 3D printing - non-watertight meshes may fail to slice
    or produce unexpected results.

    If the mesh is not watertight, this function can optionally attempt to repair it by:
    - Merging duplicate vertices
    - Removing duplicate faces
    - Filling holes

    Args:
        stl_mesh: numpy-stl mesh object
        scale_xy: XY scaling factor for volume/area units (reserved for future use)
        scale_z: Z scaling factor for volume/area units (reserved for future use)
        attempt_repair: Whether to attempt mesh repair if not watertight (default: True)

    Returns:
        Dictionary with:
        - is_watertight: bool (True if closed manifold, None if check unavailable)
        - volume_mm3: float (mesh volume in mm³, None if not watertight or unavailable)
        - surface_area_mm2: float (mesh surface area in mm², None if unavailable)
        - num_vertices: int (vertex count, None if unavailable)
        - num_faces: int (face/triangle count, None if unavailable)
        - watertight_check_available: bool (False if trimesh not installed)
        - mesh_validation_time: float (seconds, 0.0 if check unavailable)
        - mesh_repaired: bool (True if repair was attempted and successful)
        - repair_attempted: bool (True if repair was attempted)
    """
    start_time = time.time()

    # If trimesh not available, return minimal info
    if not TRIMESH_AVAILABLE:
        return {
            "is_watertight": None,
            "volume_mm3": None,
            "surface_area_mm2": None,
            "num_vertices": None,
            "num_faces": len(stl_mesh.vectors),  # Can get this from numpy-stl
            "watertight_check_available": False,
            "mesh_validation_time": 0.0,
            "mesh_repaired": False,
            "repair_attempted": False,
        }

    try:
        # Convert numpy-stl mesh to trimesh format
        # numpy-stl stores mesh as Nx3x3 array (N triangles, 3 vertices, 3 coords)
        vertices = stl_mesh.vectors.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)

        # Create trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Check if mesh is watertight
        is_watertight = trimesh_mesh.is_watertight
        mesh_repaired = False
        repair_attempted = False

        # Attempt to repair mesh if not watertight
        if not is_watertight and attempt_repair:
            repair_attempted = True
            try:
                # Merge duplicate vertices (common issue with voxel meshes)
                trimesh_mesh.merge_vertices()

                # Fill holes if any
                trimesh_mesh.fill_holes()

                # Check if repair was successful
                is_watertight_after_repair = trimesh_mesh.is_watertight
                if is_watertight_after_repair:
                    is_watertight = True
                    mesh_repaired = True
            except Exception:
                # If repair fails, continue with original mesh
                pass

        # Compute volume (only meaningful for watertight meshes)
        # Volume is in cubic units based on coordinate space
        volume = trimesh_mesh.volume if is_watertight else None

        # Compute surface area
        surface_area = trimesh_mesh.area

        # Get mesh complexity metrics
        num_vertices = len(trimesh_mesh.vertices)
        num_faces = len(trimesh_mesh.faces)

        validation_time = time.time() - start_time

        return {
            "is_watertight": bool(is_watertight),
            "volume_mm3": float(volume) if volume is not None else None,
            "surface_area_mm2": float(surface_area),
            "num_vertices": int(num_vertices),
            "num_faces": int(num_faces),
            "watertight_check_available": True,
            "mesh_validation_time": float(validation_time),
            "mesh_repaired": mesh_repaired,
            "repair_attempted": repair_attempted,
        }
    except Exception as e:
        # If trimesh validation fails, return error info
        validation_time = time.time() - start_time
        return {
            "is_watertight": None,
            "volume_mm3": None,
            "surface_area_mm2": None,
            "num_vertices": None,
            "num_faces": len(stl_mesh.vectors),
            "watertight_check_available": False,
            "mesh_validation_time": float(validation_time),
            "mesh_repaired": False,
            "repair_attempted": False,
            "error": f"Trimesh validation failed: {e!s}",
        }


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
        - connected_design: bool (2D connectivity: single component in design array)
        - num_components: int (2D connectivity: number of disconnected regions)
        - is_watertight: bool | None (3D printability: mesh forms closed manifold solid)
        - volume_mm3: float | None (mesh volume in mm³)
        - surface_area_mm2: float | None (mesh surface area in mm²)
        - num_vertices: int | None (mesh vertex count)
        - num_faces: int | None (mesh face count)
        - watertight_check_available: bool (whether trimesh validation was performed)
        - mesh_validation_time: float (seconds spent on watertightness check)
        - mesh_repaired: bool (whether mesh was successfully repaired)
        - repair_attempted: bool (whether repair was attempted)
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

        # Check mesh watertightness for 3D printability assessment
        watertightness_info = _check_mesh_watertightness(
            beam_mesh,
            scale_xy=scale_xy,
            scale_z=scale_z,
        )

        # Build message with watertightness info if available
        message = (
            f"Successfully converted {input_path.name} to outputs/{output_path.name}"
            + (" (mirrored along Y-axis)" if mirror_y else "")
            + f". Created 3D extruded mesh with {num_triangles} triangles from {non_zero_count} cells. "
            + f"Connected: {connected_design} ({num_components} component{'s' if num_components != 1 else ''}). "
            + f"Dimensions: {width}x{height} grid, scaled by {scale_xy}x{scale_xy}x{scale_z}"
        )

        # Add watertightness info to message if check was performed
        if watertightness_info.get("watertight_check_available"):
            is_wt = watertightness_info.get("is_watertight")
            if is_wt is not None:
                message += f". Watertight: {is_wt}"

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
            "message": message,
            **watertightness_info,  # Unpack all watertightness metrics
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
