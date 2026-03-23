#!/usr/bin/env python3
"""
Convert a 2D heatmap from .npy file to 3D STL file with extrusion
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stl import mesh


def create_stl_from_heatmap(data, scale_z=10, scale_xy=1):
    """
    Convert 2D binary heatmap to 3D STL file with extrusion
    Creates solid blocks for cells with value 1

    Parameters:
    - data: 2D numpy array with values 0 or 1
    - scale_z: extrusion height
    - scale_xy: how much to scale x and y dimensions
    """
    height, width = data.shape
    faces = []

    # Process each cell
    for i in range(height):
        for j in range(width):
            if data[i, j] == 0:
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D binary heatmap from .npy file to 3D STL file with extrusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.npy output.stl
  %(prog)s data.npy output.stl --scale-z 20 --scale-xy 2
  %(prog)s data.npy output.stl -z 15 -xy 1.5
        """,
    )

    parser.add_argument("input", help="Input .npy file containing 2D array")
    parser.add_argument("output", help="Output .stl file path")
    parser.add_argument(
        "-z",
        "--scale-z",
        type=float,
        default=10,
        help="Extrusion height (default: 10)",
    )
    parser.add_argument(
        "-xy", "--scale-xy", type=float, default=1, help="X/Y scale factor (default: 1)"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    if not args.input.endswith(".npy"):
        print("Warning: Input file doesn't have .npy extension", file=sys.stderr)

    try:
        # Load the numpy array
        print(f"Loading data from {args.input}...")
        data = np.load(args.input)

        dimension = 2
        # Validate data
        if data.ndim != dimension:
            print(f"Error: Expected 2D array, got {data.ndim}D array", file=sys.stderr)
            sys.exit(1)

        print(f"Data shape: {data.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")

        # Round to binary values (0 or 1)
        print("Rounding values to binary (0 or 1)...")
        data = np.round(data)
        print(f"After rounding: [{data.min():.0f}, {data.max():.0f}]")

        # Count non-zero cells
        non_zero_count = np.count_nonzero(data)
        print(
            f"Non-zero cells: {non_zero_count} ({non_zero_count / data.size * 100:.1f}%)"
        )

        # Create the STL
        print(
            f"Generating extruded STL with height={args.scale_z}, scale_xy={args.scale_xy}..."
        )
        beam_mesh = create_stl_from_heatmap(
            data,
            scale_z=args.scale_z,
            scale_xy=args.scale_xy,
        )

        # Save the STL
        print(f"Saving to {args.output}...")
        beam_mesh.save(args.output)

        print(f"Success! STL file created with {len(beam_mesh.vectors)} triangles")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
