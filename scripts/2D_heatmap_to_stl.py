#!/usr/bin/env python3
"""
Convert a 2D heatmap from .npy file to 3D STL file
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stl import mesh


def create_vertices_faces(data, scale_z=10, scale_xy=1, base_thickness=1):
    # Create top surface vertices - flip the Y coordinate
    height, width = data.shape
    vertices = []
    for i in range(height):
        for j in range(width):
            x = j * scale_xy
            y = (height - 1 - i) * scale_xy
            z = data[i, j] * scale_z + base_thickness
            vertices.append([x, y, z])

    # Same for bottom surface
    for i in range(height):
        for j in range(width):
            x = j * scale_xy
            y = (height - 1 - i) * scale_xy
            z = 0
            vertices.append([x, y, z])
    return vertices


def create_faces(data):
    height, width = data.shape
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            # Two triangles per grid square
            v1 = i * width + j
            v2 = i * width + (j + 1)
            v3 = (i + 1) * width + j
            v4 = (i + 1) * width + (j + 1)

            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    # Create faces for bottom surface
    offset = height * width
    for i in range(height - 1):
        for j in range(width - 1):
            v1 = offset + i * width + j
            v2 = offset + i * width + (j + 1)
            v3 = offset + (i + 1) * width + j
            v4 = offset + (i + 1) * width + (j + 1)

            # Reverse winding order for bottom
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])

    # Create side walls
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

    faces = np.array(faces)
    return faces


def create_stl_from_heatmap(data, scale_z=10, scale_xy=1, base_thickness=1):
    """
    Convert 2D heatmap to 3D STL file

    Parameters:
    - data: 2D numpy array with values 0-1
    - scale_z: how much to scale the height
    - scale_xy: how much to scale x and y dimensions
    - base_thickness: thickness of the base layer
    """

    # Create vertices for the surface

    vertices = create_vertices_faces(data, scale_z, scale_xy, base_thickness)
    faces = create_faces(data)

    faces = np.array(faces)

    # Create the mesh
    beam_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            beam_mesh.vectors[i][j] = vertices[face[j]]

    return beam_mesh


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D heatmap from .npy file to 3D STL file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.npy output.stl
  %(prog)s data.npy output.stl --scale-z 20 --scale-xy 2
  %(prog)s data.npy output.stl -z 15 -xy 1.5 -t 2
        """,
    )

    parser.add_argument("input", help="Input .npy file containing 2D array")
    parser.add_argument("output", help="Output .stl file path")
    parser.add_argument(
        "-z",
        "--scale-z",
        type=float,
        default=10,
        help="Height scale factor (default: 10)",
    )
    parser.add_argument(
        "-xy", "--scale-xy", type=float, default=1, help="X/Y scale factor (default: 1)"
    )
    parser.add_argument(
        "-t", "--thickness", type=float, default=1, help="Base thickness (default: 1)"
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

        # Create the STL
        print(
            f"Generating STL with scale_z={args.scale_z}, scale_xy={args.scale_xy}, thickness={args.thickness}..."
        )
        beam_mesh = create_stl_from_heatmap(
            data,
            scale_z=args.scale_z,
            scale_xy=args.scale_xy,
            base_thickness=args.thickness,
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
