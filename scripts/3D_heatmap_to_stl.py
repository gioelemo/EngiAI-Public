from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from stl import mesh  # type: ignore[import-untyped]

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {
    "train": "data/train-00000-of-00001.parquet",
    "val": "data/val-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/IDEALLab/heat_conduction_3d_v0/" + splits["train"])

# Filter by design parameters instead of index
TARGET_VOLUME = 0.4  # Target volume fraction (0.0 to 1.0)
TARGET_AREA = 0.4  # Target surface area ratio (0.0 to 1.0)
TOLERANCE = 0.05  # Tolerance for matching (+/- 5%)

# Find designs matching the target parameters
filtered_df = df[
    (df["volume"] >= TARGET_VOLUME - TOLERANCE)
    & (df["volume"] <= TARGET_VOLUME + TOLERANCE)
    & (df["area"] >= TARGET_AREA - TOLERANCE)
    & (df["area"] <= TARGET_AREA + TOLERANCE)
]

if len(filtered_df) == 0:
    print(
        f"Warning: No designs found matching volume={TARGET_VOLUME}, area={TARGET_AREA}"
    )
    print("Using closest match instead...")
    # Find closest match by computing distance
    df["distance"] = (
        (df["volume"] - TARGET_VOLUME) ** 2 + (df["area"] - TARGET_AREA) ** 2
    ) ** 0.5
    closest_idx = df["distance"].idxmin()
    selected_design = df.loc[closest_idx]
    print(
        f"Selected design: volume={selected_design['volume']:.3f}, area={selected_design['area']:.3f}"
    )
else:
    print(f"Found {len(filtered_df)} designs matching criteria")
    # Use the first matching design
    selected_design = filtered_df.iloc[0]
    print(
        f"Selected design: volume={selected_design['volume']:.3f}, area={selected_design['area']:.3f}"
    )

optimal_design_array = selected_design["optimal_design"]

# Debug: check what we're dealing with
print(f"Type: {type(optimal_design_array)}")
print(f"Is ndarray: {isinstance(optimal_design_array, np.ndarray)}")
if isinstance(optimal_design_array, np.ndarray):
    print(f"Dtype: {optimal_design_array.dtype}")
    print(f"Shape: {optimal_design_array.shape}")

# Parquet stores nested arrays in a special way
# Use numpy's stack/concatenate to properly handle nested structures
if isinstance(optimal_design_array, (list, np.ndarray)):
    # Recursively convert - use np.asarray which handles nested structures better
    try:
        design = np.asarray(optimal_design_array, dtype=np.float64)
    except (ValueError, TypeError):
        # If that fails, manually stack the nested structure
        # Assuming shape is (depth, height, width) or similar
        design = np.stack(
            [
                np.stack([np.array(row, dtype=np.float64) for row in layer])
                for layer in optimal_design_array
            ]
        )
else:
    design = np.array(optimal_design_array, dtype=np.float64)

print(f"Design type: {type(design)}")
print(f"Design shape: {design.shape}")
print(f"Design dtype: {design.dtype}")
print(f"Design min: {design.min():.3f}, max: {design.max():.3f}")

# Get the actual size from the array dimensions
size = design.shape[0] + 1  # Assuming cubic grid

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x, y, z = np.indices((size + 1, size + 1, size + 1)) / size  # Normalize to [0,1]

# Define which voxels to plot (threshold for material presence)
threshold = 0.5
filled = design > threshold
print(
    f"Voxels filled: {np.sum(filled)} / {design.size} ({100 * np.sum(filled) / design.size:.1f}%)"
)
# Adjust voxel positions by shifting their centers
ax.voxels(
    x[:-1, :-1, :-1],
    y[:-1, :-1, :-1],
    z[:-1, :-1, :-1],
    filled,
    edgecolor="k",
    alpha=0.7,
)

# ============================================================================
# Generate STL from voxel data
# ============================================================================


def voxel_to_stl(
    voxel_array,
    threshold=0.5,
    output_path="outputs/heat_conduction_3d.stl",
    add_base=True,
):
    """
    Convert a 3D voxel array to an STL mesh suitable for 3D printing.

    Args:
        voxel_array: 3D numpy array with material density values (0-1)
        threshold: Threshold for material presence (default: 0.5)
        output_path: Path to save the STL file
        add_base: Add a solid base layer for stability (default: True)
    """
    # Create binary voxel grid
    voxels = voxel_array > threshold

    # Add a solid base layer for stability (prevents floating parts warning)
    if add_base:
        print("\nAdding solid base layer for print stability...")
        # Add 2-3 layers of solid base at the bottom (z=0)
        voxels_with_base = np.copy(voxels)
        base_height = min(3, voxels.shape[2])  # 3 layers or less if grid is small
        voxels_with_base[:, :, :base_height] = True
        voxels = voxels_with_base
        print(f"Base height: {base_height} layers")

    print("\nGenerating STL mesh...")
    print(f"Voxel grid size: {voxels.shape}")
    print(f"Filled voxels: {np.sum(voxels)} / {voxels.size}")

    # Create vertices and faces for each filled voxel
    vertices = []
    faces = []

    # Define the 8 corners of a unit cube
    cube_vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # Bottom face
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],  # Top face
        ]
    )

    # Define the 12 triangles (2 per face, 6 faces)
    cube_faces = np.array(
        [
            # Bottom face (z=0)
            [0, 1, 2],
            [0, 2, 3],
            # Top face (z=1)
            [4, 6, 5],
            [4, 7, 6],
            # Front face (y=0)
            [0, 5, 1],
            [0, 4, 5],
            # Back face (y=1)
            [2, 7, 3],
            [2, 6, 7],
            # Left face (x=0)
            [0, 3, 7],
            [0, 7, 4],
            # Right face (x=1)
            [1, 6, 2],
            [1, 5, 6],
        ]
    )

    # Iterate through all voxels
    vertex_offset = 0
    for i in range(voxels.shape[0]):
        for j in range(voxels.shape[1]):
            for k in range(voxels.shape[2]):
                if voxels[i, j, k]:
                    # Add vertices for this voxel (shifted by i,j,k position)
                    voxel_verts = cube_vertices + np.array([i, j, k])
                    vertices.append(voxel_verts)

                    # Add faces for this voxel (with correct vertex offset)
                    voxel_faces = cube_faces + vertex_offset
                    faces.append(voxel_faces)

                    vertex_offset += 8

    if len(vertices) == 0:
        print("Error: No voxels above threshold. Try lowering the threshold.")
        return None

    # Concatenate all vertices and faces
    all_vertices = np.vstack(vertices)
    all_faces = np.vstack(faces)

    print(f"Total vertices: {len(all_vertices)}")
    print(f"Total triangles: {len(all_faces)}")

    # Create mesh
    stl_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))

    for i, face in enumerate(all_faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = all_vertices[face[j]]

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    stl_mesh.save(output_path)
    print(f"\n✓ STL file saved to: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    return stl_mesh


# Generate STL with solid base for better printability
output_file = "outputs/heat_conduction_3d_design.stl"
stl_mesh = voxel_to_stl(
    design,
    threshold=threshold,
    output_path=output_file,
    add_base=True,  # Set to False if you don't want a solid base
)

# Show visualization
open_window = True  # Set to True to display the matplotlib visualization
if open_window:
    plt.show()

print("\n" + "=" * 60)
print("STL generation complete!")
print("=" * 60)
print("You can now:")
print(
    f"  - Open in PrusaSlicer: open_gui_application('PrusaSlicer', file_path='{output_file}')"
)
print("  - View in any 3D viewer")
print("  - 3D print the design")
print("\nSLICING RECOMMENDATIONS:")
print("  ✓ Solid base layer added for stability")
print("  ✓ Orient model with base on print bed")
print("  - Use 'Generate support' for overhangs if needed")
print("  - Enable 'Brim' for better bed adhesion")
print("  - Recommended infill: 15-20%")
print("=" * 60)
