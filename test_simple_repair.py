"""
Simple example showing successful mesh repair.
"""

import numpy as np
import trimesh

print("=" * 70)
print("CREATING A MESH THAT CAN BE REPAIRED")
print("=" * 70)

# Start with a perfect box
perfect_box = trimesh.creation.box(extents=[10, 10, 10])
print(f"\nPerfect box:")
print(f"  Watertight: {perfect_box.is_watertight}")
print(f"  Vertices: {len(perfect_box.vertices)}")
print(f"  Volume: {perfect_box.volume:.2f}")

# Create a mesh with ONLY duplicate vertices (simplest fixable issue)
# Keep the exact same connectivity but duplicate vertex positions
print(f"\n{'='*70}")
print("DAMAGING: Duplicating vertices at same positions")
print("="*70)

# Original: 8 vertices for a box
# We'll create the same box but with each vertex duplicated
# Each face will use the duplicate instead of sharing vertices

vertices_original = perfect_box.vertices.copy()
n_original = len(vertices_original)

# Create duplicated vertices - same positions, different indices
vertices_damaged = np.vstack([
    vertices_original,  # Original 8 vertices
    vertices_original,  # Duplicate 8 vertices (indices 8-15)
])

# Modify faces to use duplicate vertices instead of shared ones
# This creates the same geometry but with unnecessary duplicate vertices
faces_damaged = perfect_box.faces.copy()
# Use second set of vertices for second half of faces
faces_damaged[6:] = faces_damaged[6:] + n_original

mesh_with_duplicates = trimesh.Trimesh(
    vertices=vertices_damaged,
    faces=faces_damaged,
    process=False  # Don't auto-process
)

print(f"\nMesh with duplicate vertices:")
print(f"  Vertices: {len(mesh_with_duplicates.vertices)} (should be 8, but is 16)")
print(f"  Faces: {len(mesh_with_duplicates.faces)}")
print(f"  Watertight: {mesh_with_duplicates.is_watertight}")

# REPAIR: Merge duplicate vertices
print(f"\n{'='*70}")
print("REPAIRING: Merging duplicate vertices")
print("="*70)

mesh_with_duplicates.merge_vertices()

print(f"\nAfter merge_vertices():")
print(f"  Vertices: {len(mesh_with_duplicates.vertices)} (merged down to 8)")
print(f"  Faces: {len(mesh_with_duplicates.faces)}")
print(f"  Watertight: {mesh_with_duplicates.is_watertight}")

if mesh_with_duplicates.is_watertight:
    print(f"  Volume: {mesh_with_duplicates.volume:.2f}")
    print(f"\n{'✓'*35}")
    print("✓ SUCCESS! Mesh repair worked!")
    print(f"{'✓'*35}")
else:
    print("\n✗ Still not watertight (might have other issues)")

print(f"\n{'='*70}")
print("WHY THIS WORKS vs VOXEL MESHES")
print("="*70)
print("""
This repair works because:
  • The mesh has proper topology (manifold edges)
  • Only issue was duplicate vertices at same positions
  • merge_vertices() collapsed duplicates

Voxel meshes don't repair because:
  • Non-manifold edges where cubes meet
  • Internal faces between adjacent voxels
  • Fundamental topology issues, not just duplicates
  • Would need complete remeshing, not simple repair

Think of it like:
  • Fixable: Two copies of the same clean blueprint
  • Not fixable: A building with walls going through each other
""")
