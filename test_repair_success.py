"""
Demonstrate successful mesh repair with a mesh that has fixable issues.
"""

import numpy as np
import trimesh

print("=" * 60)
print("EXAMPLE 1: Merging Duplicate Vertices")
print("=" * 60)

# Create a simple box and intentionally duplicate vertices
box = trimesh.creation.box(extents=[10, 10, 10])

print(f"\nOriginal box:")
print(f"  Watertight: {box.is_watertight}")
print(f"  Vertices: {len(box.vertices)}")
print(f"  Faces: {len(box.faces)}")
print(f"  Volume: {box.volume:.2f}")

# Create a mesh with duplicate vertices (common voxel mesh issue)
# Duplicate all vertices
duplicated_vertices = np.vstack([box.vertices, box.vertices])
# Update faces to reference different vertex indices for same positions
offset_faces = box.faces + len(box.vertices)
all_faces = np.vstack([box.faces, offset_faces])

damaged = trimesh.Trimesh(vertices=duplicated_vertices, faces=all_faces, process=False)

print(f"\nWith duplicate vertices:")
print(f"  Vertices: {len(damaged.vertices)} (doubled!)")
print(f"  Faces: {len(damaged.faces)}")

# Repair by merging duplicates
damaged.merge_vertices()
print(f"\nAfter merge_vertices():")
print(f"  Vertices: {len(damaged.vertices)} (merged back)")
print(f"  Faces: {len(damaged.faces)}")
print(f"  Watertight: {damaged.is_watertight}")

print("\n" + "=" * 60)
print("EXAMPLE 2: Filling Holes in a Sphere")
print("=" * 60)

# Create a sphere
sphere = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
print(f"\nOriginal sphere:")
print(f"  Watertight: {sphere.is_watertight}")
print(f"  Faces: {len(sphere.faces)}")
print(f"  Volume: {sphere.volume:.2f}")

# Create a hole by removing some faces
num_faces_to_remove = 3
sphere_with_hole = trimesh.Trimesh(
    vertices=sphere.vertices,
    faces=sphere.faces[num_faces_to_remove:],
    process=False
)

print(f"\nWith {num_faces_to_remove} faces removed (creating hole):")
print(f"  Watertight: {sphere_with_hole.is_watertight}")
print(f"  Faces: {len(sphere_with_hole.faces)}")

# Attempt repair
sphere_with_hole.fill_holes()

print(f"\nAfter fill_holes():")
print(f"  Watertight: {sphere_with_hole.is_watertight}")
print(f"  Faces: {len(sphere_with_hole.faces)}")

if sphere_with_hole.is_watertight:
    print(f"  Volume: {sphere_with_hole.volume:.2f}")
    print("\n✓ SUCCESS: Small hole was successfully filled!")
else:
    print("\n✗ Hole was too complex to fill automatically")

print("\n" + "=" * 60)
print("EXAMPLE 3: Using trimesh.repair")
print("=" * 60)

# Create a cylinder with intentional issues
cylinder = trimesh.creation.cylinder(radius=5.0, height=10.0)
print(f"\nOriginal cylinder:")
print(f"  Watertight: {cylinder.is_watertight}")
print(f"  Volume: {cylinder.volume:.2f}")

# Damage it by removing some faces
damaged_cylinder = trimesh.Trimesh(
    vertices=cylinder.vertices,
    faces=cylinder.faces[10:],  # Remove 10 faces
    process=False
)

print(f"\nDamaged cylinder:")
print(f"  Watertight: {damaged_cylinder.is_watertight}")
print(f"  Faces: {len(damaged_cylinder.faces)}")

# Use trimesh.repair module
print(f"\nApplying repairs...")
damaged_cylinder.merge_vertices()
damaged_cylinder.fill_holes()

print(f"\nAfter repair:")
print(f"  Watertight: {damaged_cylinder.is_watertight}")
print(f"  Faces: {len(damaged_cylinder.faces)}")

if damaged_cylinder.is_watertight:
    print(f"  Volume: {damaged_cylinder.volume:.2f}")
    print("\n✓ SUCCESS: Mesh was successfully repaired!")
else:
    print("\n✗ Damage was too severe for automatic repair")

print("\n" + "=" * 60)
print("KEY TAKEAWAY")
print("=" * 60)
print("""
Mesh repair can fix:
  ✓ Duplicate vertices (merge_vertices)
  ✓ Small holes (fill_holes)
  ✓ Simple topology issues

Mesh repair CANNOT fix:
  ✗ Internal faces (voxel mesh issue)
  ✗ Non-manifold edges (complex topology)
  ✗ Large or complex holes
  ✗ Fundamental geometric problems

The voxel extrusion method creates non-manifold edges
where cubes meet, which is why repair doesn't work for those meshes.
""")
