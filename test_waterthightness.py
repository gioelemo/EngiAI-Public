import numpy as np

from src.tools.stl_export import convert_design_to_stl

# Create a simple 3x3 connected design
design = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# Save design to file
np.save("test_design.npy", design)

# Convert to STL with watertightness check
result = convert_design_to_stl.invoke(
    {"npy_file_path": "test_design.npy", "stl_file_path": "test_output.stl"}
)

# Print results
print("Success:", result["success"])
print("STL Path:", result["stl_path"])
print("\n--- 2D Connectivity (Precursor Check) ---")
print("Connected Design:", result["connected_design"])
print("Num Components:", result["num_components"])
print("\n--- 3D Watertightness (Printability Check) ---")
print("Watertight:", result["is_watertight"])
print("Volume (mm³):", result["volume_mm3"])
print("Surface Area (mm²):", result["surface_area_mm2"])
print("Num Vertices:", result["num_vertices"])
print("Num Faces:", result["num_faces"])
print("Validation Time:", result["mesh_validation_time"], "seconds")
print("\n--- Mesh Repair ---")
print("Repair Attempted:", result["repair_attempted"])
print("Mesh Repaired:", result["mesh_repaired"])
print("\nMessage:", result["message"])
