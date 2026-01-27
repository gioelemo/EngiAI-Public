import time
from pathlib import Path

import gmsh  # type: ignore[import-untyped]
import trimesh


def benchmark_trimesh(file_path):
    start = time.time()
    mesh = trimesh.load(file_path)
    # Trimesh check
    is_solid = mesh.is_watertight
    end = time.time()
    return is_solid, end - start


def benchmark_gmsh(file_path):
    start = time.time()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Mute console output
    try:
        gmsh.open(file_path)
        # Gmsh identifies if surfaces form a closed shell
        entities = gmsh.model.getEntities(2)
        # In Gmsh, we check for holes by looking for boundary loops (1D) on 2D surfaces
        has_holes = False
        for entity in entities:
            if len(gmsh.model.getBoundary([entity])) > 0:
                has_holes = True
                break
        is_solid = not has_holes
    except Exception:
        is_solid = False

    gmsh.finalize()
    end = time.time()
    return is_solid, end - start


# Execution
file_path = "/Users/gioelemolinari/Desktop/test.stl"  # Replace with your STL path
path_obj = Path(file_path)

if path_obj.exists():
    t_solid, t_time = benchmark_trimesh(file_path)
    g_solid, g_time = benchmark_gmsh(file_path)

    print(f"--- Results for {file_path} ---")
    print(f"Trimesh: Watertight={t_solid} | Time={t_time:.4f}s")
    print(f"Gmsh:    Watertight={g_solid} | Time={g_time:.4f}s")
else:
    print("File not found. Please provide a valid STL path.")
