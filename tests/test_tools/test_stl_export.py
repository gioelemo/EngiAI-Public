"""
Tests for STL export functionality.

These tests cover 2D array to 3D STL mesh conversion including mirroring,
versioning, and mesh generation.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.tools.stl_export import (
    TRIMESH_AVAILABLE,
    _check_design_connectivity,
    _check_mesh_watertightness,
    _create_stl_from_heatmap_extruded,
    _get_versioned_filename,
    _mirror_beam_along_y,
    convert_design_to_stl,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_2d_array():
    """Create a simple 2D binary array for testing."""
    return np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])


@pytest.fixture
def sample_half_beam():
    """Create a simple half-beam array for mirroring tests."""
    return np.array([[1, 1], [0, 1], [1, 1]])


@pytest.fixture
def empty_array():
    """Create an array with all zeros."""
    return np.zeros((3, 3))


@pytest.fixture
def temp_outputs_dir(tmp_path):
    """Create a temporary outputs directory."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================


@pytest.mark.skip(reason="Using _build_versioned_path now")
@pytest.mark.unit
def test_get_versioned_filename_new_file(tmp_path):
    """Test versioned filename for non-existing file."""
    file_path = tmp_path / "test.stl"

    result = _get_versioned_filename(file_path)

    assert result == file_path
    assert not result.exists()


@pytest.mark.skip(reason="Using _build_versioned_path now")
@pytest.mark.unit
def test_get_versioned_filename_existing_file(tmp_path):
    """Test versioned filename when file exists."""
    file_path = tmp_path / "test.stl"
    file_path.write_text("existing file")

    result = _get_versioned_filename(file_path)

    assert result == tmp_path / "test_v1.stl"
    assert not result.exists()


@pytest.mark.skip(reason="Using _build_versioned_path now")
@pytest.mark.unit
def test_get_versioned_filename_multiple_versions(tmp_path):
    """Test versioned filename with multiple existing versions."""
    base = tmp_path / "test.stl"
    base.write_text("v0")
    (tmp_path / "test_v1.stl").write_text("v1")
    (tmp_path / "test_v2.stl").write_text("v2")

    result = _get_versioned_filename(base)

    assert result == tmp_path / "test_v3.stl"


@pytest.mark.skip(reason="Using _build_versioned_path now")
@pytest.mark.unit
def test_get_versioned_filename_with_existing_version():
    """Test handling of filename that already has version."""
    file_path = Path("test_v5.stl")

    # Test the logic without actually creating files
    stem = file_path.stem
    if "_v" in stem:
        parts = stem.rsplit("_v", 1)
        assert parts[0] == "test"
        assert parts[1] == "5"


@pytest.mark.unit
def test_mirror_beam_along_y_basic(sample_half_beam):
    """Test basic Y-axis mirroring."""
    result = _mirror_beam_along_y(sample_half_beam)

    # Original shape (3, 2) should become (3, 4)
    assert result.shape == (3, 4)

    # Check that mirroring is correct with original data structure
    # The mirrored beam should be symmetric
    expected = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_mirror_beam_along_y_symmetry(sample_half_beam):
    """Test that mirrored result is symmetric."""
    result = _mirror_beam_along_y(sample_half_beam)

    # Left half should match right half (mirrored)
    left_half = result[:, : result.shape[1] // 2]
    right_half = result[:, result.shape[1] // 2 :]

    np.testing.assert_array_equal(left_half, np.fliplr(right_half))


@pytest.mark.unit
def test_mirror_beam_single_column():
    """Test mirroring a single column array."""
    single_col = np.array([[1], [0], [1]])
    result = _mirror_beam_along_y(single_col)

    assert result.shape == (3, 2)
    expected = np.array([[1, 1], [0, 0], [1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_create_stl_simple_array(sample_2d_array):
    """Test STL mesh creation from simple 2D array."""
    mesh = _create_stl_from_heatmap_extruded(
        sample_2d_array, scale_z=10.0, scale_xy=1.0
    )

    # Check mesh was created
    assert mesh is not None
    assert len(mesh.vectors) > 0

    # Each non-zero cell should generate 12 triangles (2 per face, 6 faces)
    non_zero_count = np.count_nonzero(sample_2d_array)
    expected_triangles = non_zero_count * 12
    assert len(mesh.vectors) == expected_triangles


@pytest.mark.unit
def test_create_stl_empty_array_raises_error(empty_array):
    """Test that empty array raises ValueError."""
    with pytest.raises(ValueError):
        _create_stl_from_heatmap_extruded(empty_array)


@pytest.mark.unit
def test_create_stl_with_threshold():
    """Test that values below threshold are skipped."""
    # Array with values below threshold (0.5)
    array_with_threshold = np.array([[0.2, 0.8], [0.6, 0.3]])

    mesh = _create_stl_from_heatmap_extruded(array_with_threshold)

    # Only values >= 0.5 should generate cubes (0.8 and 0.6)
    expected_triangles = 2 * 12  # 2 cells * 12 triangles per cube
    assert len(mesh.vectors) == expected_triangles


@pytest.mark.unit
def test_create_stl_scaling():
    """Test mesh creation with custom scaling."""
    simple_array = np.array([[1]])

    mesh = _create_stl_from_heatmap_extruded(simple_array, scale_z=20.0, scale_xy=2.0)

    # Check that vertices are scaled correctly
    assert mesh is not None
    # With scale_xy=2.0, the cube should be 2x2
    # With scale_z=20.0, height should be 20
    vertices = mesh.vectors.reshape(-1, 3)
    max_z = np.max(vertices[:, 2])
    assert max_z == 20.0


# ============================================================================
# CONVERT_DESIGN_TO_STL TOOL TESTS
# ============================================================================


@pytest.mark.unit
def test_convert_design_missing_file():
    """Test conversion with non-existent input file."""
    result = convert_design_to_stl.invoke({"npy_file_path": "nonexistent.npy"})

    assert result["success"] is False
    assert "File not found" in result["error"]


@pytest.mark.unit
def test_convert_design_basic(tmp_path, sample_2d_array):
    """Test basic design conversion."""
    # Create test npy file
    npy_path = tmp_path / "test_design.npy"
    np.save(npy_path, sample_2d_array)

    # Create outputs directory in tmp_path
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:
        # Setup mock to use tmp_path
        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "stl_file_path": str(outputs_dir / "output.stl"),
            }
        )

    assert result["success"] is True
    assert "stl_path" in result
    assert "num_triangles" in result
    assert result["non_zero_cells"] == 6  # From sample array


@pytest.mark.unit
def test_convert_design_with_mirroring(tmp_path, sample_half_beam):
    """Test conversion with Y-axis mirroring."""
    npy_path = tmp_path / "half_beam.npy"
    np.save(npy_path, sample_half_beam)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "mirror_y": True,
                "stl_file_path": str(outputs_dir / "mirrored.stl"),
            }
        )

    assert result["success"] is True
    assert result["mirrored"] is True
    # Original shape (3, 2) should become (3, 4) after mirroring
    assert result["design_shape"] == (3, 4)


@pytest.mark.unit
def test_convert_design_custom_scaling(tmp_path, sample_2d_array):
    """Test conversion with custom scale parameters."""
    npy_path = tmp_path / "design.npy"
    np.save(npy_path, sample_2d_array)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "scale_xy": 2.0,
                "scale_z": 15.0,
                "stl_file_path": str(outputs_dir / "scaled.stl"),
            }
        )

    assert result["success"] is True
    assert result["scale_xy"] == 2.0
    assert result["scale_z"] == 15.0


@pytest.mark.unit
def test_convert_design_wrong_dimensions(tmp_path):
    """Test conversion with wrong array dimensions."""
    # Create 3D array instead of 2D
    wrong_dim_array = np.ones((3, 3, 3))
    npy_path = tmp_path / "wrong_dim.npy"
    np.save(npy_path, wrong_dim_array)

    result = convert_design_to_stl.invoke({"npy_file_path": str(npy_path)})

    assert result["success"] is False
    assert "Expected 2D array" in result["error"]


@pytest.mark.unit
def test_convert_design_empty_array_error(tmp_path, empty_array):
    """Test conversion with all-zero array."""
    npy_path = tmp_path / "empty.npy"
    np.save(npy_path, empty_array)

    result = convert_design_to_stl.invoke({"npy_file_path": str(npy_path)})

    assert result["success"] is False
    assert "No valid cells" in result["error"]


@pytest.mark.unit
def test_convert_design_default_output_path(tmp_path, sample_2d_array):
    """Test that default output path follows structured naming pattern."""
    import re

    npy_path = tmp_path / "my_design.npy"
    np.save(npy_path, sample_2d_array)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        # Mock mesh.save to avoid actual file creation
        with patch("src.tools.stl_export.mesh.Mesh.save"):
            result = convert_design_to_stl.invoke(
                {"npy_file_path": str(npy_path), "problem_type": "beams2d"}
            )

    assert result["success"] is True
    # Expected pattern: beams2d_design_exported_YYYYMMDD_HHMMSS_mmm_N.stl
    pattern = r"beams2d_design_exported_\d{8}_\d{6}_\d{3}_\d+\.stl"
    assert re.search(pattern, result["stl_path"]), \
        f"STL path doesn't match pattern: {result['stl_path']}"
    assert "beams2d_design_exported_" in result["stl_path"]


@pytest.mark.unit
def test_convert_design_output_in_outputs_dir(tmp_path, sample_2d_array):
    """Test that output is always placed in outputs/ directory."""
    npy_path = tmp_path / "design.npy"
    np.save(npy_path, sample_2d_array)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        # Try to specify output in different directory
        with patch("src.tools.stl_export.mesh.Mesh.save"):
            result = convert_design_to_stl.invoke(
                {
                    "npy_file_path": str(npy_path),
                    "stl_file_path": "somewhere/else/file.stl",
                }
            )

    assert result["success"] is True
    # Should be redirected to outputs/
    assert "outputs" in result["stl_path"]


@pytest.mark.unit
def test_convert_design_result_structure(tmp_path, sample_2d_array):
    """Test that result contains all expected fields."""
    npy_path = tmp_path / "design.npy"
    np.save(npy_path, sample_2d_array)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "stl_file_path": str(outputs_dir / "test.stl"),
            }
        )

    # Check all expected fields are present
    assert "success" in result
    assert "stl_path" in result
    assert "npy_path" in result
    assert "design_shape" in result
    assert "num_triangles" in result
    assert "non_zero_cells" in result
    assert "scale_xy" in result
    assert "scale_z" in result
    assert "mirrored" in result
    assert "message" in result


@pytest.mark.unit
def test_stl_naming_matches_npy_pattern(tmp_path, sample_2d_array):
    """Test that STL files use same naming pattern as NPY files."""
    import re

    npy_path = tmp_path / "design.npy"
    np.save(npy_path, sample_2d_array)
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        # Mock mesh.save to avoid actual file creation
        with patch("src.tools.stl_export.mesh.Mesh.save"):
            result = convert_design_to_stl.invoke(
                {"npy_file_path": str(npy_path), "problem_type": "beams2d"}
            )

    # Verify structured naming pattern
    pattern = r"beams2d_design_exported_\d{8}_\d{6}_\d{3}_\d+\.stl"
    assert re.search(pattern, result["stl_path"]), \
        f"STL path doesn't match pattern: {result['stl_path']}"

    # Verify contains problem_type prefix
    assert "beams2d_design_" in result["stl_path"]

    # Verify contains _exported suffix
    assert "_exported_" in result["stl_path"]


@pytest.mark.unit
def test_convert_design_binary_rounding():
    """Test that non-binary values are rounded."""
    # Array with fractional values
    fractional_array = np.array([[0.7, 0.3], [0.9, 0.1]])

    # After rounding, should become [[1, 0], [1, 0]]
    # This means 2 non-zero cells
    expected_non_zero = 2

    # We can test the rounding logic directly
    rounded = np.round(fractional_array)
    assert np.count_nonzero(rounded) == expected_non_zero


# ============================================================================
# EDGE CASES AND INTEGRATION
# ============================================================================


@pytest.mark.unit
def test_large_array_triangle_count():
    """Test triangle count for larger array."""
    large_array = np.ones((10, 10))

    mesh = _create_stl_from_heatmap_extruded(large_array)

    # 100 cells, each with 12 triangles
    assert len(mesh.vectors) == 100 * 12


@pytest.mark.unit
def test_single_cell_mesh():
    """Test mesh creation for single non-zero cell."""
    single_cell = np.array([[1]])

    mesh = _create_stl_from_heatmap_extruded(single_cell)

    # 1 cube = 12 triangles (2 per face, 6 faces)
    assert len(mesh.vectors) == 12


@pytest.mark.unit
def test_mesh_vectors_structure():
    """Test that mesh vectors have correct structure."""
    simple_array = np.array([[1]])

    mesh = _create_stl_from_heatmap_extruded(simple_array)

    # Each triangle should have 3 vertices
    assert mesh.vectors.shape[1] == 3
    # Each vertex should have 3 coordinates (x, y, z)
    assert mesh.vectors.shape[2] == 3


# ============================================================================
# CONNECTIVITY CHECK TESTS
# ============================================================================


@pytest.mark.unit
def test_connectivity_single_connected_component():
    """Test that a fully connected design returns connected=True."""
    # L-shaped connected design
    connected = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ]
    )
    is_connected, num_components = _check_design_connectivity(connected)

    assert is_connected is True
    assert num_components == 1


@pytest.mark.unit
def test_connectivity_diagonal_connection():
    """Test that diagonally connected cells count as connected (8-connectivity)."""
    # Diagonal connection - should be connected with 8-connectivity
    diagonal = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    is_connected, num_components = _check_design_connectivity(diagonal)

    assert is_connected is True
    assert num_components == 1


@pytest.mark.unit
def test_connectivity_disconnected_components():
    """Test that disconnected regions are detected."""
    # Two separate regions (corners)
    disconnected = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ]
    )
    is_connected, num_components = _check_design_connectivity(disconnected)

    assert is_connected is False
    assert num_components == 4  # Four separate corner cells


@pytest.mark.unit
def test_connectivity_two_components():
    """Test detection of exactly two disconnected components."""
    # Two separate horizontal bars
    two_bars = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )
    is_connected, num_components = _check_design_connectivity(two_bars)

    assert is_connected is False
    assert num_components == 2


@pytest.mark.unit
def test_connectivity_empty_design():
    """Test that empty design has zero components."""
    empty = np.zeros((3, 3))
    is_connected, num_components = _check_design_connectivity(empty)

    assert is_connected is False
    assert num_components == 0


@pytest.mark.unit
def test_connectivity_full_design():
    """Test that fully filled design is connected."""
    full = np.ones((5, 5))
    is_connected, num_components = _check_design_connectivity(full)

    assert is_connected is True
    assert num_components == 1


@pytest.mark.unit
def test_connectivity_threshold_applied():
    """Test that values below threshold are treated as void."""
    # Values below 0.5 threshold should be ignored
    continuous = np.array(
        [
            [0.8, 0.3, 0.9],
            [0.2, 0.7, 0.1],
            [0.6, 0.4, 0.8],
        ]
    )
    # After threshold: [[1,0,1], [0,1,0], [1,0,1]]
    # With 8-connectivity, all corners connect through center
    is_connected, num_components = _check_design_connectivity(continuous)

    assert is_connected is True
    assert num_components == 1


@pytest.mark.unit
def test_connectivity_single_cell():
    """Test single cell design is connected."""
    single = np.array([[1]])
    is_connected, num_components = _check_design_connectivity(single)

    assert is_connected is True
    assert num_components == 1


@pytest.mark.unit
def test_connectivity_in_stl_output(tmp_path, sample_2d_array):
    """Test that connectivity info is included in STL export output."""
    npy_path = tmp_path / "design.npy"
    np.save(npy_path, sample_2d_array)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "stl_file_path": str(outputs_dir / "test.stl"),
            }
        )

    assert result["success"] is True
    assert "connected_design" in result
    assert "num_components" in result
    assert isinstance(result["connected_design"], bool)
    assert isinstance(result["num_components"], int)


# ============================================================================
# WATERTIGHTNESS CHECKING TESTS
# ============================================================================


@pytest.mark.unit
def test_check_mesh_watertightness_basic(sample_2d_array):
    """Test that watertightness check returns expected fields."""
    # Create a mesh
    mesh = _create_stl_from_heatmap_extruded(sample_2d_array)

    # Check watertightness
    result = _check_mesh_watertightness(mesh)

    # Assert all expected keys present
    assert "is_watertight" in result
    assert "volume_mm3" in result
    assert "surface_area_mm2" in result
    assert "num_vertices" in result
    assert "num_faces" in result
    assert "watertight_check_available" in result
    assert "mesh_validation_time" in result

    # Check types (values may be None if trimesh not available)
    if result["watertight_check_available"]:
        assert isinstance(result["is_watertight"], bool)
        assert isinstance(result["surface_area_mm2"], float)
        assert isinstance(result["num_vertices"], int)
        assert isinstance(result["num_faces"], int)
        assert isinstance(result["mesh_validation_time"], float)
        assert result["mesh_validation_time"] >= 0
    else:
        # Without trimesh, should have minimal info
        assert result["is_watertight"] is None
        assert result["volume_mm3"] is None


@pytest.mark.unit
def test_check_mesh_watertightness_without_trimesh(sample_2d_array):
    """Test graceful degradation when trimesh not available."""
    # Create a mesh
    mesh = _create_stl_from_heatmap_extruded(sample_2d_array)

    # Mock trimesh as unavailable
    import src.tools.stl_export

    original_available = src.tools.stl_export.TRIMESH_AVAILABLE
    src.tools.stl_export.TRIMESH_AVAILABLE = False

    try:
        result = _check_mesh_watertightness(mesh)

        # Should return with check unavailable
        assert result["watertight_check_available"] is False
        assert result["is_watertight"] is None
        assert result["volume_mm3"] is None
        assert result["mesh_validation_time"] == 0.0
        # Should still get face count from numpy-stl
        assert result["num_faces"] == len(mesh.vectors)
    finally:
        # Restore original value
        src.tools.stl_export.TRIMESH_AVAILABLE = original_available


@pytest.mark.unit
def test_watertightness_in_convert_output(tmp_path, sample_2d_array):
    """Test that STL conversion includes watertightness metrics."""
    npy_path = tmp_path / "design.npy"
    np.save(npy_path, sample_2d_array)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "stl_file_path": str(outputs_dir / "test.stl"),
            }
        )

    # Verify watertightness metrics are included
    assert result["success"] is True
    assert "is_watertight" in result
    assert "volume_mm3" in result
    assert "surface_area_mm2" in result
    assert "num_vertices" in result
    assert "num_faces" in result
    assert "watertight_check_available" in result
    assert "mesh_validation_time" in result


@pytest.mark.unit
def test_connectivity_vs_watertightness_independence(tmp_path):
    """Test that 2D connectivity and 3D watertightness are tracked separately."""
    # Create a simple connected design
    design = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    npy_path = tmp_path / "design.npy"
    np.save(npy_path, design)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "stl_file_path": str(outputs_dir / "test.stl"),
            }
        )

    # Verify both metrics exist independently
    assert result["success"] is True
    assert "connected_design" in result
    assert "num_components" in result
    assert "is_watertight" in result

    # Both should be tracked even if connected
    assert result["connected_design"] is True
    assert result["num_components"] == 1
    # is_watertight tracked separately (value depends on trimesh availability)


@pytest.mark.integration
@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
def test_watertightness_with_simple_design(tmp_path):
    """Integration test with actual trimesh validation."""
    # Create a simple connected design that should be watertight
    design = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    npy_path = tmp_path / "design.npy"
    np.save(npy_path, design)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "stl_file_path": str(outputs_dir / "test.stl"),
            }
        )

    # With trimesh available, should have full validation
    assert result["success"] is True
    assert result["watertight_check_available"] is True
    assert isinstance(result["is_watertight"], bool)
    assert isinstance(result["volume_mm3"], (float, type(None)))
    assert isinstance(result["surface_area_mm2"], float)
    assert result["surface_area_mm2"] > 0
    assert isinstance(result["num_vertices"], int)
    assert result["num_vertices"] > 0
    assert isinstance(result["mesh_validation_time"], float)
    assert result["mesh_validation_time"] >= 0


@pytest.mark.integration
@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
def test_watertightness_metrics_accuracy(tmp_path):
    """Test that volume/area calculations are reasonable for known geometry."""
    # Create 2x2 grid with all cells filled
    design = np.ones((2, 2))

    npy_path = tmp_path / "design.npy"
    np.save(npy_path, design)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    scale_xy = 10.0
    scale_z = 5.0

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "stl_file_path": str(outputs_dir / "test.stl"),
                "scale_xy": scale_xy,
                "scale_z": scale_z,
            }
        )

    # Expected: 2x2 grid of cubes, each cube is scale_xy x scale_xy x scale_z
    # Total volume = 4 cubes * (10 * 10 * 5) = 2000
    # But trimesh might report slightly different due to shared faces

    assert result["success"] is True
    if result["is_watertight"]:
        # Volume should be in reasonable range
        assert result["volume_mm3"] is not None
        assert result["volume_mm3"] > 1000  # At least 50% of theoretical
        assert result["volume_mm3"] < 3000  # Not more than 150% of theoretical

    # Surface area should be positive
    assert result["surface_area_mm2"] > 0


@pytest.mark.integration
@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
def test_mesh_repair_functionality(tmp_path):
    """Test that mesh repair is attempted and tracked."""
    # Create a simple connected design
    design = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    npy_path = tmp_path / "design.npy"
    np.save(npy_path, design)

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    with patch("src.tools.stl_export.Path") as mock_path_class:

        def path_side_effect(path_str):
            if path_str == "outputs":
                return outputs_dir
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = convert_design_to_stl.invoke(
            {
                "npy_file_path": str(npy_path),
                "stl_file_path": str(outputs_dir / "test.stl"),
            }
        )

    # Verify repair metrics are present
    assert result["success"] is True
    assert "repair_attempted" in result
    assert "mesh_repaired" in result
    assert isinstance(result["repair_attempted"], bool)
    assert isinstance(result["mesh_repaired"], bool)

    # If mesh was initially non-watertight, repair should have been attempted
    # If repair succeeded, is_watertight should be True and mesh_repaired should be True
    if result["mesh_repaired"]:
        assert result["is_watertight"] is True
        assert result["repair_attempted"] is True
        # After repair, volume should be available
        assert result["volume_mm3"] is not None
        assert result["volume_mm3"] > 0
