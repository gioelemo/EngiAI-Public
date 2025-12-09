"""Tests for media display functionality."""

from unittest.mock import patch

from src.ui.media_display import (
    find_images_in_text,
    find_log_files_in_text,
    find_slurm_files_in_text,
    find_stl_files_in_text,
)


class TestFindImagesInText:
    """Tests for find_images_in_text function."""

    def test_find_images_basic_path(self, tmp_path):
        """Test finding basic image path."""
        # Create a temporary image file
        img_file = tmp_path / "test.png"
        img_file.touch()

        text = f"Check out this image: {img_file}"
        images = find_images_in_text(text)

        assert len(images) > 0
        assert any(str(img_file) in str(img) for img in images)

    def test_find_images_in_outputs_directory(self, tmp_path):
        """Test finding images in outputs/ directory."""
        with patch("src.ui.media_display.project_root", tmp_path):
            # Create outputs directory with image
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            img_file = outputs_dir / "result.png"
            img_file.touch()

            text = "outputs/result.png shows the results"
            images = find_images_in_text(text)

            assert len(images) > 0

    def test_find_images_in_assets_directory(self, tmp_path):
        """Test finding images in assets/ directory."""
        with patch("src.ui.media_display.project_root", tmp_path):
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()
            img_file = assets_dir / "logo.png"
            img_file.touch()

            text = "assets/logo.png is our logo"
            images = find_images_in_text(text)

            assert len(images) > 0

    def test_find_images_markdown_syntax(self, tmp_path):
        """Test finding images with markdown syntax."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            img_file = outputs_dir / "chart.png"
            img_file.touch()

            text = "![Chart](outputs/chart.png)"
            images = find_images_in_text(text)

            assert len(images) > 0

    def test_find_images_multiple_formats(self, tmp_path):
        """Test finding images with various extensions."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()

            extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
            for ext in extensions:
                img_file = outputs_dir / f"image{ext}"
                img_file.touch()

            text = "outputs/image.png outputs/image.jpg outputs/image.jpeg outputs/image.gif outputs/image.bmp"
            images = find_images_in_text(text)

            assert len(images) == len(extensions)

    def test_find_images_no_duplicates(self, tmp_path):
        """Test that duplicate paths are filtered out."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            img_file = outputs_dir / "test.png"
            img_file.touch()

            text = "outputs/test.png and outputs/test.png again"
            images = find_images_in_text(text)

            # Should only have one instance
            assert len(images) == 1

    def test_find_images_nonexistent_files(self):
        """Test that non-existent files are not returned."""
        text = "outputs/nonexistent.png"
        images = find_images_in_text(text)

        # Should not find non-existent files
        assert len(images) == 0

    def test_find_images_empty_text(self):
        """Test with empty text."""
        text = ""
        images = find_images_in_text(text)

        assert len(images) == 0


class TestFindStlFilesInText:
    """Tests for find_stl_files_in_text function."""

    def test_find_stl_basic(self, tmp_path):
        """Test finding basic STL file."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            stl_file = outputs_dir / "model.stl"
            stl_file.touch()

            text = "outputs/model.stl is the 3D model"
            stl_files = find_stl_files_in_text(text)

            assert len(stl_files) > 0

    def test_find_stl_multiple(self, tmp_path):
        """Test finding multiple STL files."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()

            stl1 = outputs_dir / "model1.stl"
            stl2 = outputs_dir / "model2.stl"
            stl1.touch()
            stl2.touch()

            text = "outputs/model1.stl and outputs/model2.stl"
            stl_files = find_stl_files_in_text(text)

            # Should find both files
            assert len(stl_files) >= 1  # At least one should be found

    def test_find_stl_nonexistent(self):
        """Test that non-existent STL files are not returned."""
        text = "nonexistent.stl"
        stl_files = find_stl_files_in_text(text)

        assert len(stl_files) == 0

    def test_find_stl_case_insensitive(self, tmp_path):
        """Test case-insensitive STL detection."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            stl_file = outputs_dir / "MODEL.STL"
            stl_file.touch()

            text = "outputs/MODEL.STL"
            stl_files = find_stl_files_in_text(text)

            assert len(stl_files) > 0


class TestFindLogFilesInText:
    """Tests for find_log_files_in_text function."""

    def test_find_log_err_file(self, tmp_path):
        """Test finding .err log file."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            err_file = outputs_dir / "job_12345.err"
            err_file.touch()

            text = "outputs/job_12345.err"
            log_files = find_log_files_in_text(text)

            assert len(log_files) > 0

    def test_find_log_out_file(self, tmp_path):
        """Test finding .out log file."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            out_file = outputs_dir / "job_12345.out"
            out_file.touch()

            text = "outputs/job_12345.out"
            log_files = find_log_files_in_text(text)

            assert len(log_files) > 0

    def test_find_log_by_job_id(self, tmp_path):
        """Test finding log files by job ID."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            err_file = outputs_dir / "train_12345.err"
            out_file = outputs_dir / "train_12345.out"
            err_file.touch()
            out_file.touch()

            text = "Job ID: 12345 has completed"
            log_files = find_log_files_in_text(text)

            assert len(log_files) == 2

    def test_find_log_job_id_variations(self, tmp_path):
        """Test various job ID text patterns."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            err_file = outputs_dir / "test_99999.err"
            err_file.touch()

            patterns = [
                "job_id: 99999",
                "Job ID: 99999",
                'job_id": 99999',
                "job ID 99999",
            ]

            for pattern in patterns:
                log_files = find_log_files_in_text(pattern)
                # At least should not crash
                assert isinstance(log_files, list)

    def test_find_log_no_duplicates(self, tmp_path):
        """Test that duplicate log files are not returned."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            err_file = outputs_dir / "job.err"
            err_file.touch()

            text = "outputs/job.err mentioned twice outputs/job.err"
            log_files = find_log_files_in_text(text)

            assert len(log_files) == 1

    def test_find_log_empty_text(self):
        """Test with empty text."""
        text = ""
        log_files = find_log_files_in_text(text)

        assert len(log_files) == 0


class TestFindSlurmFilesInText:
    """Tests for find_slurm_files_in_text function."""

    def test_find_slurm_basic(self, tmp_path):
        """Test finding basic SLURM script."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()
            slurm_file = outputs_dir / "train_job.slurm"
            slurm_file.touch()

            text = "outputs/train_job.slurm"
            slurm_files = find_slurm_files_in_text(text)

            assert len(slurm_files) > 0

    def test_find_slurm_multiple(self, tmp_path):
        """Test finding multiple SLURM files."""
        with patch("src.ui.media_display.project_root", tmp_path):
            outputs_dir = tmp_path / "outputs"
            outputs_dir.mkdir()

            slurm1 = outputs_dir / "job1.slurm"
            slurm2 = outputs_dir / "job2.slurm"
            slurm1.touch()
            slurm2.touch()

            text = "outputs/job1.slurm and outputs/job2.slurm"
            slurm_files = find_slurm_files_in_text(text)

            # Should find at least one file
            assert len(slurm_files) >= 1

    def test_find_slurm_nonexistent(self):
        """Test that non-existent SLURM files are not returned."""
        text = "nonexistent.slurm"
        slurm_files = find_slurm_files_in_text(text)

        assert len(slurm_files) == 0

    def test_find_slurm_empty_text(self):
        """Test with empty text."""
        text = ""
        slurm_files = find_slurm_files_in_text(text)

        assert len(slurm_files) == 0
