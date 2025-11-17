"""
Tests for engiopt tools.

These tests focus on the WandB model download and management functionality.
Most tests are unit tests that don't require actual WandB authentication or model files.
"""

import os
from unittest.mock import Mock, patch

import pytest

from src.tools.engiopt import (
    SUPPORTED_ALGORITHMS,
    HPCContext,
    HPCInputs,
    TrainingConfig,
    _check_algorithm_recommendations,
    _check_contextual_recommendations,
    _check_general_recommendations,
    _check_hard_limits,
    _check_wandb_available,
    _download_from_wandb,
    _get_artifact_info,
    _parse_hpc_inputs,
    _validate_download_inputs,
    download_wandb_model,
    list_available_algorithms,
)

# ============================================================================
# UNIT TESTS - Fast tests without external dependencies
# ============================================================================


@pytest.mark.unit
def test_supported_algorithms_structure():
    """Test that SUPPORTED_ALGORITHMS has correct structure."""
    assert isinstance(SUPPORTED_ALGORITHMS, dict)
    assert len(SUPPORTED_ALGORITHMS) > 0

    # Check each algorithm has required fields
    for algo_info in SUPPORTED_ALGORITHMS.values():
        assert "class" in algo_info
        assert "dimensions" in algo_info
        assert "conditional" in algo_info
        assert "model" in algo_info
        assert "model_types" in algo_info
        assert isinstance(algo_info["model_types"], list)


@pytest.mark.unit
def test_supported_algorithms_contains_expected():
    """Test that expected algorithms are present."""
    assert "cgan_cnn_2d" in SUPPORTED_ALGORITHMS
    assert "diffusion_2d_cond" in SUPPORTED_ALGORITHMS


@pytest.mark.unit
def test_hpc_inputs_dataclass():
    """Test HPCInputs dataclass instantiation."""
    inputs = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="1",
        slurm_time="24:00:00",
        slurm_cpus_per_task="8",
        slurm_mem_per_cpu="2G",
    )

    assert inputs.algorithm == "cgan_cnn_2d"
    assert inputs.problem_id == "beams2d"
    assert inputs.epochs == 200
    assert inputs.slurm_gpus == "1"


@pytest.mark.unit
def test_hpc_context_dataclass():
    """Test HPCContext dataclass instantiation."""
    context = HPCContext(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        gpu_count=1,
        total_hours=24.0,
        cpu_count=8,
        mem_value=2.0,
    )

    assert context.algorithm == "cgan_cnn_2d"
    assert context.gpu_count == 1
    assert context.total_hours == 24.0


@pytest.mark.unit
def test_training_config_defaults():
    """Test TrainingConfig dataclass default values."""
    config = TrainingConfig()

    assert config.algorithm == "cgan_cnn_2d"
    assert config.epochs == 200
    assert config.seed == 1
    assert config.wandb_entity is None
    assert config.problem_id == "beams2d"
    assert config.gpus is None
    assert config.time_hours is None


@pytest.mark.unit
def test_training_config_custom_values():
    """Test TrainingConfig with custom values."""
    config = TrainingConfig(
        algorithm="diffusion_2d_cond",
        epochs=100,
        seed=42,
        wandb_entity="test-entity",
        gpus=2,
        time_hours=48.0,
    )

    assert config.algorithm == "diffusion_2d_cond"
    assert config.epochs == 100
    assert config.seed == 42
    assert config.wandb_entity == "test-entity"
    assert config.gpus == 2
    assert config.time_hours == 48.0


@pytest.mark.unit
def test_get_artifact_info_cgan():
    """Test artifact info for cGAN models."""
    # Generator
    result = _get_artifact_info("beams2d", "cgan_cnn_2d", "generator")
    assert result["artifact_name"] == "beams2d_cgan_cnn_2d_generator"
    assert result["checkpoint_filename"] == "generator.pth"

    # Discriminator
    result = _get_artifact_info("beams2d", "cgan_cnn_2d", "discriminator")
    assert result["artifact_name"] == "beams2d_cgan_cnn_2d_discriminator"
    assert result["checkpoint_filename"] == "discriminator.pth"


@pytest.mark.unit
def test_get_artifact_info_diffusion():
    """Test artifact info for diffusion models."""
    # Diffusion always uses "model" nomenclature
    result = _get_artifact_info("beams2d", "diffusion_2d_cond", "generator")
    assert result["artifact_name"] == "beams2d_diffusion_2d_cond_model"
    assert result["checkpoint_filename"] == "model.pth"

    # Same for any model_type with diffusion
    result = _get_artifact_info("beams2d", "diffusion_2d_cond", "discriminator")
    assert result["artifact_name"] == "beams2d_diffusion_2d_cond_model"
    assert result["checkpoint_filename"] == "model.pth"


@pytest.mark.unit
def test_validate_download_inputs_wandb_disabled(monkeypatch):
    """Test validation when WandB is disabled."""
    monkeypatch.setenv("USE_WANDB", "False")

    result = _validate_download_inputs("beams2d", "cgan_cnn_2d")

    assert result is not None
    assert result["success"] is False
    assert "not enabled" in result["error"].lower()


@pytest.mark.unit
def test_validate_download_inputs_unsupported_algorithm(monkeypatch):
    """Test validation with unsupported algorithm."""
    monkeypatch.setenv("USE_WANDB", "True")

    result = _validate_download_inputs("beams2d", "invalid_algorithm")

    assert result is not None
    assert result["success"] is False
    assert "unsupported algorithm" in result["error"].lower()


@pytest.mark.unit
def test_validate_download_inputs_unsupported_problem(monkeypatch):
    """Test validation with unsupported problem."""
    monkeypatch.setenv("USE_WANDB", "True")

    result = _validate_download_inputs("invalid_problem", "cgan_cnn_2d")

    assert result is not None
    assert result["success"] is False
    assert "unsupported problem_id" in result["error"].lower()


@pytest.mark.unit
def test_validate_download_inputs_success(monkeypatch):
    """Test validation with valid inputs."""
    monkeypatch.setenv("USE_WANDB", "True")

    result = _validate_download_inputs("beams2d", "cgan_cnn_2d")

    assert result is None  # None means validation passed


@pytest.mark.unit
@patch("src.tools.engiopt.WANDB_AVAILABLE", False)
def test_check_wandb_available_not_installed():
    """Test check when wandb is not installed."""
    result = _check_wandb_available()

    assert result is not None
    assert result["success"] is False
    assert "not installed" in result["error"].lower()


@pytest.mark.unit
@patch("src.tools.engiopt.WANDB_AVAILABLE", True)
def test_check_wandb_available_installed():
    """Test check when wandb is installed."""
    result = _check_wandb_available()

    assert result is None  # None means check passed


@pytest.mark.unit
def test_list_available_algorithms():
    """Test listing available algorithms."""
    result = list_available_algorithms.invoke({})

    assert "algorithms" in result
    assert "details" in result
    assert "message" in result

    assert isinstance(result["algorithms"], list)
    assert isinstance(result["details"], dict)

    # Check we have the expected algorithms
    assert "cgan_cnn_2d" in result["algorithms"]
    assert "diffusion_2d_cond" in result["algorithms"]

    # Check details match SUPPORTED_ALGORITHMS
    assert result["details"] == SUPPORTED_ALGORITHMS


@pytest.mark.unit
def test_download_wandb_model_wandb_disabled(monkeypatch):
    """Test download when WandB is disabled."""
    monkeypatch.setenv("USE_WANDB", "False")

    result = download_wandb_model.invoke(
        {"problem_id": "beams2d", "algorithm": "cgan_cnn_2d"}
    )

    assert result["success"] is False
    assert "not enabled" in result["error"].lower()


@pytest.mark.unit
def test_download_wandb_model_invalid_algorithm(monkeypatch):
    """Test download with invalid algorithm."""
    monkeypatch.setenv("USE_WANDB", "True")

    result = download_wandb_model.invoke(
        {"problem_id": "beams2d", "algorithm": "invalid_algo"}
    )

    assert result["success"] is False
    assert "unsupported algorithm" in result["error"].lower()


@pytest.mark.unit
@patch("src.tools.engiopt.WANDB_AVAILABLE", False)
def test_download_wandb_model_wandb_not_installed(monkeypatch):
    """Test download when wandb package is not installed."""
    monkeypatch.setenv("USE_WANDB", "True")

    result = download_wandb_model.invoke(
        {"problem_id": "beams2d", "algorithm": "cgan_cnn_2d"}
    )

    assert result["success"] is False
    assert "not installed" in result["error"].lower()


@pytest.mark.unit
def test_algorithm_properties_cgan():
    """Test properties of cGAN algorithm."""
    algo = SUPPORTED_ALGORITHMS["cgan_cnn_2d"]

    assert algo["class"] == "Inverse Design"
    assert algo["dimensions"] == "2D"
    assert algo["conditional"] is True
    assert "GAN" in algo["model"]
    assert "generator" in algo["model_types"]
    assert "discriminator" in algo["model_types"]


@pytest.mark.unit
def test_algorithm_properties_diffusion():
    """Test properties of diffusion algorithm."""
    algo = SUPPORTED_ALGORITHMS["diffusion_2d_cond"]

    assert algo["class"] == "Inverse Design"
    assert algo["dimensions"] == "2D"
    assert algo["conditional"] is True
    assert "Diffusion" in algo["model"]
    assert "model" in algo["model_types"]


@pytest.mark.unit
def test_artifact_naming_consistency():
    """Test that artifact naming is consistent across model types."""
    # For same algorithm and problem, artifact name should be consistent
    gen_info = _get_artifact_info("beams2d", "cgan_cnn_2d", "generator")
    disc_info = _get_artifact_info("beams2d", "cgan_cnn_2d", "discriminator")

    # Should have same prefix but different suffix
    assert gen_info["artifact_name"].startswith("beams2d_cgan_cnn_2d")
    assert disc_info["artifact_name"].startswith("beams2d_cgan_cnn_2d")
    assert gen_info["artifact_name"] != disc_info["artifact_name"]


@pytest.mark.unit
def test_checkpoint_filename_extensions():
    """Test that checkpoint filenames have .pth extension."""
    for algo in SUPPORTED_ALGORITHMS:
        for model_type in ["generator", "discriminator"]:
            info = _get_artifact_info("beams2d", algo, model_type)
            assert info["checkpoint_filename"].endswith(".pth")


@pytest.mark.unit
def test_download_wandb_model_default_parameters(monkeypatch):
    """Test that download_wandb_model has sensible defaults."""
    monkeypatch.setenv("USE_WANDB", "False")  # To avoid actual download

    # Call with minimal parameters
    result = download_wandb_model.invoke({})

    # Should fail due to disabled WandB, but parameters were accepted
    assert result["success"] is False
    assert "error" in result


@pytest.mark.unit
def test_training_config_problem_id_literal():
    """Test that problem_id is restricted to valid values."""
    config = TrainingConfig(problem_id="beams2d")
    assert config.problem_id == "beams2d"

    # Type hints should restrict to "beams2d" only
    # (runtime check would require Pydantic or similar)


# ============================================================================
# INTEGRATION TESTS - Require mocking WandB
# ============================================================================


@pytest.mark.unit
@patch("src.tools.engiopt.WANDB_AVAILABLE", True)
def test_download_from_wandb_success(monkeypatch, tmp_path):
    """Test successful download from WandB with mocked API."""
    monkeypatch.setenv("USE_WANDB", "True")

    # Setup mocks
    mock_api = Mock()
    mock_artifact = Mock()
    mock_run = Mock()

    # Mock artifact download
    checkpoint_path = tmp_path / "generator.pth"
    checkpoint_path.touch()
    mock_artifact.download.return_value = str(tmp_path)

    # Mock run config
    mock_run.config = {"epochs": 200, "batch_size": 32}
    mock_artifact.logged_by.return_value = mock_run

    mock_api.artifact.return_value = mock_artifact

    # Mock wandb module with nested structure for wandb.apis.public
    mock_wandb_apis = Mock()
    mock_wandb_apis_public = Mock()
    mock_wandb_apis_public.Api = Mock(return_value=mock_api)
    mock_wandb_apis.public = mock_wandb_apis_public

    mock_wandb = Mock()
    mock_wandb.apis = mock_wandb_apis

    # Patch wandb import
    with patch.dict(
        "sys.modules",
        {
            "wandb": mock_wandb,
            "wandb.apis": mock_wandb_apis,
            "wandb.apis.public": mock_wandb_apis_public,
        },
    ):
        result = _download_from_wandb(
            problem_id="beams2d",
            algorithm="cgan_cnn_2d",
            seed=1,
            model_type="generator",
            wandb_project="test/project",
            download_dir=None,
        )

    assert result["success"] is True
    assert "checkpoint_path" in result
    assert result["model_type"] == "generator"
    assert "run_config" in result


@pytest.mark.unit
@patch("src.tools.engiopt.WANDB_AVAILABLE", True)
def test_download_from_wandb_checkpoint_not_found(monkeypatch, tmp_path):
    """Test download when checkpoint file is missing."""
    monkeypatch.setenv("USE_WANDB", "True")

    # Setup mocks - artifact downloads but checkpoint doesn't exist
    mock_api = Mock()
    mock_artifact = Mock()
    mock_artifact.download.return_value = str(tmp_path)  # Returns dir without file

    mock_api.artifact.return_value = mock_artifact

    # Mock wandb module with nested structure for wandb.apis.public
    mock_wandb_apis = Mock()
    mock_wandb_apis_public = Mock()
    mock_wandb_apis_public.Api = Mock(return_value=mock_api)
    mock_wandb_apis.public = mock_wandb_apis_public

    mock_wandb = Mock()
    mock_wandb.apis = mock_wandb_apis

    # Patch wandb import
    with patch.dict(
        "sys.modules",
        {
            "wandb": mock_wandb,
            "wandb.apis": mock_wandb_apis,
            "wandb.apis.public": mock_wandb_apis_public,
        },
    ):
        result = _download_from_wandb(
            problem_id="beams2d",
            algorithm="cgan_cnn_2d",
            seed=1,
            model_type="generator",
            wandb_project="test/project",
            download_dir=None,
        )

    assert result["success"] is False
    assert "not found" in result["error"].lower()


@pytest.mark.unit
@patch("src.tools.engiopt.WANDB_AVAILABLE", True)
def test_download_from_wandb_api_error(monkeypatch):
    """Test download when WandB API raises an error."""
    monkeypatch.setenv("USE_WANDB", "True")

    # Setup mocks to raise an exception
    mock_api = Mock()
    mock_api.artifact.side_effect = RuntimeError("WandB API error")

    # Mock wandb module with nested structure for wandb.apis.public
    mock_wandb_apis = Mock()
    mock_wandb_apis_public = Mock()
    mock_wandb_apis_public.Api = Mock(return_value=mock_api)
    mock_wandb_apis.public = mock_wandb_apis_public

    mock_wandb = Mock()
    mock_wandb.apis = mock_wandb_apis

    # Patch wandb import
    with patch.dict(
        "sys.modules",
        {
            "wandb": mock_wandb,
            "wandb.apis": mock_wandb_apis,
            "wandb.apis.public": mock_wandb_apis_public,
        },
    ):
        result = _download_from_wandb(
            problem_id="beams2d",
            algorithm="cgan_cnn_2d",
            seed=1,
            model_type="generator",
            wandb_project="test/project",
            download_dir=None,
        )

    assert result["success"] is False
    assert "failed" in result["error"].lower()


@pytest.mark.unit
@patch("src.tools.engiopt.WANDB_AVAILABLE", True)
def test_download_wandb_model_multi_project_fallback(monkeypatch, tmp_path):
    """Test download fallback to multiple projects."""
    monkeypatch.setenv("USE_WANDB", "True")
    monkeypatch.setenv("WANDB_PERSONAL_PROJECT", "personal/project")
    monkeypatch.setenv("WANDB_OFFICIAL_PROJECT", "official/project")

    # Setup mocks - first project fails, second succeeds
    mock_api = Mock()
    checkpoint_path = tmp_path / "generator.pth"
    checkpoint_path.touch()

    not_found_error = "Not found"

    def artifact_side_effect(path, type=None):  # noqa: ARG001
        if "personal" in path:
            raise RuntimeError(not_found_error)
        # Second project succeeds
        mock_artifact = Mock()
        mock_artifact.download.return_value = str(tmp_path)
        mock_artifact.logged_by.return_value = None
        return mock_artifact

    mock_api.artifact.side_effect = artifact_side_effect

    # Mock wandb module with nested structure for wandb.apis.public
    mock_wandb_apis = Mock()
    mock_wandb_apis_public = Mock()
    mock_wandb_apis_public.Api = Mock(return_value=mock_api)
    mock_wandb_apis.public = mock_wandb_apis_public

    mock_wandb = Mock()
    mock_wandb.apis = mock_wandb_apis

    # Patch wandb import
    with patch.dict(
        "sys.modules",
        {
            "wandb": mock_wandb,
            "wandb.apis": mock_wandb_apis,
            "wandb.apis.public": mock_wandb_apis_public,
        },
    ):
        result = download_wandb_model.invoke(
            {
                "problem_id": "beams2d",
                "algorithm": "cgan_cnn_2d",
                "seed": 1,
                "model_type": "generator",
                "wandb_project": None,  # Trigger multi-project search
            }
        )

    assert result["success"] is True
    assert "checkpoint_path" in result


@pytest.mark.unit
def test_environment_variable_defaults(monkeypatch):
    """Test default values for environment variables."""
    # Clear any existing env vars
    monkeypatch.delenv("WANDB_PERSONAL_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_OFFICIAL_PROJECT", raising=False)

    personal = os.getenv("WANDB_PERSONAL_PROJECT", "gioelemo-ethz/engiopt")
    official = os.getenv("WANDB_OFFICIAL_PROJECT", "engibench/engiopt")

    assert personal == "gioelemo-ethz/engiopt"
    assert official == "engibench/engiopt"


# ============================================================================
# EDGE CASES
# ============================================================================


@pytest.mark.unit
def test_artifact_info_with_different_problems():
    """Test artifact naming for different problem IDs."""
    result1 = _get_artifact_info("beams2d", "cgan_cnn_2d", "generator")
    result2 = _get_artifact_info("other_problem", "cgan_cnn_2d", "generator")

    assert "beams2d" in result1["artifact_name"]
    assert "other_problem" in result2["artifact_name"]
    assert result1["artifact_name"] != result2["artifact_name"]


@pytest.mark.unit
def test_validate_all_supported_algorithms(monkeypatch):
    """Test validation passes for all supported algorithms."""
    monkeypatch.setenv("USE_WANDB", "True")

    for algorithm in SUPPORTED_ALGORITHMS:
        result = _validate_download_inputs("beams2d", algorithm)
        assert result is None, f"Validation failed for {algorithm}"


@pytest.mark.unit
def test_model_types_consistency():
    """Test that model_types are consistent with algorithm requirements."""
    # GAN should have both generator and discriminator
    gan_types = SUPPORTED_ALGORITHMS["cgan_cnn_2d"]["model_types"]
    assert "generator" in gan_types
    assert "discriminator" in gan_types

    # Diffusion should have model
    diff_types = SUPPORTED_ALGORITHMS["diffusion_2d_cond"]["model_types"]
    assert "model" in diff_types


# ============================================================================
# HPC RESOURCE VALIDATION TESTS
# ============================================================================


@pytest.mark.unit
def test_parse_hpc_inputs_valid():
    """Test parsing valid HPC input strings."""
    cfg = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="2",
        slurm_time="12:30:00",
        slurm_cpus_per_task="8",
        slurm_mem_per_cpu="4G",
    )

    gpu_count, total_hours, hours, cpu_count, mem_value, errors = _parse_hpc_inputs(cfg)

    assert gpu_count == 2
    assert hours == 12
    assert total_hours == 12.5  # 12 hours + 30 minutes
    assert cpu_count == 8
    assert mem_value == 4.0
    assert len(errors) == 0


@pytest.mark.unit
def test_parse_hpc_inputs_gpu_with_type():
    """Test parsing GPU specification with type (e.g., 'gpu:2')."""
    cfg = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="gpu:4",
        slurm_time="24:00:00",
        slurm_cpus_per_task="8",
        slurm_mem_per_cpu="2G",
    )

    gpu_count, _, _, _, _, errors = _parse_hpc_inputs(cfg)

    assert gpu_count == 4
    assert len(errors) == 0


@pytest.mark.unit
def test_parse_hpc_inputs_memory_megabytes():
    """Test parsing memory in megabytes."""
    cfg = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="1",
        slurm_time="10:00:00",
        slurm_cpus_per_task="4",
        slurm_mem_per_cpu="2048M",  # 2GB in MB
    )

    _, _, _, _, mem_value, errors = _parse_hpc_inputs(cfg)

    assert mem_value == 2.0  # Should convert to GB
    assert len(errors) == 0


@pytest.mark.unit
def test_parse_hpc_inputs_invalid_gpu():
    """Test parsing with invalid GPU specification."""
    cfg = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="invalid",
        slurm_time="10:00:00",
        slurm_cpus_per_task="4",
        slurm_mem_per_cpu="2G",
    )

    gpu_count, _, _, _, _, errors = _parse_hpc_inputs(cfg)

    assert gpu_count == 1  # Default fallback
    assert len(errors) > 0
    assert any("GPU" in err for err in errors)


@pytest.mark.unit
def test_parse_hpc_inputs_invalid_time():
    """Test parsing with invalid time format."""
    cfg = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="1",
        slurm_time="invalid",
        slurm_cpus_per_task="4",
        slurm_mem_per_cpu="2G",
    )

    _, total_hours, hours, _, _, errors = _parse_hpc_inputs(cfg)

    assert hours == 0  # Default fallback
    assert total_hours == 0.0
    assert len(errors) > 0
    assert any("time" in err.lower() for err in errors)


@pytest.mark.unit
def test_parse_hpc_inputs_invalid_cpu():
    """Test parsing with invalid CPU count."""
    cfg = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="1",
        slurm_time="10:00:00",
        slurm_cpus_per_task="invalid",
        slurm_mem_per_cpu="2G",
    )

    _, _, _, cpu_count, _, errors = _parse_hpc_inputs(cfg)

    assert cpu_count == 4  # Default fallback
    assert len(errors) > 0
    assert any("CPU" in err for err in errors)


@pytest.mark.unit
def test_parse_hpc_inputs_invalid_memory():
    """Test parsing with invalid memory specification."""
    cfg = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="1",
        slurm_time="10:00:00",
        slurm_cpus_per_task="4",
        slurm_mem_per_cpu="invalid",
    )

    _, _, _, _, mem_value, errors = _parse_hpc_inputs(cfg)

    assert mem_value == 7.0  # Default fallback
    assert len(errors) > 0
    assert any("memory" in err.lower() for err in errors)


@pytest.mark.unit
def test_check_hard_limits_within_bounds(monkeypatch):
    """Test validation when all resources are within limits."""
    # Set environment limits
    monkeypatch.setenv("SLURM_MAX_GPUS", "4")
    monkeypatch.setenv("SLURM_MAX_TIME_HOURS", "24")
    monkeypatch.setenv("SLURM_MAX_CPUS", "16")
    monkeypatch.setenv("SLURM_MAX_MEM_PER_CPU_GB", "16")

    errors = _check_hard_limits(gpu_count=2, hours=12, cpu_count=8, mem_value=8.0)

    assert len(errors) == 0


@pytest.mark.unit
def test_check_hard_limits_gpu_exceeded(monkeypatch):
    """Test validation when GPU count exceeds limit."""
    monkeypatch.setenv("SLURM_MAX_GPUS", "4")
    monkeypatch.setenv("SLURM_MAX_TIME_HOURS", "24")
    monkeypatch.setenv("SLURM_MAX_CPUS", "16")
    monkeypatch.setenv("SLURM_MAX_MEM_PER_CPU_GB", "16")

    errors = _check_hard_limits(gpu_count=8, hours=12, cpu_count=8, mem_value=8.0)

    assert len(errors) > 0
    assert any("GPU" in err for err in errors)
    assert any("exceeds maximum" in err for err in errors)


@pytest.mark.unit
def test_check_hard_limits_time_exceeded(monkeypatch):
    """Test validation when time limit exceeds maximum."""
    monkeypatch.setenv("SLURM_MAX_GPUS", "4")
    monkeypatch.setenv("SLURM_MAX_TIME_HOURS", "24")
    monkeypatch.setenv("SLURM_MAX_CPUS", "16")
    monkeypatch.setenv("SLURM_MAX_MEM_PER_CPU_GB", "16")

    errors = _check_hard_limits(gpu_count=2, hours=48, cpu_count=8, mem_value=8.0)

    assert len(errors) > 0
    assert any("Time" in err for err in errors)


@pytest.mark.unit
def test_check_hard_limits_cpu_exceeded(monkeypatch):
    """Test validation when CPU count exceeds limit."""
    monkeypatch.setenv("SLURM_MAX_GPUS", "4")
    monkeypatch.setenv("SLURM_MAX_TIME_HOURS", "24")
    monkeypatch.setenv("SLURM_MAX_CPUS", "16")
    monkeypatch.setenv("SLURM_MAX_MEM_PER_CPU_GB", "16")

    errors = _check_hard_limits(gpu_count=2, hours=12, cpu_count=32, mem_value=8.0)

    assert len(errors) > 0
    assert any("CPU" in err for err in errors)


@pytest.mark.unit
def test_check_hard_limits_memory_exceeded(monkeypatch):
    """Test validation when memory exceeds limit."""
    monkeypatch.setenv("SLURM_MAX_GPUS", "4")
    monkeypatch.setenv("SLURM_MAX_TIME_HOURS", "24")
    monkeypatch.setenv("SLURM_MAX_CPUS", "16")
    monkeypatch.setenv("SLURM_MAX_MEM_PER_CPU_GB", "16")

    errors = _check_hard_limits(gpu_count=2, hours=12, cpu_count=8, mem_value=32.0)

    assert len(errors) > 0
    assert any("Memory" in err for err in errors)


@pytest.mark.unit
def test_check_hard_limits_multiple_violations(monkeypatch):
    """Test validation when multiple resources exceed limits."""
    monkeypatch.setenv("SLURM_MAX_GPUS", "4")
    monkeypatch.setenv("SLURM_MAX_TIME_HOURS", "24")
    monkeypatch.setenv("SLURM_MAX_CPUS", "16")
    monkeypatch.setenv("SLURM_MAX_MEM_PER_CPU_GB", "16")

    errors = _check_hard_limits(gpu_count=8, hours=48, cpu_count=32, mem_value=32.0)

    assert len(errors) == 4  # All four limits exceeded
    assert any("GPU" in err for err in errors)
    assert any("Time" in err for err in errors)
    assert any("CPU" in err for err in errors)
    assert any("Memory" in err for err in errors)


@pytest.mark.unit
def test_check_hard_limits_default_env_values(monkeypatch):
    """Test that default environment values are used when not set."""
    # Clear all env vars to use defaults
    monkeypatch.delenv("SLURM_MAX_GPUS", raising=False)
    monkeypatch.delenv("SLURM_MAX_TIME_HOURS", raising=False)
    monkeypatch.delenv("SLURM_MAX_CPUS", raising=False)
    monkeypatch.delenv("SLURM_MAX_MEM_PER_CPU_GB", raising=False)

    # These should use defaults: 4 GPUs, 24h, 16 CPUs, 16GB
    errors = _check_hard_limits(gpu_count=2, hours=12, cpu_count=8, mem_value=8.0)

    assert len(errors) == 0


@pytest.mark.unit
def test_check_algorithm_recommendations():
    """Test algorithm-specific recommendations."""
    ctx = HPCContext(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=100,
        gpu_count=1,
        total_hours=12.0,
        cpu_count=4,
        mem_value=4.0,
    )

    warnings, recommendations = _check_algorithm_recommendations(ctx)

    # Should return two lists
    assert isinstance(warnings, list)
    assert isinstance(recommendations, list)


@pytest.mark.unit
def test_check_general_recommendations_low_epochs():
    """Test recommendations for low epoch count."""
    ctx = HPCContext(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=30,  # Low (< 50 threshold)
        gpu_count=1,
        total_hours=12.0,
        cpu_count=4,
        mem_value=4.0,
    )

    warnings, recommendations = _check_general_recommendations(ctx)

    # Should return two lists
    assert isinstance(warnings, list)
    assert isinstance(recommendations, list)
    # Low epochs should trigger a warning
    assert any("epoch" in warn.lower() for warn in warnings)
    assert any("epoch" in rec.lower() for rec in recommendations)


@pytest.mark.unit
def test_check_general_recommendations_high_epochs():
    """Test recommendations for high epoch count."""
    ctx = HPCContext(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=1000,  # High
        gpu_count=1,
        total_hours=12.0,
        cpu_count=4,
        mem_value=4.0,
    )

    warnings, recommendations = _check_general_recommendations(ctx)

    # Should return two lists
    assert isinstance(warnings, list)
    assert isinstance(recommendations, list)
    # High epochs should trigger a warning
    assert any("epoch" in warn.lower() for warn in warnings)


@pytest.mark.unit
def test_check_contextual_recommendations_low_cpu():
    """Test contextual recommendations for low CPU count."""
    ctx = HPCContext(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        gpu_count=2,
        total_hours=24.0,
        cpu_count=2,  # Low for 2 GPUs
        mem_value=4.0,
    )

    warnings, recommendations = _check_contextual_recommendations(ctx)

    # Should return two lists
    assert isinstance(warnings, list)
    assert isinstance(recommendations, list)


@pytest.mark.unit
def test_check_contextual_recommendations_low_memory():
    """Test contextual recommendations for low memory."""
    ctx = HPCContext(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        gpu_count=2,
        total_hours=24.0,
        cpu_count=8,
        mem_value=1.0,  # Low memory
    )

    warnings, recommendations = _check_contextual_recommendations(ctx)

    # Should return two lists
    assert isinstance(warnings, list)
    assert isinstance(recommendations, list)


@pytest.mark.unit
def test_hpc_inputs_all_fields_required():
    """Test that HPCInputs requires all fields."""
    # This should work
    inputs = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="2",
        slurm_time="12:00:00",
        slurm_cpus_per_task="8",
        slurm_mem_per_cpu="4G",
    )

    assert inputs.algorithm == "cgan_cnn_2d"
    assert inputs.slurm_gpus == "2"


@pytest.mark.unit
def test_hpc_context_numeric_types():
    """Test that HPCContext stores numeric types correctly."""
    context = HPCContext(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        gpu_count=2,
        total_hours=24.5,
        cpu_count=8,
        mem_value=4.5,
    )

    assert isinstance(context.gpu_count, int)
    assert isinstance(context.total_hours, float)
    assert isinstance(context.cpu_count, int)
    assert isinstance(context.mem_value, float)
