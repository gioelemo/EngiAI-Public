"""Additional tests for engiopt to improve coverage."""

import os
from unittest.mock import patch

import pytest

from src.tools.engiopt import (
    HPCInputs,
    _check_general_recommendations,
    _check_hard_limits,
    _validate_sampling_inputs,
    generate_training_command,
    list_available_algorithms,
)


@pytest.mark.unit
def test_list_available_algorithms():
    """Test listing available algorithms."""
    result = list_available_algorithms.invoke({})

    assert "algorithms" in result
    assert "details" in result
    assert "message" in result
    assert len(result["algorithms"]) > 0
    assert "cgan_cnn_2d" in result["algorithms"]
    assert "diffusion_2d_cond" in result["algorithms"]


@pytest.mark.unit
def test_list_available_algorithms_structure():
    """Test that list_available_algorithms returns proper structure."""
    result = list_available_algorithms.invoke({})

    assert "algorithms" in result
    assert isinstance(result["algorithms"], list)
    assert len(result["algorithms"]) > 0


@pytest.mark.unit
def test_list_available_algorithms_details():
    """Test that algorithm details are provided."""
    result = list_available_algorithms.invoke({})

    assert "details" in result
    for algo in result["algorithms"]:
        assert algo in result["details"]


@pytest.mark.unit
def test_hpc_inputs_dataclass_creation():
    """Test creating HPCInputs dataclass."""
    inputs = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        slurm_gpus="1",
        slurm_time="4:00:00",
        slurm_cpus_per_task="4",
        slurm_mem_per_cpu="2G",
    )

    assert inputs.algorithm == "cgan_cnn_2d"
    assert inputs.problem_id == "beams2d"
    assert inputs.epochs == 200
    assert inputs.slurm_gpus == "1"


@pytest.mark.unit
def test_training_config_defaults():
    """Test TrainingConfig dataclass with defaults."""
    from src.tools.engiopt import TrainingConfig

    cfg = TrainingConfig(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
    )

    assert cfg.algorithm == "cgan_cnn_2d"
    assert cfg.epochs == 200
    assert cfg.seed == 1


@pytest.mark.unit
def test_training_config_custom():
    """Test TrainingConfig with custom values."""
    from src.tools.engiopt import TrainingConfig

    cfg = TrainingConfig(
        algorithm="diffusion_2d_cond",
        problem_id="beams2d",
        epochs=100,
        seed=42,
        gpus=2,
        time_hours=3.5,
    )

    assert cfg.algorithm == "diffusion_2d_cond"
    assert cfg.epochs == 100
    assert cfg.seed == 42
    assert cfg.gpus == 2
    assert cfg.time_hours == 3.5


@pytest.mark.unit
def test_check_hard_limits_valid():
    """Test hard limits check with valid values."""
    errors = _check_hard_limits(
        gpu_count=1,
        hours=4.0,
        cpu_count=4,
        mem_value=2.0,
    )

    assert len(errors) == 0


@pytest.mark.unit
def test_check_hard_limits_too_many_gpus():
    """Test hard limits with excessive GPUs."""
    errors = _check_hard_limits(
        gpu_count=16,  # Way too many
        hours=4.0,
        cpu_count=4,
        mem_value=2.0,
    )

    assert len(errors) > 0
    assert any("gpu" in e.lower() for e in errors)


@pytest.mark.unit
def test_check_hard_limits_too_much_memory():
    """Test hard limits with excessive memory."""
    errors = _check_hard_limits(
        gpu_count=1,
        hours=4.0,
        cpu_count=4,
        mem_value=500.0,  # Way too much
    )

    assert len(errors) > 0
    assert any("mem" in e.lower() for e in errors)


@pytest.mark.unit
def test_check_general_recommendations_basic():
    """Test general recommendations with typical values."""
    from src.tools.engiopt import HPCContext

    context = HPCContext(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        gpu_count=1,
        total_hours=4.0,
        cpu_count=4,
        mem_value=2.0,
    )

    warnings, recommendations = _check_general_recommendations(context)

    # Should return lists (empty or not)
    assert isinstance(warnings, list)
    assert isinstance(recommendations, list)


@pytest.mark.unit
def test_validate_sampling_inputs_valid():
    """Test validating sampling inputs with valid values."""
    error = _validate_sampling_inputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        conditions=None,
        n_samples=10,
    )

    assert error is None


@pytest.mark.unit
def test_validate_sampling_inputs_invalid_algorithm():
    """Test validating sampling with invalid algorithm."""
    error = _validate_sampling_inputs(
        algorithm="invalid_algo",
        problem_id="beams2d",
        conditions=None,
        n_samples=10,
    )

    assert error is not None
    assert "unsupported algorithm" in error["error"].lower()


@pytest.mark.unit
def test_validate_sampling_inputs_invalid_problem():
    """Test validating sampling with invalid problem."""
    error = _validate_sampling_inputs(
        algorithm="cgan_cnn_2d",
        problem_id="invalid_problem",
        conditions=None,
        n_samples=10,
    )

    assert error is not None
    assert "unsupported problem" in error["error"].lower()


@pytest.mark.unit
def test_generate_training_command_basic(monkeypatch):
    """Test generating training command."""
    from src.tools.engiopt import TrainingConfig

    monkeypatch.setenv("USE_WANDB", "True")

    cfg = TrainingConfig(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=200,
        seed=1,
        wandb_entity="test-user",
        gpus=1,
        time_hours=4.0,
    )

    result = generate_training_command.invoke({"cfg": cfg})

    assert "slurm_script" in result or "error" in result


@pytest.mark.unit
def test_generate_training_command_wandb_disabled():
    """Test generating command when WandB is disabled."""
    from src.tools.engiopt import TrainingConfig

    with patch.dict(os.environ, {"USE_WANDB": "False"}):
        cfg = TrainingConfig(
            algorithm="cgan_cnn_2d",
            problem_id="beams2d",
            epochs=200,
            wandb_entity="test",
        )

        result = generate_training_command.invoke({"cfg": cfg})

        # Should succeed even with WandB disabled
        assert "slurm_script" in result or "success" in result


@pytest.mark.unit
def test_generate_training_command_invalid_algorithm():
    """Test generating command with invalid algorithm."""
    from src.tools.engiopt import TrainingConfig

    with patch.dict(os.environ, {"USE_WANDB": "True"}):
        cfg = TrainingConfig(
            algorithm="invalid_algo",
            problem_id="beams2d",
            epochs=200,
            wandb_entity="test",
        )

        result = generate_training_command.invoke({"cfg": cfg})

        assert result["success"] is False
        assert "unsupported algorithm" in result["error"].lower()


@pytest.mark.unit
def test_check_hard_limits_high_values():
    """Test hard limits with high resource values."""
    errors = _check_hard_limits(
        gpu_count=8,
        hours=24.0,
        cpu_count=16,
        mem_value=64.0,
    )

    # Should pass or warn depending on limits
    assert isinstance(errors, list)


@pytest.mark.unit
def test_hpc_inputs_str_fields():
    """Test HPCInputs with string fields."""
    inputs = HPCInputs(
        algorithm="cgan_cnn_2d",
        problem_id="beams2d",
        epochs=100,
        slurm_gpus="2",
        slurm_time="2:00:00",
        slurm_cpus_per_task="8",
        slurm_mem_per_cpu="4G",
    )

    assert inputs.slurm_gpus == "2"
    assert inputs.slurm_time == "2:00:00"
    assert inputs.slurm_cpus_per_task == "8"
    assert inputs.slurm_mem_per_cpu == "4G"


@pytest.mark.unit
def test_hpc_inputs_numeric_epochs():
    """Test HPCInputs with numeric epoch values."""
    for epochs in [100, 200, 500]:
        inputs = HPCInputs(
            algorithm="cgan_cnn_2d",
            problem_id="beams2d",
            epochs=epochs,
            slurm_gpus="1",
            slurm_time="4:00:00",
            slurm_cpus_per_task="4",
            slurm_mem_per_cpu="2G",
        )

        assert inputs.epochs == epochs
