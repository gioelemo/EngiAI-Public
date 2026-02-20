"""Tests for the HPC workflow scorer.

Tests for score_hpc_workflow() and its helper functions in
benchmarks/shared/scorers/hpc_workflow_scorer.py.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add project root to path to import benchmark modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.shared.scorers.hpc_workflow_scorer import (  # noqa: E402
    _check_evaluate_model,
    _check_tool_called,
    _check_training_config,
    _extract_evaluation_metrics,
    score_hpc_workflow,
)

# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================


class TestCheckToolCalled:
    """Tests for _check_tool_called."""

    @pytest.mark.unit
    def test_tool_present(self):
        tc_info = [{"name": "submit_slurm_job", "args": {}}]
        assert _check_tool_called(tc_info, "submit_slurm_job") is True

    @pytest.mark.unit
    def test_tool_absent(self):
        tc_info = [{"name": "submit_slurm_job", "args": {}}]
        assert _check_tool_called(tc_info, "monitor_job_until_complete") is False

    @pytest.mark.unit
    def test_empty_list(self):
        assert _check_tool_called([], "submit_slurm_job") is False

    @pytest.mark.unit
    def test_multiple_tools(self):
        tc_info = [
            {"name": "generate_training_command", "args": {}},
            {"name": "submit_slurm_job", "args": {}},
            {"name": "monitor_job_until_complete", "args": {}},
        ]
        assert _check_tool_called(tc_info, "monitor_job_until_complete") is True
        assert _check_tool_called(tc_info, "evaluate_model") is False


class TestCheckTrainingConfig:
    """Tests for _check_training_config."""

    @pytest.mark.unit
    def test_correct_config_top_level(self):
        """Config passed as top-level args."""
        tc_info = [
            {
                "name": "generate_training_command",
                "args": {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"},
            }
        ]
        config = {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"}
        assert _check_training_config(tc_info, config) is True

    @pytest.mark.unit
    def test_correct_config_nested_cfg(self):
        """Config passed as nested 'cfg' dict."""
        tc_info = [
            {
                "name": "generate_training_command",
                "args": {
                    "cfg": {"seed": 2, "epochs": 50, "algorithm": "cgan_cnn_2d"}
                },
            }
        ]
        config = {"seed": 2, "epochs": 50, "algorithm": "cgan_cnn_2d"}
        assert _check_training_config(tc_info, config) is True

    @pytest.mark.unit
    def test_wrong_seed(self):
        tc_info = [
            {
                "name": "generate_training_command",
                "args": {"seed": 99, "epochs": 20, "algorithm": "cgan_cnn_2d"},
            }
        ]
        config = {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"}
        assert _check_training_config(tc_info, config) is False

    @pytest.mark.unit
    def test_wrong_epochs(self):
        tc_info = [
            {
                "name": "generate_training_command",
                "args": {"seed": 1, "epochs": 999, "algorithm": "cgan_cnn_2d"},
            }
        ]
        config = {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"}
        assert _check_training_config(tc_info, config) is False

    @pytest.mark.unit
    def test_wrong_algorithm(self):
        tc_info = [
            {
                "name": "generate_training_command",
                "args": {"seed": 1, "epochs": 20, "algorithm": "diffusion_2d"},
            }
        ]
        config = {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"}
        assert _check_training_config(tc_info, config) is False

    @pytest.mark.unit
    def test_algorithm_defaults_to_cgan(self):
        """Algorithm defaults to cgan_cnn_2d when not specified."""
        tc_info = [
            {
                "name": "generate_training_command",
                "args": {"seed": 1, "epochs": 20},
            }
        ]
        config = {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"}
        assert _check_training_config(tc_info, config) is True

    @pytest.mark.unit
    def test_no_generate_training_command(self):
        tc_info = [{"name": "submit_slurm_job", "args": {}}]
        config = {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"}
        assert _check_training_config(tc_info, config) is False

    @pytest.mark.unit
    def test_empty_tool_calls(self):
        config = {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"}
        assert _check_training_config([], config) is False


class TestCheckEvaluateModel:
    """Tests for _check_evaluate_model."""

    @pytest.mark.unit
    def test_direct_tool_call(self):
        tc_info = [{"name": "evaluate_model", "args": {}}]
        assert _check_evaluate_model(tc_info, []) is True

    @pytest.mark.unit
    def test_cli_fallback_engiopt_module(self):
        """CLI command using python -m engiopt.cgan_cnn_2d.evaluate_cgan_cnn_2d."""
        tc_info = [
            {
                "name": "execute_cli_command",
                "args": {
                    "command": "python -m engiopt.cgan_cnn_2d.evaluate_cgan_cnn_2d --seed 1"
                },
            }
        ]
        assert _check_evaluate_model(tc_info, []) is True

    @pytest.mark.unit
    def test_cli_loose_match_rejected(self):
        """Loose match that doesn't follow engiopt.*.evaluate_* pattern."""
        tc_info = [
            {
                "name": "execute_cli_command",
                "args": {"command": "evaluate_something engiopt"},
            }
        ]
        # "engiopt." not in command (no dot), so should NOT match
        assert _check_evaluate_model(tc_info, []) is False

    @pytest.mark.unit
    def test_unrelated_cli_command(self):
        tc_info = [
            {
                "name": "execute_cli_command",
                "args": {"command": "python train.py --seed 1"},
            }
        ]
        assert _check_evaluate_model(tc_info, []) is False

    @pytest.mark.unit
    def test_no_tool_calls(self):
        assert _check_evaluate_model([], []) is False


class TestExtractEvaluationMetrics:
    """Tests for _extract_evaluation_metrics."""

    @pytest.mark.unit
    def test_all_metrics_extracted(self):
        messages = [
            SimpleNamespace(
                content=(
                    "Results: IOG: 975.586, COG: 1026.445, FOG: -1.2008, "
                    "MMD: 0.04744, DPP: 3.37e-18, viol: 0.76"
                )
            )
        ]
        metrics = _extract_evaluation_metrics(messages)
        assert len(metrics) == 6
        assert metrics["IOG"] == pytest.approx(975.586, rel=1e-3)
        assert metrics["COG"] == pytest.approx(1026.445, rel=1e-3)
        assert metrics["FOG"] == pytest.approx(-1.2008, rel=1e-3)
        assert metrics["MMD"] == pytest.approx(0.04744, rel=1e-3)
        assert metrics["DPP"] == pytest.approx(3.37e-18, rel=1e-3)
        assert metrics["viol"] == pytest.approx(0.76, rel=1e-3)

    @pytest.mark.unit
    def test_negative_values(self):
        """Regression test: regex must handle negative numbers."""
        messages = [SimpleNamespace(content="FOG: -1.2717858638874315")]
        metrics = _extract_evaluation_metrics(messages)
        assert "FOG" in metrics
        assert metrics["FOG"] < 0

    @pytest.mark.unit
    def test_scientific_notation_lowercase(self):
        """Regression test: regex must handle lowercase 'e' in sci notation."""
        messages = [SimpleNamespace(content="DPP: 7.109550764169547e-21")]
        metrics = _extract_evaluation_metrics(messages)
        assert "DPP" in metrics
        assert metrics["DPP"] == pytest.approx(7.11e-21, rel=1e-2)

    @pytest.mark.unit
    def test_scientific_notation_uppercase(self):
        """Regex handles uppercase 'E' in sci notation."""
        messages = [SimpleNamespace(content="IOG: 5.06E+07")]
        metrics = _extract_evaluation_metrics(messages)
        assert "IOG" in metrics
        assert metrics["IOG"] == pytest.approx(5.06e7, rel=1e-2)

    @pytest.mark.unit
    def test_json_style_keys(self):
        """Metrics in JSON-style output."""
        messages = [
            SimpleNamespace(content='{"IOG": 100.5, "COG": 200.3, "viol": 0.44}')
        ]
        metrics = _extract_evaluation_metrics(messages)
        assert metrics["IOG"] == pytest.approx(100.5, rel=1e-3)
        assert metrics["COG"] == pytest.approx(200.3, rel=1e-3)
        assert metrics["viol"] == pytest.approx(0.44, rel=1e-3)

    @pytest.mark.unit
    def test_python_dict_style_keys(self):
        """Metrics in Python dict-style output (single quotes)."""
        messages = [SimpleNamespace(content="{'IOG': 100.5, 'COG': 200.3}")]
        metrics = _extract_evaluation_metrics(messages)
        assert "IOG" in metrics
        assert "COG" in metrics

    @pytest.mark.unit
    def test_no_metrics_in_messages(self):
        messages = [SimpleNamespace(content="Training complete. Model saved.")]
        metrics = _extract_evaluation_metrics(messages)
        assert metrics == {}

    @pytest.mark.unit
    def test_empty_messages(self):
        assert _extract_evaluation_metrics([]) == {}

    @pytest.mark.unit
    def test_non_string_content_ignored(self):
        messages = [SimpleNamespace(content=12345)]
        assert _extract_evaluation_metrics(messages) == {}

    @pytest.mark.unit
    def test_first_occurrence_wins(self):
        """If a metric appears in multiple messages, the first one is kept."""
        messages = [
            SimpleNamespace(content="IOG: 100.0"),
            SimpleNamespace(content="IOG: 999.0"),
        ]
        metrics = _extract_evaluation_metrics(messages)
        assert metrics["IOG"] == pytest.approx(100.0)


# ============================================================================
# COMPOSITE SCORER TESTS
# ============================================================================


def _make_output(tool_names, messages=None):
    """Helper to build scorer output dict."""
    return {
        "tool_calls_info": [{"name": n, "args": {}} for n in tool_names],
        "messages": messages or [],
        "response": "",
    }


def _make_output_with_config(tool_names, config_args, messages=None):
    """Helper to build scorer output with specific args for generate_training_command."""
    tc_info = []
    for n in tool_names:
        if n == "generate_training_command":
            tc_info.append({"name": n, "args": config_args})
        else:
            tc_info.append({"name": n, "args": {}})
    return {
        "tool_calls_info": tc_info,
        "messages": messages or [],
        "response": "",
    }


_BASE_METADATA = {
    "example_id": 0,
    "training_config": {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"},
}


class TestScoreHpcWorkflow:
    """Tests for score_hpc_workflow composite scorer."""

    @pytest.mark.unit
    def test_perfect_score_no_metrics(self):
        """All 4 tools called with correct config but no eval metrics."""
        output = _make_output_with_config(
            [
                "generate_training_command",
                "submit_slurm_job",
                "monitor_job_until_complete",
                "evaluate_model",
            ],
            {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"},
        )
        result = score_hpc_workflow(output, {}, _BASE_METADATA)

        assert result["step_completion_rate"] == 1.0
        assert result["training_config_correct"] is True
        assert result["config_score"] == 1.0
        assert result["eval_metrics_score"] == 0.0
        # 0.70 * 1.0 + 0.15 * 1.0 + 0.15 * 0.0 = 0.85
        assert result["hpc_workflow_score"] == pytest.approx(0.85)

    @pytest.mark.unit
    def test_perfect_score_with_all_metrics(self):
        """All 4 tools called, correct config, all 6 eval metrics present."""
        messages = [
            SimpleNamespace(
                content=(
                    "IOG: 100.0, COG: 200.0, FOG: -1.0, "
                    "MMD: 0.05, DPP: 1e-18, viol: 0.5"
                )
            )
        ]
        output = _make_output_with_config(
            [
                "generate_training_command",
                "submit_slurm_job",
                "monitor_job_until_complete",
                "evaluate_model",
            ],
            {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"},
            messages=messages,
        )
        result = score_hpc_workflow(output, {}, _BASE_METADATA)

        assert result["step_completion_rate"] == 1.0
        assert result["config_score"] == 1.0
        assert result["eval_metrics_score"] == 1.0
        # 0.70 * 1.0 + 0.15 * 1.0 + 0.15 * 1.0 = 1.0
        assert result["hpc_workflow_score"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_no_tools_called(self):
        """No tools called at all → score 0."""
        output = _make_output([])
        result = score_hpc_workflow(output, {}, _BASE_METADATA)

        assert result["step_completion_rate"] == 0.0
        assert result["config_score"] == 0.0
        assert result["eval_metrics_score"] == 0.0
        assert result["hpc_workflow_score"] == 0.0

    @pytest.mark.unit
    def test_partial_steps(self):
        """Only 2 of 4 tools called."""
        output = _make_output(["generate_training_command", "submit_slurm_job"])
        result = score_hpc_workflow(output, {}, _BASE_METADATA)

        assert result["step_completion_rate"] == pytest.approx(0.5)
        assert result["steps_completed_count"] == 2
        assert result["steps_total"] == 4
        assert result["step_generate_training_command"] is True
        assert result["step_submit_slurm_job"] is True
        assert result["step_monitor_job_until_complete"] is False
        assert result["step_evaluate_model"] is False

    @pytest.mark.unit
    def test_wrong_config_penalised(self):
        """All tools called but wrong seed → config_score = 0."""
        output = _make_output_with_config(
            [
                "generate_training_command",
                "submit_slurm_job",
                "monitor_job_until_complete",
                "evaluate_model",
            ],
            {"seed": 99, "epochs": 20, "algorithm": "cgan_cnn_2d"},
        )
        result = score_hpc_workflow(output, {}, _BASE_METADATA)

        assert result["step_completion_rate"] == 1.0
        assert result["training_config_correct"] is False
        assert result["config_score"] == 0.0
        # 0.70 * 1.0 + 0.15 * 0.0 + 0.15 * 0.0 = 0.70
        assert result["hpc_workflow_score"] == pytest.approx(0.70)

    @pytest.mark.unit
    def test_partial_metrics_score(self):
        """3 of 6 metrics present → eval_metrics_score = 0.5."""
        messages = [SimpleNamespace(content="IOG: 100.0, COG: 200.0, FOG: -1.0")]
        output = _make_output_with_config(
            [
                "generate_training_command",
                "submit_slurm_job",
                "monitor_job_until_complete",
                "evaluate_model",
            ],
            {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"},
            messages=messages,
        )
        result = score_hpc_workflow(output, {}, _BASE_METADATA)

        assert result["eval_metrics_count"] == 3
        assert result["eval_metrics_score"] == pytest.approx(0.5)
        # 0.70 * 1.0 + 0.15 * 1.0 + 0.15 * 0.5 = 0.925
        assert result["hpc_workflow_score"] == pytest.approx(0.925)

    @pytest.mark.unit
    def test_cli_fallback_for_evaluate(self):
        """evaluate_model via CLI fallback counts as step completion."""
        output = _make_output_with_config(
            ["generate_training_command", "submit_slurm_job", "monitor_job_until_complete"],
            {"seed": 1, "epochs": 20, "algorithm": "cgan_cnn_2d"},
        )
        # Add CLI fallback for evaluate_model
        output["tool_calls_info"].append(
            {
                "name": "execute_cli_command",
                "args": {
                    "command": "python -m engiopt.cgan_cnn_2d.evaluate_cgan_cnn_2d --seed 1"
                },
            }
        )
        result = score_hpc_workflow(output, {}, _BASE_METADATA)

        assert result["step_evaluate_model"] is True
        assert result["step_completion_rate"] == 1.0

    @pytest.mark.unit
    def test_return_dict_structure(self):
        """Verify all expected keys are present in the result."""
        output = _make_output([])
        result = score_hpc_workflow(output, {}, _BASE_METADATA)

        # Primary metric
        assert "hpc_workflow_score" in result
        # Step details
        assert "step_completion_rate" in result
        assert "steps_completed_count" in result
        assert "steps_total" in result
        assert "step_generate_training_command" in result
        assert "step_submit_slurm_job" in result
        assert "step_monitor_job_until_complete" in result
        assert "step_evaluate_model" in result
        # Config
        assert "training_config_correct" in result
        assert "config_score" in result
        # Eval metrics
        assert "eval_metrics" in result
        assert "eval_metrics_count" in result
        assert "eval_metrics_score" in result
        # Metadata
        assert "example_id" in result
