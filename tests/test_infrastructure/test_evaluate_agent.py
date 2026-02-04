"""Integration tests for evaluate_agent.py evaluation flow.

These tests verify the full evaluation pipeline without making actual LLM calls.
They test dataset preparation, agent prediction flow, and scorer integration.
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

# Add project root to path before importing project modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.shared.scorers import (  # noqa: E402
    score_output_quality,
)

# Set up random generator for reproducible tests
rng = np.random.default_rng(seed=42)


def prepare_evaluation_dataset(
    prompts: list[dict], sample_size: int, problem_type: str, dataset_name: str
) -> list[dict]:
    """Helper function extracted from evaluate_agent for testing."""
    return [
        {
            "prompt": prompt_data["prompt"],
            "conditions": prompt_data["conditions"],
            "metadata": {
                **prompt_data.get("metadata", {}),
                "example_id": prompt_data.get("example_id", i),
                "dataset_split": prompt_data.get("dataset_split", "test"),
                "problem_type": problem_type,
                "dataset_name": dataset_name,
            },
            "target": prompt_data.get("target", {}),
        }
        for i, prompt_data in enumerate(prompts[:sample_size])
    ]


class TestPrepareEvaluationDataset:
    """Tests for dataset preparation."""

    def test_basic_dataset_preparation(self):
        """Test preparing a basic evaluation dataset."""
        prompts = [
            {
                "prompt": "Design a beam",
                "conditions": {"volfrac": 0.3, "rmin": 2.0},
                "metadata": {"force_description": "concentrated"},
                "target": {"compliance": 100.0},
                "example_id": 0,
            },
            {
                "prompt": "Design another beam",
                "conditions": {"volfrac": 0.4, "rmin": 3.0},
                "metadata": {"force_description": "uniform"},
                "target": {"compliance": 150.0},
                "example_id": 1,
            },
        ]

        dataset = prepare_evaluation_dataset(
            prompts=prompts,
            sample_size=2,
            problem_type="beams2d",
            dataset_name="IDEALLab/beams_2d_50_100_v0",
        )

        assert len(dataset) == 2
        assert dataset[0]["prompt"] == "Design a beam"
        assert dataset[0]["conditions"]["volfrac"] == 0.3
        assert dataset[0]["metadata"]["problem_type"] == "beams2d"
        assert dataset[0]["metadata"]["example_id"] == 0
        assert dataset[0]["target"]["compliance"] == 100.0

    def test_sample_size_limit(self):
        """Test that sample_size limits the dataset correctly."""
        prompts = [
            {"prompt": f"Design {i}", "conditions": {}, "target": {}} for i in range(10)
        ]

        dataset = prepare_evaluation_dataset(
            prompts=prompts,
            sample_size=3,
            problem_type="beams2d",
            dataset_name="test",
        )

        assert len(dataset) == 3

    def test_missing_fields_handled(self):
        """Test that missing optional fields are handled gracefully."""
        prompts = [
            {
                "prompt": "Design a beam",
                "conditions": {},
                # Missing metadata, target, example_id
            }
        ]

        dataset = prepare_evaluation_dataset(
            prompts=prompts,
            sample_size=1,
            problem_type="beams2d",
            dataset_name="test",
        )

        assert len(dataset) == 1
        assert dataset[0]["metadata"]["example_id"] == 0  # Default
        assert "problem_type" in dataset[0]["metadata"]


class TestScorerIntegration:
    """Tests for scorer integration with evaluation flow."""

    def _create_mock_output(self):
        """Create mock agent output for scorer testing."""
        design = rng.random((50, 100))

        tool_message = ToolMessage(
            content=json.dumps(
                {
                    "optimized_design": design.tolist(),
                    "final_compliance": 100.0,
                    "volfrac": 0.3,
                    "success": True,
                }
            ),
            tool_call_id="test",
            name="simulate_design",
        )

        return {
            "messages": [HumanMessage(content="test"), tool_message],
            "response": "Done",
            "model": "test",
        }

    @patch("benchmarks.shared.scorers.output_quality_visual_scorer.get_hf_dataset")
    def test_generic_scorer_beams2d(self, mock_get_hf_dataset):
        """Test generic scorer with beams2d problem."""
        # Create target with ground truth design
        target_design = rng.random((50, 100))

        # Mock the HuggingFace dataset to avoid network calls in CI
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(
            return_value={
                "optimal_design": target_design.tolist(),  # Ground truth design (keep 2D shape)
                "c": 95.0,  # Compliance
            }
        )
        mock_dataset.__len__ = Mock(return_value=100)  # Dataset size
        mock_get_hf_dataset.return_value = mock_dataset

        output = self._create_mock_output()

        target = {
            "optimal_design": target_design.tolist(),
            "c": 95.0,  # Target compliance
        }

        metadata = {
            "example_id": 0,
            "problem_type": "beams2d",
            "dataset_name": "IDEALLab/beams_2d_50_100_v0",
        }

        # Run scorer
        score_result = score_output_quality(output, target, metadata)

        # Verify score structure
        assert "score" in score_result
        assert "design_found" in score_result
        assert "iou" in score_result
        assert "pixel_accuracy" in score_result
        assert "agent_compliance" in score_result
        assert "target_compliance" in score_result

        # Check that scores are in valid range
        assert 0.0 <= score_result["score"] <= 1.0
        assert score_result["design_found"] is True

    @patch("benchmarks.shared.scorers.output_quality_visual_scorer.get_hf_dataset")
    def test_generic_scorer_photonics2d(self, mock_get_hf_dataset):
        """Test generic scorer with photonics2d problem."""
        design = rng.random((120, 120))
        target_design = rng.random((120, 120))

        # Mock the HuggingFace dataset to avoid network calls in CI
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(
            return_value={
                "optimal_design": target_design.tolist(),  # Ground truth design (keep 2D shape)
                "total_overlap": 0.90,
            }
        )
        mock_dataset.__len__ = Mock(return_value=100)  # Dataset size
        mock_get_hf_dataset.return_value = mock_dataset

        tool_message = ToolMessage(
            content=json.dumps(
                {
                    "optimized_design": design.tolist(),
                    "total_overlap": 0.85,
                    "success": True,
                }
            ),
            tool_call_id="test",
            name="simulate_design",
        )

        output = {
            "messages": [HumanMessage(content="test"), tool_message],
            "response": "Done",
            "model": "test",
        }

        target = {
            "optimal_design": target_design.tolist(),
            "total_overlap": 0.90,
        }

        metadata = {
            "example_id": 0,
            "problem_type": "photonics2d",
            "dataset_name": "IDEALLab/photonics_2d_120_120_v0",
        }

        score_result = score_output_quality(output, target, metadata)

        # Verify photonics-specific fields
        assert "score" in score_result
        assert "agent_total_overlap" in score_result
        assert "target_total_overlap" in score_result
        assert 0.0 <= score_result["score"] <= 1.0

    def test_scorer_with_missing_design(self):
        """Test scorer behavior when design is not found."""
        output = {
            "messages": [HumanMessage(content="test")],  # No tool message
            "response": "I couldn't optimize the design",
            "model": "test",
        }

        target = {
            "optimal_design": rng.random((50, 100)).tolist(),
            "c": 100.0,
        }

        metadata = {
            "example_id": 0,
            "problem_type": "beams2d",
            "dataset_name": "test",
        }

        score_result = score_output_quality(output, target, metadata)

        # Should return zero score when design not found
        assert score_result["design_found"] is False
        assert score_result["score"] == 0.0


class TestEvaluationWorkflow:
    """Integration tests for the evaluation workflow components."""

    @pytest.fixture
    def mock_prompts_file(self, tmp_path):
        """Create a temporary prompts file for testing."""
        prompts = [
            {
                "prompt": f"Design beam {i}",
                "conditions": {"volfrac": 0.3, "rmin": 2.0, "forcedist": 0.1},
                "metadata": {"force_description": "concentrated"},
                "target": {
                    "compliance": 100.0 + i * 10,
                    "optimal_design": rng.random((50, 100)).tolist(),
                },
                "example_id": i,
            }
            for i in range(3)
        ]

        prompts_file = tmp_path / "test_prompts.json"
        with prompts_file.open("w") as f:
            json.dump(prompts, f)

        return prompts_file

    def test_load_prompts(self, mock_prompts_file):
        """Test loading prompts from file."""
        with mock_prompts_file.open() as f:
            prompts = json.load(f)

        assert len(prompts) == 3
        assert all("prompt" in p for p in prompts)
        assert all("conditions" in p for p in prompts)

    def test_dataset_preparation_from_file(self, mock_prompts_file):
        """Test preparing dataset from loaded prompts."""
        with mock_prompts_file.open() as f:
            prompts = json.load(f)

        dataset = prepare_evaluation_dataset(
            prompts=prompts,
            sample_size=2,
            problem_type="beams2d",
            dataset_name="test",
        )

        assert len(dataset) == 2
        assert all("prompt" in d for d in dataset)
        assert all("metadata" in d for d in dataset)

    def test_end_to_end_scoring_workflow(self, mock_prompts_file):
        """Test the complete workflow from prompts to scores."""
        # Load prompts
        with mock_prompts_file.open() as f:
            prompts = json.load(f)

        # Prepare dataset
        dataset = prepare_evaluation_dataset(
            prompts=prompts, sample_size=2, problem_type="beams2d", dataset_name="test"
        )

        # Simulate agent output
        for example in dataset:
            # Create mock agent output
            design = rng.random((50, 100))
            tool_msg = ToolMessage(
                content=json.dumps(
                    {
                        "optimized_design": design.tolist(),
                        "final_compliance": 105.0,
                        "volfrac": 0.3,
                        "success": True,
                    }
                ),
                tool_call_id="test",
                name="simulate_design",
            )

            output = {
                "messages": [HumanMessage(content=example["prompt"]), tool_msg],
                "response": "Done",
                "model": "test",
            }

            # Score the output
            score = score_output_quality(output, example["target"], example["metadata"])

            # Verify score structure
            assert "score" in score
            assert 0.0 <= score["score"] <= 1.0
            assert "design_found" in score
            assert score["design_found"] is True
