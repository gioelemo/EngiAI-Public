"""Tests for objective extraction from tool messages."""

from langchain_core.messages import ToolMessage

from benchmarks.shared.objective_extractor import (
    calculate_objective_score,
    extract_objectives_from_tool_messages,
)
from benchmarks.shared.problem_registry import get_problem_config


class TestObjectiveExtraction:
    """Tests for extracting objectives from tool messages."""

    def test_extract_from_dict_content(self):
        """Test extraction when tool returns dict content."""
        config = get_problem_config("beams2d")

        # Create mock tool message with dict content
        tool_msg = ToolMessage(
            content={"final_compliance": 123.45, "success": True},
            tool_call_id="test_call",
            name="optimize_design",
        )

        messages = [tool_msg]
        objectives = extract_objectives_from_tool_messages(
            messages, config, example_id=0
        )

        assert objectives["compliance"] == 123.45

    def test_extract_from_string_json(self):
        """Test extraction from JSON string content."""
        config = get_problem_config("beams2d")

        tool_msg = ToolMessage(
            content='{"final_compliance": 98.76, "success": true}',
            tool_call_id="test_call",
            name="optimize_design",
        )

        messages = [tool_msg]
        objectives = extract_objectives_from_tool_messages(
            messages, config, example_id=0
        )

        assert objectives["compliance"] == 98.76

    def test_extract_from_string_python_dict(self):
        """Test extraction from Python dict string (with single quotes)."""
        config = get_problem_config("beams2d")

        tool_msg = ToolMessage(
            content="{'final_compliance': 111.11, 'success': True}",
            tool_call_id="test_call",
            name="optimize_design",
        )

        messages = [tool_msg]
        objectives = extract_objectives_from_tool_messages(
            messages, config, example_id=0
        )

        assert objectives["compliance"] == 111.11

    def test_extract_with_alias(self):
        """Test extraction using field alias."""
        config = get_problem_config("beams2d")

        # Use alias "final_c" instead of "final_compliance"
        tool_msg = ToolMessage(
            content={"final_c": 55.55},
            tool_call_id="test_call",
            name="optimize_design",
        )

        messages = [tool_msg]
        objectives = extract_objectives_from_tool_messages(
            messages, config, example_id=0
        )

        assert objectives["compliance"] == 55.55

    def test_extract_photonics2d_objective(self):
        """Test extraction for photonics2d problem."""
        config = get_problem_config("photonics2d")

        tool_msg = ToolMessage(
            content={"total_overlap": 0.987},
            tool_call_id="test_call",
            name="optimize_design",
        )

        messages = [tool_msg]
        objectives = extract_objectives_from_tool_messages(
            messages, config, example_id=0
        )

        assert objectives["total_overlap"] == 0.987

    def test_no_tool_messages(self):
        """Test when no tool messages are present."""
        config = get_problem_config("beams2d")

        messages = []  # Empty message list
        objectives = extract_objectives_from_tool_messages(
            messages, config, example_id=0
        )

        # Should return None for all objectives
        assert objectives["compliance"] is None

    def test_wrong_tool_name(self):
        """Test when tool message has wrong name."""
        config = get_problem_config("beams2d")

        tool_msg = ToolMessage(
            content={"final_compliance": 123.45},
            tool_call_id="test_call",
            name="different_tool",  # Wrong tool name
        )

        messages = [tool_msg]
        objectives = extract_objectives_from_tool_messages(
            messages, config, example_id=0
        )

        assert objectives["compliance"] is None


class TestObjectiveScoring:
    """Tests for objective score calculation."""

    def test_perfect_match(self):
        """Test scoring with perfect objective match."""
        config = get_problem_config("beams2d")

        agent_objectives = {"compliance": 100.0}
        target_objectives = {"compliance": 100.0}

        score, metrics = calculate_objective_score(
            agent_objectives, target_objectives, config, example_id=0
        )

        assert score == 1.0
        assert metrics["compliance_relative_error"] == 0.0
        assert metrics["compliance_score"] == 1.0

    def test_within_threshold(self):
        """Test scoring when within threshold."""
        config = get_problem_config("beams2d")

        # 10% error, threshold is 20% -> score should be 0.5
        agent_objectives = {"compliance": 110.0}
        target_objectives = {"compliance": 100.0}

        score, metrics = calculate_objective_score(
            agent_objectives, target_objectives, config, example_id=0
        )

        assert 0.4 < score < 0.6  # Approximately 0.5
        assert metrics["compliance_relative_error"] == 0.1

    def test_exceed_threshold(self):
        """Test scoring when error exceeds threshold."""
        config = get_problem_config("beams2d")

        # 50% error, threshold is 20% -> score should be 0.0
        agent_objectives = {"compliance": 150.0}
        target_objectives = {"compliance": 100.0}

        score, metrics = calculate_objective_score(
            agent_objectives, target_objectives, config, example_id=0
        )

        assert score == 0.0
        assert metrics["compliance_relative_error"] == 0.5

    def test_missing_agent_objective(self):
        """Test scoring when agent objective is missing."""
        config = get_problem_config("beams2d")

        agent_objectives = {"compliance": None}
        target_objectives = {"compliance": 100.0}

        score, metrics = calculate_objective_score(
            agent_objectives, target_objectives, config, example_id=0
        )

        assert score == 0.0
        assert metrics["agent_compliance"] is None

    def test_maximize_objective(self):
        """Test scoring for maximize objective (photonics2d)."""
        config = get_problem_config("photonics2d")

        agent_objectives = {"total_overlap": 0.9}
        target_objectives = {"total_overlap": 1.0}

        score, metrics = calculate_objective_score(
            agent_objectives, target_objectives, config, example_id=0
        )

        # 10% error with 20% threshold -> score = 0.5
        assert 0.4 < score < 0.6
        assert abs(metrics["total_overlap_relative_error"] - 0.1) < 0.001
