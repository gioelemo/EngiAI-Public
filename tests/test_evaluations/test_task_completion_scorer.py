"""Tests for STL parameter validation, conditional resolution, and multi-export validation.

Tests for the _validate_stl_parameters(), _resolve_conditional_params(), and
_validate_multi_export_params() functions in benchmarks/shared/scorers/task_completion_scorer.py.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path to import benchmark modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.shared.scorers.task_completion_scorer import (  # noqa: E402
    _resolve_conditional_params,
    _validate_multi_export_params,
    _validate_stl_parameters,
)

# ============================================================================
# STL PARAMETER VALIDATION TESTS
# ============================================================================


@pytest.mark.unit
def test_validate_stl_parameters_all_valid():
    """Test validation with all parameters within tolerance."""
    stl_details = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should pass validation
    assert score == 1.0
    assert metrics["stl_param_violations"] == 0
    assert metrics["stl_param_validation_score"] == 1.0

    # Check individual parameters
    assert metrics["stl_scale_xy_valid"] is True
    assert metrics["stl_scale_z_valid"] is True
    assert metrics["stl_threshold_valid"] is True
    assert metrics["stl_mirror_y_valid"] is True


@pytest.mark.unit
def test_validate_stl_parameters_within_tolerance():
    """Test validation with float parameters within tolerance."""
    float_tolerance = 0.01

    stl_details = {
        "scale_xy": 2.505,  # Within tolerance of 2.5
        "scale_z": 10.008,  # Within tolerance of 10.0
        "threshold": 0.499,  # Within tolerance of 0.5
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should pass validation (all within 0.01 tolerance)
    assert score == 1.0
    assert metrics["stl_param_violations"] == 0

    # Check errors are within tolerance
    assert metrics["stl_scale_xy_error"] < float_tolerance
    assert metrics["stl_scale_z_error"] < float_tolerance
    assert metrics["stl_threshold_error"] < float_tolerance


@pytest.mark.unit
def test_validate_stl_parameters_outside_tolerance():
    """Test validation with parameters outside tolerance."""
    stl_details = {
        "scale_xy": 2.56,  # Outside tolerance of 2.5 (error = 0.06 > 0.05)
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should fail validation
    assert score == 0.0
    assert metrics["stl_param_violations"] == 1
    assert metrics["stl_scale_xy_valid"] is False
    assert metrics["stl_scale_xy_error"] >= 0.05  # Outside tolerance


@pytest.mark.unit
def test_validate_stl_parameters_boolean_mismatch():
    """Test validation with boolean parameter mismatch."""
    stl_details = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": False,  # Mismatch
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,  # Expected True
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should fail validation due to boolean mismatch
    assert score == 0.0
    assert metrics["stl_param_violations"] == 1
    assert metrics["stl_mirror_y_valid"] is False
    assert metrics["stl_mirror_y_error"] == 1.0  # Boolean mismatch


@pytest.mark.unit
def test_validate_stl_parameters_multiple_violations():
    """Test validation with multiple parameter violations."""
    stl_details = {
        "scale_xy": 3.0,  # Expected 2.5, error = 0.5
        "scale_z": 15.0,  # Expected 10.0, error = 5.0
        "threshold": 0.5,
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should fail validation
    assert score == 0.0
    assert metrics["stl_param_violations"] == 2
    assert metrics["stl_scale_xy_valid"] is False
    assert metrics["stl_scale_z_valid"] is False
    assert metrics["stl_threshold_valid"] is True
    assert metrics["stl_mirror_y_valid"] is True


@pytest.mark.unit
def test_validate_stl_parameters_missing_actual_param():
    """Test validation when actual parameter is missing."""
    stl_details = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        # threshold missing
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should fail validation
    assert score == 0.0
    assert metrics["stl_param_violations"] >= 1
    assert metrics["stl_threshold_valid"] is False


@pytest.mark.unit
def test_validate_stl_parameters_missing_expected_param():
    """Test validation when expected parameter is missing."""
    stl_details = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        # threshold missing in expected
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should fail validation for missing expected param
    assert score == 0.0
    assert metrics["stl_param_violations"] >= 1
    assert metrics["stl_threshold_valid"] is False


@pytest.mark.unit
def test_validate_stl_parameters_empty_expected():
    """Test validation with no expected parameters."""
    stl_details = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": True,
    }
    expected_params = {}
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should pass validation (nothing to validate)
    assert score == 1.0
    assert metrics == {}


@pytest.mark.unit
def test_validate_stl_parameters_metrics_structure():
    """Test that validation returns all expected metrics fields."""
    stl_details = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    _, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Check all expected metric keys are present
    for param in ["scale_xy", "scale_z", "threshold", "mirror_y"]:
        assert f"stl_{param}_actual" in metrics
        assert f"stl_{param}_expected" in metrics
        assert f"stl_{param}_error" in metrics
        assert f"stl_{param}_valid" in metrics

    # Check summary metrics
    assert "stl_param_violations" in metrics
    assert "stl_param_validation_score" in metrics


@pytest.mark.unit
def test_validate_stl_parameters_exact_tolerance_boundary():
    """Test validation at exact tolerance boundary."""
    float_tolerance = 0.01

    # Exactly at tolerance boundary (should fail, since error < tolerance, not <=)
    stl_details = {
        "scale_xy": 2.51,  # Error ~= 0.01 (at boundary, may have float precision issues)
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # At exact tolerance (0.01), should pass since error is very close to tolerance
    # Due to floating point precision, 2.51 - 2.5 may not be exactly 0.01
    assert abs(metrics["stl_scale_xy_error"] - float_tolerance) < 1e-10
    # Should pass validation since error < tolerance (barely)
    assert score == 1.0
    assert metrics["stl_scale_xy_valid"] is True


@pytest.mark.unit
def test_validate_stl_parameters_type_conversion():
    """Test that parameter types are correctly converted."""
    # Pass values as different types
    stl_details = {
        "scale_xy": "2.5",  # String
        "scale_z": 10,  # Int
        "threshold": 0.5,
        "mirrored": 1,  # Truthy int
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should pass validation after type conversion
    assert score == 1.0
    assert metrics["stl_param_violations"] == 0

    # Check that values are converted to correct types
    assert isinstance(metrics["stl_scale_xy_actual"], float)
    assert isinstance(metrics["stl_scale_z_actual"], float)
    assert isinstance(metrics["stl_threshold_actual"], float)
    assert isinstance(metrics["stl_mirror_y_actual"], bool)


@pytest.mark.unit
def test_validate_stl_parameters_boolean_false():
    """Test validation with mirror_y=False."""
    stl_details = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": False,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": False,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should pass validation
    assert score == 1.0
    assert metrics["stl_param_violations"] == 0
    assert metrics["stl_mirror_y_valid"] is True
    assert metrics["stl_mirror_y_error"] == 0.0


@pytest.mark.unit
def test_validate_stl_parameters_all_params_wrong():
    """Test validation when all parameters are wrong."""
    stl_details = {
        "scale_xy": 1.0,
        "scale_z": 20.0,
        "threshold": 0.3,
        "mirrored": False,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should fail validation for all parameters
    assert score == 0.0
    assert metrics["stl_param_violations"] == 4
    assert metrics["stl_scale_xy_valid"] is False
    assert metrics["stl_scale_z_valid"] is False
    assert metrics["stl_threshold_valid"] is False
    assert metrics["stl_mirror_y_valid"] is False


@pytest.mark.unit
def test_validate_stl_parameters_negative_values():
    """Test validation with negative values (edge case)."""
    stl_details = {
        "scale_xy": -2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should fail due to sign mismatch (error = 5.0)
    assert score == 0.0
    assert metrics["stl_scale_xy_valid"] is False
    assert metrics["stl_scale_xy_error"] == 5.0


@pytest.mark.unit
def test_validate_stl_parameters_very_close_floats():
    """Test validation with very close float values."""
    stl_details = {
        "scale_xy": 2.500001,  # Very close to 2.5
        "scale_z": 9.999999,  # Very close to 10.0
        "threshold": 0.500001,  # Very close to 0.5
        "mirrored": True,
    }
    expected_params = {
        "scale_xy": 2.5,
        "scale_z": 10.0,
        "threshold": 0.5,
        "mirror_y": True,
    }
    example_id = 0

    score, metrics = _validate_stl_parameters(stl_details, expected_params, example_id)

    # Should pass validation (errors much smaller than 0.01)
    assert score == 1.0
    assert metrics["stl_param_violations"] == 0
    assert metrics["stl_scale_xy_error"] < 0.01
    assert metrics["stl_scale_z_error"] < 0.01
    assert metrics["stl_threshold_error"] < 0.01


# ============================================================================
# WORKFLOW-CONDITIONAL BRANCH RESOLUTION TESTS
# ============================================================================


@pytest.mark.unit
def test_resolve_conditional_params_high_branch():
    """Test that high branch is selected when compliance > threshold."""
    conditional_params = {
        "conditional": True,
        "compliance_threshold": 200.0,
        "branch_high": {"threshold": 0.42, "mirror_y": True},
        "branch_low": {"threshold": 0.61, "mirror_y": False},
        "common": {"scale_xy": 3.14, "scale_z": 11.7},
    }
    target = {"compliance": 250.0}  # > 200.0

    resolved, metrics = _resolve_conditional_params(conditional_params, target, 0)

    assert resolved["threshold"] == 0.42
    assert resolved["mirror_y"] is True
    assert resolved["scale_xy"] == 3.14
    assert resolved["scale_z"] == 11.7
    assert metrics["conditional_correct_branch"] == "high"
    assert metrics["conditional_compliance_threshold"] == 200.0
    assert metrics["conditional_gt_compliance"] == 250.0


@pytest.mark.unit
def test_resolve_conditional_params_low_branch():
    """Test that low branch is selected when compliance <= threshold."""
    conditional_params = {
        "conditional": True,
        "compliance_threshold": 200.0,
        "branch_high": {"threshold": 0.42, "mirror_y": True},
        "branch_low": {"threshold": 0.61, "mirror_y": False},
        "common": {"scale_xy": 3.14, "scale_z": 11.7},
    }
    target = {"compliance": 150.0}  # <= 200.0

    resolved, metrics = _resolve_conditional_params(conditional_params, target, 0)

    assert resolved["threshold"] == 0.61
    assert resolved["mirror_y"] is False
    assert resolved["scale_xy"] == 3.14
    assert resolved["scale_z"] == 11.7
    assert metrics["conditional_correct_branch"] == "low"


@pytest.mark.unit
def test_resolve_conditional_params_at_boundary():
    """Test that compliance == threshold selects low branch (<=)."""
    conditional_params = {
        "conditional": True,
        "compliance_threshold": 200.0,
        "branch_high": {"threshold": 0.42, "mirror_y": True},
        "branch_low": {"threshold": 0.61, "mirror_y": False},
        "common": {"scale_xy": 3.14, "scale_z": 11.7},
    }
    target = {"compliance": 200.0}  # == threshold, should be "low" (<=)

    resolved, metrics = _resolve_conditional_params(conditional_params, target, 0)

    assert metrics["conditional_correct_branch"] == "low"
    assert resolved["threshold"] == 0.61
    assert resolved["mirror_y"] is False


@pytest.mark.unit
def test_resolve_conditional_params_missing_compliance():
    """Test resolution when target has no compliance (defaults to 0.0)."""
    conditional_params = {
        "conditional": True,
        "compliance_threshold": 200.0,
        "branch_high": {"threshold": 0.42, "mirror_y": True},
        "branch_low": {"threshold": 0.61, "mirror_y": False},
        "common": {"scale_xy": 3.14, "scale_z": 11.7},
    }
    target = {}  # No compliance → defaults to 0.0 → low branch

    _resolved, metrics = _resolve_conditional_params(conditional_params, target, 0)

    assert metrics["conditional_correct_branch"] == "low"
    assert metrics["conditional_gt_compliance"] == 0.0


# ============================================================================
# WORKFLOW-MULTI-EXPORT VALIDATION TESTS
# ============================================================================

# Shared fixtures for multi-export tests
_EXPORT_A_EXPECTED = {
    "label": "A",
    "mirror_y": True,
    "scale_xy": 1.50,
    "scale_z": 8.0,
    "threshold": 0.35,
}
_EXPORT_B_EXPECTED = {
    "label": "B",
    "mirror_y": False,
    "scale_xy": 3.20,
    "scale_z": 15.5,
    "threshold": 0.65,
}

_STL_DETAILS_A = {
    "scale_xy": 1.50,
    "scale_z": 8.0,
    "threshold": 0.35,
    "mirrored": True,
}
_STL_DETAILS_B = {
    "scale_xy": 3.20,
    "scale_z": 15.5,
    "threshold": 0.65,
    "mirrored": False,
}


@pytest.mark.unit
def test_validate_multi_export_params_both_valid():
    """Test validation with both exports having correct parameters."""
    score, metrics = _validate_multi_export_params(
        all_stl_details=[_STL_DETAILS_A, _STL_DETAILS_B],
        expected_exports=[_EXPORT_A_EXPECTED, _EXPORT_B_EXPECTED],
        example_id=0,
    )

    assert score == 1.0
    assert metrics["multi_export_both_valid"] is True
    assert metrics["multi_export_count"] == 2
    assert metrics["export_a_stl_param_violations"] == 0
    assert metrics["export_b_stl_param_violations"] == 0


@pytest.mark.unit
def test_validate_multi_export_params_first_invalid():
    """Test validation when first export has wrong parameters."""
    wrong_a = {**_STL_DETAILS_A, "scale_xy": 999.0}

    score, metrics = _validate_multi_export_params(
        all_stl_details=[wrong_a, _STL_DETAILS_B],
        expected_exports=[_EXPORT_A_EXPECTED, _EXPORT_B_EXPECTED],
        example_id=0,
    )

    assert score == 0.0
    assert metrics["multi_export_both_valid"] is False
    assert metrics["export_a_stl_scale_xy_valid"] is False
    assert metrics["export_b_stl_param_violations"] == 0


@pytest.mark.unit
def test_validate_multi_export_params_second_invalid():
    """Test validation when second export has wrong parameters."""
    wrong_b = {**_STL_DETAILS_B, "mirrored": True}  # Should be False

    score, metrics = _validate_multi_export_params(
        all_stl_details=[_STL_DETAILS_A, wrong_b],
        expected_exports=[_EXPORT_A_EXPECTED, _EXPORT_B_EXPECTED],
        example_id=0,
    )

    assert score == 0.0
    assert metrics["multi_export_both_valid"] is False
    assert metrics["export_a_stl_param_violations"] == 0
    assert metrics["export_b_stl_mirror_y_valid"] is False


@pytest.mark.unit
def test_validate_multi_export_params_only_one_call():
    """Test validation when only one STL call was made."""
    score, metrics = _validate_multi_export_params(
        all_stl_details=[_STL_DETAILS_A],
        expected_exports=[_EXPORT_A_EXPECTED, _EXPORT_B_EXPECTED],
        example_id=0,
    )

    assert score == 0.0
    assert metrics["multi_export_both_valid"] is False
    assert metrics["multi_export_count"] == 1


@pytest.mark.unit
def test_validate_multi_export_params_swapped_order():
    """Test validation when parameters are swapped between the two calls."""
    score, metrics = _validate_multi_export_params(
        all_stl_details=[_STL_DETAILS_B, _STL_DETAILS_A],  # Swapped!
        expected_exports=[_EXPORT_A_EXPECTED, _EXPORT_B_EXPECTED],
        example_id=0,
    )

    assert score == 0.0
    assert metrics["multi_export_both_valid"] is False
