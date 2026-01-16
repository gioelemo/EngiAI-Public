"""
Validate generated prompts for beam design benchmarking.

This script checks that generated prompts correctly represent the beam
conditions and are suitable for benchmarking the engineering agent.

Validation checks:
- Numerical accuracy: prompt text matches numerical conditions
- Consistency: descriptions align with parameter values
- Completeness: all required fields are present
- Range validity: parameters are within expected bounds
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import weave

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.weave_integration import init_weave  # noqa: E402

# Expected parameter ranges from dataset analysis
VOLFRAC_MIN = 0.15
VOLFRAC_MAX = 0.40
RMIN_MIN = 1.5
RMIN_MAX = 4.0
FORCEDIST_MIN = 0.0
FORCEDIST_MAX = 1.0
COMPLIANCE_MIN = 13.7
COMPLIANCE_MAX = 975.0

# Consistency check thresholds
FORCE_CONCENTRATED_THRESHOLD = 0.2
FORCE_UNIFORM_THRESHOLD = 0.8
COMPLIANCE_VERY_STIFF_THRESHOLD = 30
COMPLIANCE_FLEXIBLE_THRESHOLD = 200

# Display limits
MAX_FAILED_PROMPTS_TO_SHOW = 5


class ValidationError(Exception):
    """Custom exception for validation errors."""


@weave.op()
def validate_numerical_accuracy(prompt_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that numerical values in prompt text match the conditions.

    Args:
        prompt_data: Single prompt with conditions and text

    Returns:
        Validation result with status and details
    """
    prompt = prompt_data["prompt"]
    conditions = prompt_data["conditions"]
    errors = []

    # Check volume fraction in prompt
    volfrac = conditions["volfrac"]
    volfrac_pattern = rf"{volfrac * 100:.1f}%"
    if volfrac_pattern not in prompt:
        errors.append(f"Volume fraction {volfrac_pattern} not found in prompt text")

    # Check rmin in prompt
    rmin = conditions["rmin"]
    rmin_pattern = rf"{rmin:.1f}"
    if rmin_pattern not in prompt:
        errors.append(f"Minimum radius {rmin_pattern} not found in prompt text")

    # Verify force description is present
    if "Load condition:" not in prompt:
        errors.append("Force/load condition description missing from prompt")

    return {
        "check": "numerical_accuracy",
        "passed": len(errors) == 0,
        "errors": errors,
        "example_id": prompt_data.get("example_id", -1),
    }


@weave.op()
def validate_parameter_ranges(prompt_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that parameters are within expected ranges.

    Args:
        prompt_data: Single prompt with conditions

    Returns:
        Validation result with status and details
    """
    conditions = prompt_data["conditions"]
    errors = []

    # Check volfrac range
    volfrac = conditions["volfrac"]
    if not VOLFRAC_MIN <= volfrac <= VOLFRAC_MAX:
        errors.append(
            f"Volume fraction {volfrac} outside range [{VOLFRAC_MIN}, {VOLFRAC_MAX}]"
        )

    # Check rmin range
    rmin = conditions["rmin"]
    if not RMIN_MIN <= rmin <= RMIN_MAX:
        errors.append(f"Minimum radius {rmin} outside range [{RMIN_MIN}, {RMIN_MAX}]")

    # Check forcedist range
    forcedist = conditions["forcedist"]
    if not FORCEDIST_MIN <= forcedist <= FORCEDIST_MAX:
        errors.append(
            f"Force distribution {forcedist} outside range [{FORCEDIST_MIN}, {FORCEDIST_MAX}]"
        )

    # Check target compliance if present
    if "target" in prompt_data:
        compliance = prompt_data["target"]["compliance"]
        if not COMPLIANCE_MIN <= compliance <= COMPLIANCE_MAX:
            errors.append(
                f"Compliance {compliance} outside range [{COMPLIANCE_MIN}, {COMPLIANCE_MAX}]"
            )

    return {
        "check": "parameter_ranges",
        "passed": len(errors) == 0,
        "errors": errors,
        "example_id": prompt_data.get("example_id", -1),
    }


@weave.op()
def validate_completeness(prompt_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that all required fields are present.

    Args:
        prompt_data: Single prompt data

    Returns:
        Validation result with status and details
    """
    errors: list[str] = []
    required_fields = ["prompt", "conditions", "metadata"]

    # Check top-level fields
    errors.extend(
        f"Missing required field: {field}"
        for field in required_fields
        if field not in prompt_data
    )

    # Check conditions fields
    if "conditions" in prompt_data:
        required_conditions = ["volfrac", "rmin", "forcedist"]
        errors.extend(
            f"Missing required condition: {field}"
            for field in required_conditions
            if field not in prompt_data["conditions"]
        )

    # Check metadata fields
    if "metadata" in prompt_data:
        required_metadata = ["force_description", "expected_stiffness"]
        errors.extend(
            f"Missing required metadata: {field}"
            for field in required_metadata
            if field not in prompt_data["metadata"]
        )

    # Check prompt is not empty
    if "prompt" in prompt_data and not prompt_data["prompt"].strip():
        errors.append("Prompt text is empty")

    return {
        "check": "completeness",
        "passed": len(errors) == 0,
        "errors": errors,
        "example_id": prompt_data.get("example_id", -1),
    }


@weave.op()
def validate_consistency(prompt_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate consistency between different fields.

    Args:
        prompt_data: Single prompt data

    Returns:
        Validation result with status and details
    """
    errors = []
    conditions = prompt_data.get("conditions", {})
    metadata = prompt_data.get("metadata", {})

    # Check force description consistency
    forcedist = conditions.get("forcedist", 0.0)
    force_desc = metadata.get("force_description", "")

    # Check concentrated force consistency
    if (
        forcedist < FORCE_CONCENTRATED_THRESHOLD
        and "concentrated" not in force_desc.lower()
    ):
        errors.append(
            f"Force description '{force_desc}' should mention 'concentrated' for forcedist {forcedist}"
        )

    # Check uniform force consistency
    if forcedist >= FORCE_UNIFORM_THRESHOLD and "uniform" not in force_desc.lower():
        errors.append(
            f"Force description '{force_desc}' should mention 'uniform' for forcedist {forcedist}"
        )

    # Check compliance description consistency if target present
    if "target" in prompt_data:
        compliance = prompt_data["target"]["compliance"]
        stiffness_desc = metadata.get("expected_stiffness", "")

        if (
            compliance < COMPLIANCE_VERY_STIFF_THRESHOLD
            and "very stiff" not in stiffness_desc.lower()
        ):
            errors.append(
                f"Stiffness '{stiffness_desc}' inconsistent with compliance {compliance}"
            )

        if (
            compliance > COMPLIANCE_FLEXIBLE_THRESHOLD
            and "flexible" not in stiffness_desc.lower()
        ):
            errors.append(
                f"Stiffness '{stiffness_desc}' inconsistent with compliance {compliance}"
            )

    return {
        "check": "consistency",
        "passed": len(errors) == 0,
        "errors": errors,
        "example_id": prompt_data.get("example_id", -1),
    }


@weave.op()
def validate_prompt_dataset(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Validate an entire dataset of prompts.

    Args:
        prompts: List of prompt dictionaries

    Returns:
        Comprehensive validation report
    """
    print(f"🔍 Validating {len(prompts)} prompts...")
    print()

    all_results = []
    failed_prompts = []

    for i, prompt_data in enumerate(prompts):
        # Run all validation checks
        results = [
            validate_numerical_accuracy(prompt_data),
            validate_parameter_ranges(prompt_data),
            validate_completeness(prompt_data),
            validate_consistency(prompt_data),
        ]

        all_results.extend(results)

        # Track failed prompts
        failed_results = [r for r in results if not r["passed"]]
        if failed_results:
            failed_prompts.append(
                {
                    "example_id": prompt_data.get("example_id", i),
                    "failures": failed_results,
                }
            )

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Validated {i + 1}/{len(prompts)} prompts...")

    # Compute summary statistics
    total_checks = len(all_results)
    passed_checks = sum(1 for r in all_results if r["passed"])
    failed_checks = total_checks - passed_checks

    summary = {
        "total_prompts": len(prompts),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
        "failed_prompts": failed_prompts,
        "checks_per_prompt": {
            "numerical_accuracy": sum(
                1
                for r in all_results
                if r["check"] == "numerical_accuracy" and r["passed"]
            ),
            "parameter_ranges": sum(
                1
                for r in all_results
                if r["check"] == "parameter_ranges" and r["passed"]
            ),
            "completeness": sum(
                1 for r in all_results if r["check"] == "completeness" and r["passed"]
            ),
            "consistency": sum(
                1 for r in all_results if r["check"] == "consistency" and r["passed"]
            ),
        },
    }

    return summary


def print_validation_report(summary: dict[str, Any]) -> None:
    """Print a human-readable validation report."""
    print()
    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print()
    print(f"Total prompts validated: {summary['total_prompts']}")
    print(f"Total checks performed: {summary['total_checks']}")
    print(f"Passed checks: {summary['passed_checks']} ✅")
    print(f"Failed checks: {summary['failed_checks']} ❌")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print()

    print("Checks breakdown:")
    for check_name, count in summary["checks_per_prompt"].items():
        print(f"  • {check_name}: {count}/{summary['total_prompts']} passed")
    print()

    if summary["failed_prompts"]:
        print("Failed prompts:")
        for failed in summary["failed_prompts"][:MAX_FAILED_PROMPTS_TO_SHOW]:
            print(f"\n  Example {failed['example_id']}:")
            for failure in failed["failures"]:
                print(f"    ❌ {failure['check']}:")
                for error in failure["errors"]:
                    print(f"       - {error}")

        if len(summary["failed_prompts"]) > MAX_FAILED_PROMPTS_TO_SHOW:
            remaining = len(summary["failed_prompts"]) - MAX_FAILED_PROMPTS_TO_SHOW
            print(f"\n  ... and {remaining} more")
    else:
        print("🎉 All prompts passed validation!")


def save_validation_report(summary: dict[str, Any], output_file: Path) -> None:
    """Save validation report to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved validation report to: {output_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate generated beam design prompts"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to validate (default: test)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples in the prompt file (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    print("=" * 60)
    print("BEAM PROMPT VALIDATION")
    print("=" * 60)
    print()
    print(f"Dataset split: {args.split}")
    print(f"Expected samples: {args.samples}")
    print()

    # Initialize Weave
    print("🔧 Initializing Weave...")
    if init_weave():
        print("✅ Weave initialized successfully!")
    else:
        print("⚠️  Weave not available, continuing without tracing...")
    print()

    # Load generated prompts using new naming convention
    input_file = (
        Path(__file__).parent
        / "data"
        / "generated"
        / f"beams2d_prompts_{args.samples}_samples_{args.split}.json"
    )
    print(f"📂 Loading prompts from: {input_file}")

    if not input_file.exists():
        print(f"❌ Error: File not found: {input_file}")
        print(
            f"Please run: python benchmarks/problems/beams2d/generate_prompts.py --split {args.split} --samples {args.samples}"
        )
        return

    with input_file.open() as f:
        prompts = json.load(f)

    print(f"✅ Loaded {len(prompts)} prompts")
    print()

    # Validate prompts
    summary = validate_prompt_dataset(prompts)

    # Print report
    print_validation_report(summary)

    # Save validation report with split and sample info
    output_file = (
        Path(__file__).parent
        / "data"
        / "validated"
        / f"beams2d_validation_report_{args.samples}_samples_{args.split}.json"
    )
    save_validation_report(summary, output_file)

    print()
    print("🎉 Validation complete!")


if __name__ == "__main__":
    main()
