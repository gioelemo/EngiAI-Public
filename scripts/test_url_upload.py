#!/usr/bin/env python3
"""
Integration test script for URL upload functionality.

This script tests the URL upload process end-to-end with a real (or mock) MMORE instance.
Run this to verify the upload_url_to_mmore.py script works correctly.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.upload_url_to_mmore import (  # noqa: E402
    download_content,
    get_github_raw_url,
    upload_url_to_mmore,
)
from src.tools import MMOREClient  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_github_url_conversion():
    """Test GitHub URL conversion."""
    logger.info("=" * 60)
    logger.info("Test 1: GitHub URL Conversion")
    logger.info("=" * 60)

    test_cases = [
        (
            "https://github.com/swiss-ai/mmore/blob/master/docs/process.md",
            "https://raw.githubusercontent.com/swiss-ai/mmore/master/docs/process.md",
        ),
        (
            "https://github.com/user/repo/blob/main/README.md",
            "https://raw.githubusercontent.com/user/repo/main/README.md",
        ),
        ("https://example.com/page.html", "https://example.com/page.html"),
    ]

    all_passed = True
    for input_url, expected_url in test_cases:
        result = get_github_raw_url(input_url)
        passed = result == expected_url
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {input_url[:50]}...")
        if not passed:
            logger.error(f"  Expected: {expected_url}")
            logger.error(f"  Got: {result}")

    return all_passed


def test_mmore_connection():
    """Test MMORE service connection."""
    logger.info("=" * 60)
    logger.info("Test 2: MMORE Service Connection")
    logger.info("=" * 60)

    try:
        client = MMOREClient()
        is_healthy = client.health_check()

        if is_healthy:
            logger.info("✓ PASS: MMORE service is healthy and reachable")
            return True
        else:
            logger.warning("✗ FAIL: MMORE service is not responding")
            logger.warning("  Make sure MMORE is running at: " + client.base_url)
            return False

    except Exception:
        logger.exception("✗ FAIL: Error connecting to MMORE")
        return False


def test_download_content():
    """Test downloading content from a URL."""
    logger.info("=" * 60)
    logger.info("Test 3: Content Download")
    logger.info("=" * 60)

    # Use a simple, reliable test URL
    test_url = "https://raw.githubusercontent.com/swiss-ai/mmore/master/README.md"
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_file = temp_dir / "test_download.md"

    try:
        success = download_content(test_url, output_file)

        if success and output_file.exists():
            file_size = output_file.stat().st_size
            logger.info(f"✓ PASS: Downloaded {file_size} bytes to {output_file}")

            # Clean up
            output_file.unlink()
            return True
        else:
            logger.error("✗ FAIL: Download failed or file not created")
            return False

    except Exception:
        logger.exception("✗ FAIL: Download error")
        return False


def test_full_upload_workflow():
    """Test the complete upload workflow."""
    logger.info("=" * 60)
    logger.info("Test 4: Full Upload Workflow (Dry Run)")
    logger.info("=" * 60)

    # Use a small test file from GitHub
    test_url = "https://github.com/swiss-ai/mmore/blob/master/README.md"

    logger.info("This test will upload a file to MMORE.")
    logger.info(f"Test URL: {test_url}")

    try:
        # Check if MMORE is available first
        client = MMOREClient()
        if not client.health_check():
            logger.warning("✗ SKIP: MMORE service not available")
            logger.warning("  Start MMORE to run this test")
            return None  # Skip, not fail

        logger.info("Attempting upload...")
        success = upload_url_to_mmore(
            url=test_url, file_id="test_mmore_readme", keep_file=False
        )

        if success:
            logger.info("✓ PASS: Successfully uploaded test file to MMORE")
            logger.info("  You can now query this document through the RAG agent")
            logger.info('  Try asking: "What is MMORE?" in your Streamlit interface')
            return True
        else:
            logger.error("✗ FAIL: Upload workflow failed")
            return False

    except Exception as e:
        logger.error(f"✗ FAIL: Upload error: {e}", exc_info=True)
        return False


def run_all_tests():
    """Run all integration tests."""
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("URL Upload Integration Tests")
    logger.info("=" * 60)
    logger.info("\n")

    results = {
        "GitHub URL Conversion": test_github_url_conversion(),
        "MMORE Connection": test_mmore_connection(),
        "Content Download": test_download_content(),
        "Full Upload Workflow": test_full_upload_workflow(),
    }

    # Print summary
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    total = len(results)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        logger.info(f"{status}: {test_name}")

    logger.info("-" * 60)
    logger.info(
        f"Results: {passed} passed, {failed} failed, {skipped} skipped out of {total}"
    )
    logger.info("=" * 60)

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
