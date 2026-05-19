#!/usr/bin/env python3
"""
Upload web content (HTML/Markdown) to MMORE RAG system.

This script downloads content from a URL and uploads it to MMORE for indexing.
MMORE supports HTML files and will parse them using BeautifulSoup.
"""

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import MMOREClient  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_content(url: str, output_path: Path) -> bool:
    """
    Download content from URL and save to file.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded content

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading content from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save content
        with output_path.open("wb") as f:
            f.write(response.content)

        logger.info(f"Content saved to {output_path}")
    except Exception:
        logger.exception("Failed to download content")
        return False
    else:
        return True


def get_github_raw_url(github_url: str) -> str:
    """
    Convert GitHub web URL to raw content URL.

    Args:
        github_url: GitHub repository URL (e.g., https://github.com/user/repo/blob/main/file.md)

    Returns:
        Raw content URL (e.g., https://raw.githubusercontent.com/user/repo/main/file.md)
    """
    # Convert github.com/user/repo/blob/branch/path to raw.githubusercontent.com/user/repo/branch/path
    if "github.com" in github_url and "/blob/" in github_url:
        return github_url.replace("github.com", "raw.githubusercontent.com").replace(
            "/blob/", "/"
        )
    return github_url


def upload_url_to_mmore(
    url: str, file_id: str | None = None, keep_file: bool = False
) -> bool:
    """
    Download content from URL and upload to MMORE.

    Args:
        url: URL to download and upload
        file_id: Optional custom file ID (defaults to sanitized URL)
        keep_file: If True, keep downloaded file after upload

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize MMORE client
        mmore_client = MMOREClient()

        # Check MMORE health
        if not mmore_client.health_check():
            logger.error("MMORE service is not available")
            return False

        # Convert GitHub URLs to raw URLs if needed
        download_url = get_github_raw_url(url)

        # Determine file extension from URL
        parsed_url = urlparse(download_url)
        path_parts = Path(parsed_url.path)
        extension = path_parts.suffix or ".html"

        # Generate file ID if not provided
        if not file_id:
            # Create a clean file ID from URL
            file_id = path_parts.stem or parsed_url.netloc.replace(
                ".", "_"
            ) + "_" + parsed_url.path.replace("/", "_")
            # Clean up file_id
            file_id = "".join(c for c in file_id if c.isalnum() or c in "_-")

        # Create temporary file path
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"{file_id}{extension}"

        # Download content
        if not download_content(download_url, temp_file):
            return False

        # Upload to MMORE
        logger.info(f"Uploading to MMORE with file_id: {file_id}...")
        result = mmore_client.upload_file(file_path=str(temp_file), file_id=file_id)
        logger.info(f"Successfully uploaded to MMORE: {result}")

        # Clean up temporary file unless keep_file is True
        if not keep_file:
            temp_file.unlink(missing_ok=True)
            logger.info("Cleaned up temporary file")

        logger.info("✓ Successfully added URL content to MMORE knowledge base")

    except Exception:
        logger.exception("Failed to upload URL to MMORE")
        return False
    else:
        return True


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Download web content and upload to MMORE RAG system"
    )

    parser.add_argument("url", type=str, help="URL to download and upload")
    parser.add_argument(
        "--file-id",
        type=str,
        help="Custom file ID (defaults to sanitized URL)",
    )
    parser.add_argument(
        "--keep-file",
        action="store_true",
        help="Keep downloaded file after upload",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Upload URL to MMORE
    success = upload_url_to_mmore(
        url=args.url, file_id=args.file_id, keep_file=args.keep_file
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
