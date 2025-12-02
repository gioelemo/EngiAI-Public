#!/usr/bin/env python3
"""
Local/Mounted Share Paper Import Script for MMORE RAG System

This script imports PDF papers from a local directory or mounted network share
into the MMORE RAG system. It supports incremental updates by tracking which
files have already been imported.

This is simpler than the SMB script when you have the share already mounted
(e.g., at /Volumes/ShareName on macOS or /mnt/share on Linux).

Features:
- Scans local or mounted directory for PDFs
- Uploads PDFs to MMORE service for indexing
- Tracks import state to avoid re-importing
- Handles errors gracefully with detailed logging
- Supports dry-run mode for testing
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Add project root to path (must be before src imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path manipulation (ruff: E402 is acceptable here)
from src.tools import MMOREClient  # noqa: E402
from src.ui.database import DatabaseManager  # noqa: E402

# Configure logging
# Ensure log directory exists
log_file = Path("data/local_import.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# Constants
LARGE_FILE_THRESHOLD_MB = 50  # Warn for files larger than this
MAX_UPLOAD_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds


class LocalPaperImporter:
    """Handles importing PDF papers from local/mounted directory to MMORE RAG system."""

    def __init__(
        self,
        source_dir: str,
        state_file: str = "data/local_import_state.json",
        use_symlinks: bool = False,
    ):
        """
        Initialize Local Paper Importer.

        Args:
            source_dir: Path to directory containing PDFs (can be mounted share)
            state_file: JSON file tracking import state
            use_symlinks: If True, don't copy files, just reference them (unused with MMORE)
        """
        self.source_dir = Path(source_dir)
        if not self.source_dir.exists():
            msg = f"Source directory does not exist: {source_dir}"
            raise ValueError(msg)

        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.use_symlinks = use_symlinks

        # Initialize MMORE components
        self.mmore_client = MMOREClient()
        self.db = DatabaseManager()

        # Import state
        self.imported_files: dict[str, dict] = self._load_state()

    def _load_state(self) -> dict[str, dict]:
        """Load import state from JSON file."""
        if self.state_file.exists():
            try:
                with self.state_file.open("r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not parse state file {self.state_file}, starting fresh"
                )
                return {}
        return {}

    def _save_state(self):
        """Save import state to JSON file."""
        with self.state_file.open("w") as f:
            json.dump(self.imported_files, f, indent=2)
        logger.info(f"Saved import state to {self.state_file}")

    def list_pdf_files(self) -> list[Path]:
        """
        Recursively list all PDF files in source directory.

        Returns:
            List of Path objects for PDF files
        """
        logger.info(f"Scanning for PDF files in {self.source_dir}...")

        pdf_files = list(self.source_dir.rglob("*.pdf"))
        pdf_files.extend(self.source_dir.rglob("*.PDF"))

        # Filter out hidden files
        pdf_files = [
            f for f in pdf_files if not any(part.startswith(".") for part in f.parts)
        ]

        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files

    def get_file_hash(self, file_path: Path) -> str:
        """
        Get a hash/identifier for a file to detect changes.
        Uses file size and modification time.

        Args:
            file_path: Path to file

        Returns:
            String hash of file metadata
        """
        try:
            stat = file_path.stat()
            return f"{stat.st_size}_{int(stat.st_mtime)}"
        except Exception as e:
            logger.warning(f"Could not get file info for {file_path}: {e}")
            return ""

    def filter_new_files(self, all_files: list[Path]) -> list[Path]:
        """
        Filter out files that have already been imported and haven't changed.

        Args:
            all_files: List of all PDF files

        Returns:
            List of new or modified files to import
        """
        new_files = []

        for file_path in all_files:
            # Use relative path as key for portability
            try:
                rel_path = str(file_path.relative_to(self.source_dir))
            except ValueError:
                rel_path = str(file_path)

            file_hash = self.get_file_hash(file_path)

            if rel_path not in self.imported_files:
                logger.info(f"New file: {rel_path}")
                new_files.append(file_path)
            elif self.imported_files[rel_path].get("hash") != file_hash:
                logger.info(f"Modified file: {rel_path}")
                new_files.append(file_path)
            else:
                logger.debug(f"Already imported: {rel_path}")

        logger.info(f"Found {len(new_files)} new or modified files to import")
        return new_files

    def process_and_import(self, files: list[Path]) -> dict[str, bool]:
        """
        Process PDF files and upload them to MMORE.

        Args:
            files: List of file paths to process

        Returns:
            Dictionary mapping file path to success status
        """
        results = {}

        for file_path in files:
            try:
                # Use relative path as key
                try:
                    rel_path = str(file_path.relative_to(self.source_dir))
                except ValueError:
                    rel_path = str(file_path)

                logger.info(f"Processing {rel_path}...")

                # Check file size and warn if large
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > LARGE_FILE_THRESHOLD_MB:
                    logger.warning(
                        f"{rel_path} is {file_size_mb:.1f} MB - may take a while to process"
                    )

                # Generate unique file ID based on relative path
                # Replace path separators with underscores for cleaner IDs
                file_id = (
                    rel_path.replace("/", "_")
                    .replace("\\", "_")
                    .replace(".pdf", "")
                    .replace(".PDF", "")
                )

                # Upload file to MMORE with retry logic
                max_retries = MAX_UPLOAD_RETRIES
                retry_delay = RETRY_BASE_DELAY
                last_error = None

                for attempt in range(max_retries):
                    try:
                        self.mmore_client.upload_file(
                            file_path=str(file_path),
                            file_id=file_id,
                        )
                        break  # Success, exit retry loop
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (
                                2**attempt
                            )  # Exponential backoff
                            logger.warning(
                                f"Upload attempt {attempt + 1} failed for {rel_path}: {e}. "
                                f"Retrying in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            # Final attempt failed
                            raise last_error from e

                logger.info(
                    f"Successfully uploaded {rel_path} to MMORE (fileId: {file_id})"
                )

                # Track in database
                self.db.add_mmore_document(
                    file_id=file_id,
                    file_name=file_path.name,
                    file_path=str(file_path),
                    uploaded_by="local_import_script",
                )

                # Update state
                self.imported_files[rel_path] = {
                    "hash": self.get_file_hash(file_path),
                    "import_date": datetime.now().isoformat(),
                    "file_id": file_id,
                    "full_path": str(file_path),
                }

                # Save state after each successful upload to avoid losing progress
                self._save_state()

                results[rel_path] = True

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                results[rel_path] = False
                # Continue with next file instead of stopping

        return results

    def run(self, dry_run: bool = False, max_files: int | None = None) -> dict:
        """
        Run the import process.

        Args:
            dry_run: If True, only list files without importing
            max_files: Maximum number of files to import (None for all)

        Returns:
            Dictionary with import statistics
        """
        stats = {
            "total_files": 0,
            "new_files": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }

        try:
            # List all PDF files
            all_files = self.list_pdf_files()
            stats["total_files"] = len(all_files)

            # Filter for new/modified files
            new_files = self.filter_new_files(all_files)
            stats["new_files"] = len(new_files)

            if dry_run:
                logger.info("DRY RUN - Would import the following files:")
                for file in new_files:
                    try:
                        rel_path = file.relative_to(self.source_dir)
                    except ValueError:
                        rel_path = file
                    logger.info(f"  - {rel_path}")
                return stats

            # Limit number of files if requested
            if max_files is not None:
                new_files = new_files[:max_files]
                stats["skipped"] = stats["new_files"] - len(new_files)

            # Process and import files
            if new_files:
                results = self.process_and_import(new_files)

                stats["processed"] = len(results)
                stats["successful"] = sum(1 for success in results.values() if success)
                stats["failed"] = sum(1 for success in results.values() if not success)

                # Save state
                self._save_state()

        except Exception as e:
            logger.error(f"Import process failed: {e}", exc_info=True)
            raise
        else:
            return stats


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Import PDF papers from local/mounted directory to MMORE RAG system"
    )

    parser.add_argument(
        "source_dir",
        type=str,
        nargs="?",
        help="Path to directory containing PDFs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/local_import_config.json",
        help="Config file path",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="data/local_import_state.json",
        help="Import state file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without importing",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to import",
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

    # Priority: command-line arg > environment variable > config file
    source_dir = args.source_dir
    state_file = args.state_file

    # Try environment variables if not provided via command line
    if not source_dir:
        source_dir = os.getenv("PAPERS_SOURCE_DIR")

    if state_file == "data/local_import_state.json":  # Default value
        state_file = os.getenv("PAPERS_STATE_FILE", state_file)

    # Fall back to config file if still not found
    config_file = Path(args.config)
    if not source_dir and config_file.exists():
        logger.info(f"Loading configuration from {config_file}")
        with config_file.open("r") as f:
            config = json.load(f)
            source_dir = source_dir or config.get("source_dir")
            state_file = state_file or config.get("state_file", state_file)

    # Validate required parameters
    if not source_dir:
        logger.error(
            "Missing source directory. Provide via:\n"
            "  1. Command-line argument: python import_local_papers.py /path/to/papers\n"
            "  2. Environment variable: PAPERS_SOURCE_DIR in .env file\n"
            "  3. Config file: scripts/local_import_config.json"
        )
        parser.print_help()
        sys.exit(1)

    # Create importer
    importer = LocalPaperImporter(
        source_dir=source_dir,
        state_file=state_file,
    )

    # Run import
    logger.info("Starting local paper import...")
    logger.info(f"Source directory: {source_dir}")
    stats = importer.run(dry_run=args.dry_run, max_files=args.max_files)

    # Print summary
    logger.info("=" * 60)
    logger.info("Import Summary:")
    logger.info(f"  Total files found: {stats['total_files']}")
    logger.info(f"  New/modified files: {stats['new_files']}")
    logger.info(f"  Processed: {stats['processed']}")
    logger.info(f"  Successful: {stats['successful']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info("=" * 60)

    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
