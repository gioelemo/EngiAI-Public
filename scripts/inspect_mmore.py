#!/usr/bin/env python3
"""
MMORE Database Inspector

This script allows you to inspect the contents of your MMORE knowledge base,
including documents, metadata, and retrieval testing.
"""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import MMOREClient

# Constants
CONTENT_PREVIEW_LENGTH = 200


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title: str):
    """Print a section header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def format_metadata(metadata: Mapping[str, Any] | None, indent: int = 4) -> str:
    """Format metadata dictionary for display."""
    if not metadata:
        return " " * indent + "(no metadata)"

    lines = []
    for key, value in metadata.items():
        lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines)


def check_mmore_health(mmore_url: str | None = None):
    """Check if MMORE service is healthy."""
    print_section("MMORE Service Health Check")

    try:
        client = MMOREClient(base_url=mmore_url)
        is_healthy = client.health_check()

        if is_healthy:
            print("\n✓ MMORE service is healthy")
            print(f"  URL: {client.base_url}")
        else:
            print("\n✗ MMORE service is not responding")
            print(f"  URL: {client.base_url}")
            return False

    except Exception as e:
        print(f"\n✗ Error checking MMORE health: {e}")
        return False
    else:
        return True


def list_documents(limit: int = 0, mmore_url: str | None = None):
    """List all documents in the MMORE knowledge base."""
    print_section("Documents in MMORE Knowledge Base")

    try:
        mmore_client = MMOREClient(base_url=mmore_url)
        file_ids = mmore_client.list_files()

        if not file_ids:
            print("\nNo documents found in MMORE knowledge base.")
            print("Use the RAG agent's 'add_document' tool to upload files.")
            return []

        # Apply limit if specified
        display_ids = file_ids if limit == 0 else file_ids[:limit]

        print(f"\nTotal documents: {len(file_ids)}")
        if limit > 0 and len(file_ids) > limit:
            print(f"Showing first {limit} documents:\n")
        else:
            print()

        for i, file_id in enumerate(display_ids, 1):
            print(f"{i}. File ID: {file_id}")
            print()

    except Exception as e:
        print(f"\nError listing documents: {e}")
        return []
    else:
        return file_ids


def show_document_details(file_id: str, mmore_url: str | None = None):
    """Show detailed information about a specific document."""
    print_section(f"Document Details: {file_id}")

    try:
        client = MMOREClient(base_url=mmore_url)

        print(f"\nFile ID: {file_id}")

        # Try to retrieve sample content from MMORE
        print("\nRetrieving sample content from MMORE...")

        # Use the file_id to retrieve some content
        docs = client.retrieve(
            query="summary overview",
            file_ids=[file_id],
            max_matches=3,
            min_similarity=0.0,
        )

        if docs:
            print(f"\n✓ Found {len(docs)} chunks in MMORE")
            print("\nSample chunks:")
            for i, doc in enumerate(docs, 1):
                content = doc.page_content
                preview = (
                    content
                    if len(content) <= CONTENT_PREVIEW_LENGTH
                    else content[:CONTENT_PREVIEW_LENGTH] + "..."
                )
                score = doc.metadata.get("score", 0.0)
                chunk_id = doc.metadata.get("chunk_id", "unknown")

                print(f"\n  Chunk {i} (ID: {chunk_id}, Score: {score:.3f}):")
                print(f"    {preview}")
        else:
            print("\n⚠ No content found in MMORE (file may not be indexed yet)")

    except Exception as e:
        print(f"\nError getting document details: {e}")


def test_retrieval(
    query: str,
    file_ids: list[str] | None = None,
    mmore_url: str | None = None,
    max_matches: int = 5,
):
    """Test retrieval with a query."""
    print_section(f"Test Retrieval: '{query}'")

    if file_ids:
        print(f"\nRestricted to files: {', '.join(file_ids)}")

    try:
        client = MMOREClient(base_url=mmore_url)
        docs = client.retrieve(
            query=query, file_ids=file_ids, max_matches=max_matches, min_similarity=0.0
        )

        if not docs:
            print("\nNo results found.")
            return

        print(f"\nFound {len(docs)} result(s):\n")

        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            preview = (
                content
                if len(content) <= CONTENT_PREVIEW_LENGTH
                else content[:CONTENT_PREVIEW_LENGTH] + "..."
            )

            file_id = doc.metadata.get("source", "unknown")
            score = doc.metadata.get("score", 0.0)
            chunk_id = doc.metadata.get("chunk_id", "unknown")

            print(f"{i}. File: {file_id} (Score: {score:.3f})")
            print(f"   Chunk ID: {chunk_id}")
            print(f"   Content: {preview}")
            print()

    except Exception as e:
        print(f"\nError during retrieval: {e}")


def list_unique_sources(mmore_url: str | None = None):
    """List unique source files in MMORE."""
    print_section("Unique Source Files")

    try:
        mmore_client = MMOREClient(base_url=mmore_url)
        file_ids = mmore_client.list_files()

        if not file_ids:
            print("\nNo documents found.")
            return

        print(f"\nTotal unique files: {len(file_ids)}")

        print("\nFile IDs:")
        for i, file_id in enumerate(sorted(file_ids), 1):
            print(f"  {i}. {file_id}")

    except Exception as e:
        print(f"\nError listing sources: {e}")


def export_to_json(output_file: str, mmore_url: str | None = None):
    """Export all file IDs to a JSON file."""
    print_section("Export to JSON")

    try:
        mmore_client = MMOREClient(base_url=mmore_url)
        file_ids = mmore_client.list_files()

        if not file_ids:
            print("\nNo documents to export.")
            return

        export_data = {
            "total_documents": len(file_ids),
            "file_ids": file_ids,
        }

        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Exported {len(file_ids)} file IDs to: {output_file}")

    except Exception as e:
        print(f"\nError exporting to JSON: {e}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Inspect MMORE knowledge base contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check MMORE health and list all documents
  python scripts/inspect_mmore.py

  # List all documents
  python scripts/inspect_mmore.py --list

  # Show details for a specific document
  python scripts/inspect_mmore.py --details arxiv_2301.07098v1

  # Test retrieval with a query
  python scripts/inspect_mmore.py --retrieve "topology optimization"

  # Test retrieval on specific files
  python scripts/inspect_mmore.py --retrieve "methods" --file-ids paper1 paper2

  # List all unique sources
  python scripts/inspect_mmore.py --sources

  # Export metadata to JSON
  python scripts/inspect_mmore.py --export mmore_docs.json
        """,
    )

    parser.add_argument(
        "--mmore-url",
        type=str,
        help="MMORE service URL (default: from MMORE_RAG_URL env var or http://localhost:8000)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all documents",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of documents to show (default: 0 for all)",
    )
    parser.add_argument(
        "--details",
        "-d",
        type=str,
        metavar="FILE_ID",
        help="Show detailed information for a specific file",
    )
    parser.add_argument(
        "--retrieve",
        "-r",
        type=str,
        metavar="QUERY",
        help="Test retrieval with a query",
    )
    parser.add_argument(
        "--file-ids",
        nargs="+",
        help="Restrict retrieval to specific file IDs",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=5,
        help="Maximum number of matches for retrieval (default: 5)",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="List unique source files",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="OUTPUT_FILE",
        help="Export metadata to JSON file",
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip MMORE health check",
    )

    args = parser.parse_args()

    # Check MMORE health first (unless skipped)
    if not args.no_health_check:
        is_healthy = check_mmore_health(args.mmore_url)
        print()
        if not is_healthy:
            print("⚠ Warning: MMORE service is not healthy. Some operations may fail.")
            print()

    # If no specific action requested, list documents by default
    if not any([args.list, args.details, args.retrieve, args.sources, args.export]):
        list_documents(limit=args.limit, mmore_url=args.mmore_url)
        return

    # Execute requested actions
    if args.list:
        list_documents(limit=args.limit, mmore_url=args.mmore_url)
        print()

    if args.details:
        show_document_details(args.details, mmore_url=args.mmore_url)
        print()

    if args.retrieve:
        test_retrieval(
            args.retrieve,
            file_ids=args.file_ids,
            mmore_url=args.mmore_url,
            max_matches=args.max_matches,
        )
        print()

    if args.sources:
        list_unique_sources(mmore_url=args.mmore_url)
        print()

    if args.export:
        export_to_json(args.export, mmore_url=args.mmore_url)
        print()


if __name__ == "__main__":
    main()
