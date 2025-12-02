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
from src.ui.database import DatabaseManager

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


def list_documents(limit: int = 0, show_metadata: bool = False):
    """List all documents tracked in the database."""
    print_section("Documents in MMORE Knowledge Base")

    try:
        db = DatabaseManager()
        documents = db.get_all_mmore_documents()

        if not documents:
            print("\nNo documents found in MMORE knowledge base.")
            print("Use the RAG agent's 'add_document' tool to upload files.")
            return []

        # Apply limit if specified
        display_docs = documents if limit == 0 else documents[:limit]

        print(f"\nTotal documents: {len(documents)}")
        if limit > 0 and len(documents) > limit:
            print(f"Showing first {limit} documents:\n")
        else:
            print()

        for i, doc in enumerate(display_docs, 1):
            file_id = doc.get("file_id", "unknown")
            file_name = doc.get("file_name", "unknown")
            uploaded_at = doc.get("uploaded_at")
            file_path = doc.get("file_path")

            print(f"{i}. File ID: {file_id}")
            print(f"   Name: {file_name}")
            if uploaded_at:
                print(f"   Uploaded: {uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if file_path:
                print(f"   Path: {file_path}")

            if show_metadata:
                metadata = {
                    k: v
                    for k, v in doc.items()
                    if k not in ["file_id", "file_name", "uploaded_at", "file_path"]
                }
                if metadata:
                    print("   Additional Metadata:")
                    print(format_metadata(metadata))
            print()

    except Exception as e:
        print(f"\nError listing documents: {e}")
        return []
    else:
        return documents


def show_document_details(file_id: str, mmore_url: str | None = None):
    """Show detailed information about a specific document."""
    print_section(f"Document Details: {file_id}")

    try:
        # Get from database
        db = DatabaseManager()
        doc_info = db.get_mmore_document(file_id)

        if not doc_info:
            print(f"\nDocument '{file_id}' not found in database.")
            return

        print("\nDatabase Information:")
        print(f"  File ID: {doc_info.get('file_id')}")
        print(f"  File Name: {doc_info.get('file_name')}")
        uploaded_at = doc_info.get("uploaded_at")
        if uploaded_at:
            print(f"  Uploaded: {uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}")
        file_path = doc_info.get("file_path")
        if file_path:
            print(f"  Path: {file_path}")

        # Try to retrieve sample content from MMORE
        print("\nTrying to retrieve sample content from MMORE...")
        client = MMOREClient(base_url=mmore_url)

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


def list_unique_sources():
    """List unique source files tracked in database."""
    print_section("Unique Source Files")

    try:
        db = DatabaseManager()
        documents = db.get_all_mmore_documents()

        if not documents:
            print("\nNo documents found.")
            return

        # Extract unique file names and paths
        file_names = set()
        file_ids = set()

        for doc in documents:
            file_name = doc.get("file_name")
            file_id = doc.get("file_id")

            if file_name:
                file_names.add(file_name)
            if file_id:
                file_ids.add(file_id)

        print(f"\nTotal unique files: {len(file_ids)}")
        print(f"Total unique filenames: {len(file_names)}")

        print("\nFile IDs:")
        for i, file_id in enumerate(sorted(file_ids), 1):
            print(f"  {i}. {file_id}")

        if len(file_names) != len(file_ids):
            print("\nFilenames:")
            for i, file_name in enumerate(sorted(file_names), 1):
                print(f"  {i}. {file_name}")

    except Exception as e:
        print(f"\nError listing sources: {e}")


def export_to_json(output_file: str):
    """Export all document metadata to a JSON file."""
    print_section("Export to JSON")

    try:
        db = DatabaseManager()
        documents = db.get_all_mmore_documents()

        if not documents:
            print("\nNo documents to export.")
            return

        # Convert datetime objects to strings for JSON serialization
        export_data = {
            "total_documents": len(documents),
            "documents": [
                {
                    "file_id": doc.get("file_id"),
                    "file_name": doc.get("file_name"),
                    "file_path": doc.get("file_path"),
                    "uploaded_at": (
                        uploaded_at.isoformat()
                        if (uploaded_at := doc.get("uploaded_at"))
                        else None
                    ),
                }
                for doc in documents
            ],
        }

        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Exported {len(documents)} documents to: {output_file}")

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

  # List documents with full metadata
  python scripts/inspect_mmore.py --list --show-metadata

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
        "--show-metadata",
        action="store_true",
        help="Show full metadata when listing documents",
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
        list_documents(limit=args.limit, show_metadata=args.show_metadata)
        return

    # Execute requested actions
    if args.list:
        list_documents(limit=args.limit, show_metadata=args.show_metadata)
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
        list_unique_sources()
        print()

    if args.export:
        export_to_json(args.export)
        print()


if __name__ == "__main__":
    main()
