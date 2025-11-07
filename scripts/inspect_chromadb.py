#!/usr/bin/env python3
"""
ChromaDB Database Inspector

This script allows you to inspect the contents of your ChromaDB vector database,
including collections, documents, and metadata.
"""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import EngineerRAGStore

# Constants
CONTENT_PREVIEW_LENGTH = 200
MAX_SOURCES_TO_DISPLAY = 10


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


def list_collections():
    """List all available collections in ChromaDB."""
    print_section("Available Collections")

    try:
        vector_store = EngineerRAGStore()
        collections = vector_store.list_collections()

        if not collections:
            print("No collections found.")
            return []

        print(f"\nFound {len(collections)} collection(s):\n")
        for i, collection in enumerate(collections, 1):
            print(f"  {i}. {collection}")

    except Exception as e:
        print(f"Error listing collections: {e}")
        return []
    else:
        return collections


def show_collection_stats(collection_name: str):
    """Show statistics for a collection."""
    print_section(f"Collection Statistics: {collection_name}")

    try:
        vector_store = EngineerRAGStore(collection_name=collection_name)
        count = vector_store.get_collection_count()

        print(f"\nTotal documents: {count}")

    except Exception as e:
        print(f"Error getting collection stats: {e}")


def list_documents(
    collection_name: str, limit: int = 10, offset: int = 0, show_content: bool = False
):
    """List documents in a collection."""
    print_section(f"Documents in Collection: {collection_name}")

    try:
        vector_store = EngineerRAGStore(collection_name=collection_name)

        # Get all documents (ChromaDB get() method)
        # Note: This accesses the underlying chromadb collection directly
        collection = vector_store.vectorstore._collection

        # Get documents with limit and offset
        results = collection.get(
            limit=limit if limit > 0 else None,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        if not results["ids"]:
            print("\nNo documents found.")
            return

        # Ensure we have valid data from ChromaDB
        ids = results["ids"]
        documents = results["documents"] or []
        metadatas = results["metadatas"] or []

        total = len(ids)
        print(f"\nShowing {total} document(s) (offset: {offset}):\n")

        for i, (doc_id, doc, metadata) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                strict=True,
            ),
            1,
        ):
            print(f"\n{i}. Document ID: {doc_id}")
            print("   Metadata:")
            print(format_metadata(metadata))

            if show_content:
                preview_text = (
                    doc
                    if len(doc) <= CONTENT_PREVIEW_LENGTH
                    else doc[:CONTENT_PREVIEW_LENGTH] + "..."
                )
                print("   Content Preview:")
                print(f"      {preview_text}")
            else:
                print(f"   Content Length: {len(doc)} characters")

    except Exception as e:
        print(f"Error listing documents: {e}")


def search_by_metadata(
    collection_name: str, filter_dict: dict[str, Any], limit: int = 10
):
    """Search documents by metadata filter."""
    print_section(f"Search by Metadata: {collection_name}")
    print(f"\nFilter: {json.dumps(filter_dict, indent=2)}\n")

    try:
        vector_store = EngineerRAGStore(collection_name=collection_name)
        collection = vector_store.vectorstore._collection

        # Use ChromaDB's where clause for metadata filtering
        results = collection.get(
            where=filter_dict,
            limit=limit if limit > 0 else None,
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            print("No documents found matching the filter.")
            return

        # Ensure we have valid data from ChromaDB
        ids = results["ids"]
        documents = results["documents"] or []
        metadatas = results["metadatas"] or []

        print(f"Found {len(ids)} document(s):\n")

        for i, (doc_id, doc, metadata) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                strict=True,
            ),
            1,
        ):
            print(f"\n{i}. Document ID: {doc_id}")
            print("   Metadata:")
            print(format_metadata(metadata))
            preview_text = (
                doc
                if len(doc) <= CONTENT_PREVIEW_LENGTH
                else doc[:CONTENT_PREVIEW_LENGTH] + "..."
            )
            print("   Content Preview:")
            print(f"      {preview_text}")

    except Exception as e:
        print(f"Error searching documents: {e}")


def list_unique_sources(collection_name: str):
    """List unique source files in the collection."""
    print_section(f"Unique Sources: {collection_name}")

    try:
        vector_store = EngineerRAGStore(collection_name=collection_name)
        collection = vector_store.vectorstore._collection

        # Get all documents
        results = collection.get(include=["metadatas"])

        if not results["metadatas"]:
            print("\nNo documents found.")
            return

        # Extract unique sources
        sources: set[str] = set()
        source_types: dict[str, int] = {}

        for metadata in results["metadatas"]:
            if metadata:
                source_val = metadata.get("source", "unknown")
                source_type_val = metadata.get("source_type", "unknown")

                # Convert to string to ensure sortability
                source = str(source_val) if source_val is not None else "unknown"
                source_type = (
                    str(source_type_val) if source_type_val is not None else "unknown"
                )

                sources.add(source)

                if source_type not in source_types:
                    source_types[source_type] = 0
                source_types[source_type] += 1

        print(f"\nTotal unique sources: {len(sources)}")
        print("\nSource types breakdown:")
        for source_type, count in sorted(source_types.items()):
            print(f"  {source_type}: {count} chunks")

        print("\nAll sources:")
        for i, source in enumerate(sorted(sources), 1):
            print(f"  {i}. {source}")

    except Exception as e:
        print(f"Error listing sources: {e}")


def export_metadata_to_json(collection_name: str, output_file: str):
    """Export all metadata to a JSON file."""
    print_section(f"Exporting Metadata: {collection_name}")

    try:
        vector_store = EngineerRAGStore(collection_name=collection_name)
        collection = vector_store.vectorstore._collection

        # Get all documents
        results = collection.get(include=["metadatas"])

        if not results["metadatas"]:
            print("\nNo documents found.")
            return

        # Create export data
        export_data = {
            "collection_name": collection_name,
            "total_documents": len(results["ids"]),
            "documents": [
                {"id": doc_id, "metadata": metadata}
                for doc_id, metadata in zip(
                    results["ids"], results["metadatas"], strict=True
                )
            ],
        }

        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\nExported {len(results['ids'])} documents to: {output_file}")

    except Exception as e:
        print(f"Error exporting metadata: {e}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Inspect ChromaDB vector database contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all collections
  python scripts/inspect_chromadb.py

  # Show collection statistics
  python scripts/inspect_chromadb.py --collection engineer_docs --stats

  # List first 20 documents
  python scripts/inspect_chromadb.py --collection engineer_docs --list --limit 20

  # List documents with content preview
  python scripts/inspect_chromadb.py --collection engineer_docs --list --show-content

  # List unique source files
  python scripts/inspect_chromadb.py --collection engineer_docs --sources

  # Search by metadata (SMB-imported files)
  python scripts/inspect_chromadb.py --collection engineer_docs --search '{"source_type": "smb"}'

  # Export all metadata to JSON
  python scripts/inspect_chromadb.py --collection engineer_docs --export metadata.json
        """,
    )

    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default="engineer_docs",
        help="Collection name to inspect (default: engineer_docs)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List documents in collection",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of documents to show (default: 10, 0 for all)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for document listing (default: 0)",
    )
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="Show content preview when listing documents",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="List unique source files",
    )
    parser.add_argument(
        "--search",
        type=str,
        help='Search by metadata filter (JSON format, e.g., \'{"source_type": "smb"}\')',
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="OUTPUT_FILE",
        help="Export metadata to JSON file",
    )

    args = parser.parse_args()

    # Always list collections first
    collections = list_collections()
    print()

    # If no specific action requested, exit after showing collections
    if not any([args.stats, args.list, args.sources, args.search, args.export]):
        return

    # Validate collection exists
    if collections and args.collection not in collections:
        print(f"\nWarning: Collection '{args.collection}' not found.")
        print(f"Available collections: {', '.join(collections)}")
        print()

    # Execute requested actions
    if args.stats:
        show_collection_stats(args.collection)
        print()

    if args.list:
        list_documents(
            args.collection,
            limit=args.limit,
            offset=args.offset,
            show_content=args.show_content,
        )
        print()

    if args.sources:
        list_unique_sources(args.collection)
        print()

    if args.search:
        try:
            filter_dict = json.loads(args.search)
            search_by_metadata(args.collection, filter_dict, limit=args.limit)
            print()
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format for search filter: {args.search}")
            print('Example: \'{"source_type": "smb"}\'')

    if args.export:
        export_metadata_to_json(args.collection, args.export)
        print()


if __name__ == "__main__":
    main()
