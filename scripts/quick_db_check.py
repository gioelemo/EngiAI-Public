#!/usr/bin/env python3
"""
Quick ChromaDB Check

Simple script to quickly inspect your ChromaDB contents.
Run with: python scripts/quick_db_check.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import EngineerRAGStore

# Constants
MAX_SOURCES_TO_DISPLAY = 10


def main():
    print("\n" + "=" * 60)
    print("  ChromaDB Quick Inspection")
    print("=" * 60 + "\n")

    try:
        # Connect to vector store
        vector_store = EngineerRAGStore(collection_name="engineer_docs")

        # Get count
        count = vector_store.get_collection_count()
        print(f"Total documents in 'engineer_docs': {count}\n")

        if count == 0:
            print("No documents found in the database.")
            return

        # Get all documents
        collection = vector_store.vectorstore._collection
        results = collection.get(include=["metadatas"], limit=100)

        # Analyze sources
        sources = {}
        source_types = {}

        for metadata in results["metadatas"]:
            source = metadata.get("source", "unknown")
            source_type = metadata.get("source_type", "unknown")

            # Count by source
            if source not in sources:
                sources[source] = 0
            sources[source] += 1

            # Count by source type
            if source_type not in source_types:
                source_types[source_type] = 0
            source_types[source_type] += 1

        # Print source types
        print("Documents by source type:")
        for source_type, doc_count in sorted(source_types.items()):
            print(f"  {source_type}: {doc_count} chunks")

        # Print unique sources
        print(f"\nUnique source files: {len(sources)}")
        print("\nRecent sources:")
        for i, (source, doc_count) in enumerate(
            list(sources.items())[:MAX_SOURCES_TO_DISPLAY], 1
        ):
            print(f"  {i}. {source} ({doc_count} chunks)")

        if len(sources) > MAX_SOURCES_TO_DISPLAY:
            print(f"  ... and {len(sources) - MAX_SOURCES_TO_DISPLAY} more\n")

        print("\nFor more detailed inspection, use:")
        print("  python scripts/inspect_chromadb.py --collection engineer_docs --list")
        print()

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure ChromaDB is initialized and accessible.")


if __name__ == "__main__":
    main()
