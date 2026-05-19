#!/bin/bash
# Wrapper script to handle optional papers volume mount

# Check if PAPERS_SOURCE_DIR is set and if the path exists
if [ -n "$PAPERS_SOURCE_DIR" ] && [ ! -d "$PAPERS_SOURCE_DIR" ]; then
    echo "⚠️  WARNING: PAPERS_SOURCE_DIR is set to '$PAPERS_SOURCE_DIR' but path does not exist."
    echo "⚠️  Switching to local fallback directory './papers'"
    export PAPERS_SOURCE_DIR="./papers"
fi

# If PAPERS_SOURCE_DIR is not set, use local fallback
if [ -z "$PAPERS_SOURCE_DIR" ]; then
    echo "ℹ️  PAPERS_SOURCE_DIR not set, using local fallback './papers'"
    export PAPERS_SOURCE_DIR="./papers"
fi

# Ensure local papers directory exists
mkdir -p ./papers

echo "📁 Using papers directory: $PAPERS_SOURCE_DIR"

# Run docker-compose with all arguments passed to this script
exec docker compose "$@"
