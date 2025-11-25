#!/bin/bash
# Script to create release archives for deployment
# Generates zip files with version numbers from the codebase

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Creating release archives...${NC}"

# Get version from pyproject.toml
VERSION=$(grep -E "^version = " pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo -e "${GREEN}Version detected: ${VERSION}${NC}"

# Create releases directory if it doesn't exist
RELEASE_DIR="releases"
mkdir -p "$RELEASE_DIR"

# Archive name with version
MAIN_ARCHIVE="engineer-assistant-v${VERSION}.zip"
PRUSA_ARCHIVE="prusa-mcp-server-v${VERSION}.zip"

echo -e "${BLUE}Creating main application archive...${NC}"

# Create main application archive (exclude unnecessary files)
zip -r "${RELEASE_DIR}/${MAIN_ARCHIVE}" . \
    -x "*.git/*" \
    -x "*__pycache__/*" \
    -x "*.pytest_cache/*" \
    -x "*node_modules/*" \
    -x "*.DS_Store" \
    -x "*releases/*" \
    -x "*.egg-info/*" \
    -x "*build/*" \
    -x "*dist/*" \
    -x "*.mypy_cache/*" \
    -x "*.ruff_cache/*" \
    -x "*docs/build/*" \
    -x "*data/*.db" \
    -x "*data/*.sqlite" \
    -x "*data/*.json" \
    -x "*.env" \
    -x "*.env.local"

echo -e "${GREEN}✓ Created: ${RELEASE_DIR}/${MAIN_ARCHIVE}${NC}"

# Check if prusa-mcp directory exists
PRUSA_MCP_DIR="${HOME}/Desktop/prusa-mcp"
if [ -d "$PRUSA_MCP_DIR" ]; then
    echo -e "${BLUE}Creating Prusa MCP server archive...${NC}"

    # Create Prusa MCP archive
    (cd "$PRUSA_MCP_DIR/.." && zip -r \
        "${OLDPWD}/${RELEASE_DIR}/${PRUSA_ARCHIVE}" \
        "prusa-mcp" \
        -x "*prusa-mcp/.git/*" \
        -x "*prusa-mcp/__pycache__/*" \
        -x "*prusa-mcp/.pytest_cache/*" \
        -x "*prusa-mcp/*.DS_Store" \
        -x "*prusa-mcp/.env" \
        -x "*prusa-mcp/.env.local")

    echo -e "${GREEN}✓ Created: ${RELEASE_DIR}/${PRUSA_ARCHIVE}${NC}"
else
    echo -e "${BLUE}Prusa MCP directory not found at ${PRUSA_MCP_DIR}, skipping...${NC}"
fi

# Calculate file sizes
MAIN_SIZE=$(du -h "${RELEASE_DIR}/${MAIN_ARCHIVE}" | cut -f1)
echo -e "${GREEN}Main archive size: ${MAIN_SIZE}${NC}"

if [ -f "${RELEASE_DIR}/${PRUSA_ARCHIVE}" ]; then
    PRUSA_SIZE=$(du -h "${RELEASE_DIR}/${PRUSA_ARCHIVE}" | cut -f1)
    echo -e "${GREEN}Prusa MCP archive size: ${PRUSA_SIZE}${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Release archives created successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Archives location: ${BLUE}${RELEASE_DIR}/${NC}"
echo -e "  - ${MAIN_ARCHIVE}"
if [ -f "${RELEASE_DIR}/${PRUSA_ARCHIVE}" ]; then
    echo -e "  - ${PRUSA_ARCHIVE}"
fi
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Transfer these archives to your server"
echo "2. Follow the deployment instructions in docs/source/deployment.md"
echo ""
