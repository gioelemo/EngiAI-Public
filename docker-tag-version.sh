#!/bin/bash
# Script to tag and save current Docker images with version/date

VERSION=${1:-$(date +%Y%m%d-%H%M%S)}

echo "Tagging current images with version: $VERSION"

# Get current image IDs
CHATBOT_IMAGE=$(docker images engineer-assistant-chatbot:latest -q)
MCP_IMAGE=$(docker images engineer-assistant-prusa-mcp-server:latest -q)

if [ -n "$CHATBOT_IMAGE" ]; then
    docker tag $CHATBOT_IMAGE engineer-assistant-chatbot:$VERSION
    echo "✅ Tagged chatbot image: engineer-assistant-chatbot:$VERSION"
fi

if [ -n "$MCP_IMAGE" ]; then
    docker tag $MCP_IMAGE engineer-assistant-prusa-mcp-server:$VERSION
    echo "✅ Tagged MCP server image: engineer-assistant-prusa-mcp-server:$VERSION"
fi

echo ""
echo "Current images:"
docker images | grep -E "engineer-assistant-chatbot|prusa-mcp-server"
