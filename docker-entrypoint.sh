#!/bin/bash
set -e

# If SSH directory is mounted, set up SSH configuration with proper permissions
if [ -d "/root/.ssh-host" ] && [ "$(ls -A /root/.ssh-host 2>/dev/null)" ]; then
    echo "Setting up SSH configuration..."

    # Create .ssh directory if it doesn't exist
    mkdir -p /root/.ssh

    # Copy SSH config and keys from mounted volume (only if files exist)
    cp -r /root/.ssh-host/* /root/.ssh/ 2>/dev/null || true

    # Set proper permissions for SSH
    chmod 700 /root/.ssh

    # Set permissions for all private keys
    find /root/.ssh -type f -name "id_*" ! -name "*.pub" -exec chmod 600 {} \; 2>/dev/null || true

    # Set permissions for config file
    if [ -f /root/.ssh/config ]; then
        chmod 600 /root/.ssh/config
    fi

    # Set permissions for known_hosts
    if [ -f /root/.ssh/known_hosts ]; then
        chmod 644 /root/.ssh/known_hosts
    fi

    echo "SSH configuration complete."
else
    echo "SSH configuration skipped (no SSH keys mounted)."
fi

# Check if papers directory is accessible and has content
if [ -d "/app/papers" ]; then
    PAPER_COUNT=$(find /app/papers -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PAPER_COUNT" -gt 0 ]; then
        echo "Papers directory mounted: Found $PAPER_COUNT paper files."
    else
        echo "⚠️  WARNING: Papers directory is empty. If you're on university network, ensure PAPERS_SOURCE_DIR is set correctly in .env"
        echo "⚠️  Current papers mount: Using fallback local directory. No papers available for import."
    fi
else
    echo "⚠️  WARNING: Papers directory not accessible at /app/papers"
fi

# Execute the main command (Streamlit)
exec "$@"
