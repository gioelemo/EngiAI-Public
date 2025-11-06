#!/bin/bash
# Script to package Engineer Assistant Docker image for sharing

set -e  # Exit on error

echo "=================================="
echo "Engineer Assistant - Package Script"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

# Configuration
IMAGE_NAME="engineer-assistant-chatbot"
OUTPUT_DIR="./engineer-assistant-deploy"
TAR_FILE="engineer-assistant-chatbot.tar"
TAR_GZ_FILE="engineer-assistant-chatbot.tar.gz"

echo -e "${YELLOW}Step 1: Security Check${NC}"
echo "Checking for secrets in the Docker image..."

# Check for common secret patterns
echo "  - Checking image history..."
# Filter out known safe patterns: GPG_KEY (Python signing), apt-key, ssh-keygen commands, package keys
SECRET_CHECK=$(docker history $IMAGE_NAME --no-trunc 2>/dev/null | grep -iE "api|key|secret|password" | grep -v "apt-key\|gpg-key\|ssh-keygen\|GPG_KEY=\|dpkg-buildflags" || true)
if [ ! -z "$SECRET_CHECK" ]; then
    echo -e "${RED}Warning: Potential secrets found in image history!${NC}"
    echo "$SECRET_CHECK"
    read -p "Continue anyway? (yes/no): " -n 3 -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        exit 1
    fi
fi

# Check environment variables
echo "  - Checking environment variables..."
ENV_CHECK=$(docker inspect $IMAGE_NAME 2>/dev/null | jq '.[0].Config.Env' | grep -iE "key|secret|password|token" || echo "")
if [ ! -z "$ENV_CHECK" ]; then
    echo -e "${RED}Warning: Environment variables with secrets found:${NC}"
    echo "$ENV_CHECK"
    read -p "Continue anyway? (yes/no): " -n 3 -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ Security check complete${NC}"
echo ""

echo -e "${YELLOW}Step 2: Exporting Docker Image${NC}"
echo "This will create a large file (~2-4 GB). It may take several minutes..."

# Remove old export if it exists
if [ -f "$TAR_FILE" ]; then
    echo "  - Removing old export..."
    rm -f "$TAR_FILE"
fi

if [ -f "$TAR_GZ_FILE" ]; then
    echo "  - Removing old compressed export..."
    rm -f "$TAR_GZ_FILE"
fi

# Save the image
echo "  - Saving image to $TAR_FILE..."
docker save -o "$TAR_FILE" $IMAGE_NAME:latest

if [ ! -f "$TAR_FILE" ]; then
    echo -e "${RED}Error: Failed to create TAR file${NC}"
    exit 1
fi

TAR_SIZE=$(du -h "$TAR_FILE" | cut -f1)
echo -e "${GREEN}✓ Image saved ($TAR_SIZE)${NC}"
echo ""

echo -e "${YELLOW}Step 3: Compressing Image${NC}"
echo "Compressing the image to reduce size..."
gzip -k "$TAR_FILE"

if [ ! -f "$TAR_GZ_FILE" ]; then
    echo -e "${RED}Error: Failed to compress TAR file${NC}"
    exit 1
fi

GZ_SIZE=$(du -h "$TAR_GZ_FILE" | cut -f1)
echo -e "${GREEN}✓ Image compressed ($GZ_SIZE)${NC}"
echo ""

echo -e "${YELLOW}Step 4: Creating Deployment Package${NC}"

# Create output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo "  - Removing old deployment directory..."
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# Copy required files
echo "  - Copying configuration files..."
cp docker-compose.yml "$OUTPUT_DIR/"
cp .env.example "$OUTPUT_DIR/"
cp DOCKER_DEPLOYMENT.md "$OUTPUT_DIR/"
cp DOCKER_SHARE_GUIDE.md "$OUTPUT_DIR/"
cp RECIPIENT_SETUP.md "$OUTPUT_DIR/"

# Create a README for the recipient
cat > "$OUTPUT_DIR/README.md" << 'EOF'
# Engineer Assistant - Docker Deployment

## Quick Start

1. **Install Docker** (if not already installed)
   - Windows/Mac: Download Docker Desktop from https://www.docker.com/products/docker-desktop/
   - Linux: Follow instructions at https://docs.docker.com/engine/install/

2. **Load the Docker image**
   ```bash
   # Uncompress the image
   gunzip engineer-assistant-chatbot.tar.gz

   # Load into Docker
   docker load -i engineer-assistant-chatbot.tar
   ```

3. **Configure API keys**
   ```bash
   # Copy the example file
   cp .env.example .env

   # Edit .env and add your API keys:
   # - OPENAI_API_KEY (required)
   # - TAVILY_API_KEY (required)
   ```

4. **Start the application**
   ```bash
   docker-compose up -d
   ```

5. **Access the chatbot**
   - Open your browser to: http://localhost:8501

## Documentation

- **RECIPIENT_SETUP.md** - Detailed setup instructions
- **DOCKER_DEPLOYMENT.md** - Complete deployment guide
- **DOCKER_SHARE_GUIDE.md** - Information about the image

## System Requirements

- Docker installed and running
- 8 GB RAM minimum
- 10 GB free disk space
- Internet connection

## Troubleshooting

See RECIPIENT_SETUP.md for common issues and solutions.

## Support

For issues, check the logs:
```bash
docker-compose logs -f chatbot
```

EOF

echo -e "${GREEN}✓ Deployment package created${NC}"
echo ""

echo -e "${YELLOW}Step 5: Creating ZIP Archive${NC}"

# Create zip of deployment package (excluding the large tar files)
ZIP_FILE="engineer-assistant-deploy.zip"
if [ -f "$ZIP_FILE" ]; then
    rm -f "$ZIP_FILE"
fi

echo "  - Creating $ZIP_FILE..."
zip -r "$ZIP_FILE" "$OUTPUT_DIR"/ > /dev/null

ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo -e "${GREEN}✓ ZIP archive created ($ZIP_SIZE)${NC}"
echo ""

# Summary
echo "=================================="
echo -e "${GREEN}Package Complete!${NC}"
echo "=================================="
echo ""
echo "Files created:"
echo "  1. $TAR_FILE ($TAR_SIZE) - Uncompressed image"
echo "  2. $TAR_GZ_FILE ($GZ_SIZE) - Compressed image (RECOMMENDED TO SHARE)"
echo "  3. $ZIP_FILE ($ZIP_SIZE) - Configuration files"
echo ""
echo "To share with someone:"
echo "  1. Send them $TAR_GZ_FILE (the Docker image)"
echo "  2. Send them $ZIP_FILE (the configuration files)"
echo "  3. Tell them to read RECIPIENT_SETUP.md for instructions"
echo ""
echo "Alternative: Push to Docker Hub"
echo "  1. docker login"
echo "  2. docker tag $IMAGE_NAME:latest YOUR_USERNAME/engineer-assistant:latest"
echo "  3. docker push YOUR_USERNAME/engineer-assistant:latest"
echo ""
echo -e "${YELLOW}Note: Keep $TAR_FILE if you want the uncompressed version, otherwise you can delete it.${NC}"
echo ""
