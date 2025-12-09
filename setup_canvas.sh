#!/bin/bash
# Quick setup script for Excalidraw whiteboard component

echo "🎨 Excalidraw Whiteboard Setup"
echo "=============================="
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found!"
    echo "📦 Please install Node.js first:"
    echo "   brew install node"
    echo ""
    exit 1
fi

echo "✅ Node.js found: $(node --version)"
echo "✅ npm found: $(npm --version)"
echo ""

# Navigate to component directory
COMPONENT_DIR="$(dirname "$0")/src/ui/components/excalidraw/frontend"

if [ ! -d "$COMPONENT_DIR" ]; then
    echo "❌ Component directory not found!"
    exit 1
fi

cd "$COMPONENT_DIR"

# Install dependencies and build
echo "📦 Installing dependencies..."
npm install

echo ""
echo "🔨 Building Excalidraw component..."
npm run build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup complete!"
    echo "🚀 You can now run: streamlit run src/ui/streamlit_app.py"
    echo ""
    echo "💡 The whiteboard will now work:"
    echo "   1. Draw on the whiteboard"
    echo "   2. Click 'Send to Chat'"
    echo "   3. Image appears in chat instantly!"
else
    echo ""
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi
