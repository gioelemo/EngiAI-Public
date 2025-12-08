#!/bin/bash
# Quick setup script for canvas export feature

echo "🎨 Canvas Export Feature Setup"
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
COMPONENT_DIR="$(dirname "$0")/src/ui/components/canvas_receiver"

if [ ! -d "$COMPONENT_DIR" ]; then
    echo "❌ Component directory not found!"
    exit 1
fi

cd "$COMPONENT_DIR"

# Run the build
echo "🔨 Building custom component..."
./build.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup complete!"
    echo "🚀 You can now run: streamlit run src/main.py"
    echo ""
    echo "💡 The canvas export will now work automatically:"
    echo "   1. Draw on the whiteboard"
    echo "   2. Click 'Send to Chat'"
    echo "   3. Image appears in chat instantly!"
else
    echo ""
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi
