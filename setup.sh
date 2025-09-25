#!/bin/bash
# Setup script for Python Ruff Conda Template
# Run this script to set up the development environment

set -e  # Exit on any error

echo "🚀 Setting up Python Ruff Conda Template..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install miniforge first:"
    echo "   https://github.com/conda-forge/miniforge"
    exit 1
fi

echo "✅ Conda found"

echo "🫙 Creating conda environment..."
conda env create -f environment.yml

echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate python-ruff-template

echo "📦 Installing dev tools..."
pip install .[dev]

echo "🪝 Installing pre-commit hooks..."
pre-commit install

echo "🧪 Testing the setup..."
python src/example.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate python-ruff-template"
echo "2. Configure VS Code:"
echo "   - Open project in VS Code (will prompt for extensions)"
echo "   - Install Ruff and MyPy extensions: charliermarsh.ruff and ms-python.mypy-type-checker"
echo "   - Copy settings: .vscode/settings_template.json at bottom of existing .vscode/settings.json"
echo "3. Start coding! 🚀"
echo ""
echo "Useful commands:"
echo "  ruff check .          # Check for issues"
echo "  ruff check --fix .    # Fix auto-fixable issues"
echo "  ruff format .         # Format code"
echo "  mypy .                # Type checking"
echo "  pre-commit run --all-files  # Run all pre-commit hooks"
