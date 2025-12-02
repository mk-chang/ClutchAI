#!/bin/bash
# Setup script for ClutchAI conda environment

set -e  # Exit on error

echo "🚀 Setting up ClutchAI conda environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Check if environment already exists
if conda env list | grep -q "^ClutchAI "; then
    echo "⚠️  Environment 'ClutchAI' already exists"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Updating existing environment..."
        conda env update -f environment.yml --prune
    else
        echo "Skipping environment creation."
        exit 0
    fi
else
    echo "📦 Creating conda environment from environment.yml..."
    conda env create -f environment.yml
fi

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   conda activate ClutchAI"
echo ""
echo "2. Verify pytest is installed:"
echo "   pytest --version"
echo ""
echo "3. Run tests:"
echo "   pytest tests/ -v"
echo ""
echo "4. Configure Cursor to use this Python interpreter:"
echo "   Press Cmd+Shift+P (or Ctrl+Shift+P) and search for 'Python: Select Interpreter'"
echo "   Choose: $(conda env list | grep ClutchAI | awk '{print $2}')/bin/python"
echo ""
echo "For detailed instructions, see: docs/CONDA_PYTEST_SETUP.md"

