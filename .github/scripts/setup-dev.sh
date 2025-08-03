#!/bin/bash
# Development environment setup script for ops0

set -e

echo "🚀 Setting up ops0 development environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ is required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "📚 Installing development dependencies..."
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Run initial checks
echo "✅ Running initial checks..."
python -c "import ops0; print(f'ops0 version: {ops0.__version__}')"

# Create example configuration
echo "📄 Creating example configuration..."
mkdir -p .ops0
cat > .ops0/config.toml << EOF
[project]
name = "dev-project"
version = "0.1.0"

[development]
log_level = "DEBUG"
auto_reload = true
EOF

echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo "2. Run tests: pytest"
echo "3. Start developing: ops0 --help"
echo ""
echo "Happy coding! 🚀"