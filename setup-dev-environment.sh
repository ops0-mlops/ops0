#!/bin/bash
# Setup script pour développement ops0

echo "🚀 Setting up ops0 development environment..."

# 1. Create Python virtual environment
echo "📦 Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 2. Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# 3. Installer les dépendances de développement
echo "🔧 Installing development dependencies..."
pip install -e ".[dev,ml,cloud,monitoring]"

# 4. Installer les outils de développement
pip install \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    pytest-mock>=3.11.0 \
    black>=23.0.0 \
    ruff>=0.1.0 \
    mypy>=1.5.0 \
    pre-commit>=3.0.0 \
    ipython>=8.0.0 \
    jupyter>=1.0.0

# 5. Installer pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# 6. Créer les répertoires nécessaires
echo "📁 Creating project directories..."
mkdir -p .ops0/storage
mkdir -p .ops0/cache
mkdir -p logs
mkdir -p build
mkdir -p dist

# 7. Copier les fichiers de configuration
echo "⚙️ Setting up configuration files..."

# Créer .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
.venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/
.cache
nosetests.xml
coverage.xml

# ops0 specific
.ops0/
*.ops0
logs/
.env.local

# Jupyter
.ipynb_checkpoints

# MacOS
.DS_Store

# Docker
Dockerfile.dev
docker-compose.override.yml
EOF

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.284
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF

# 8. Create local environment file
cat > .env.example << 'EOF'
# ops0 Development Configuration
OPS0_ENV=development
OPS0_LOG_LEVEL=DEBUG
OPS0_STORAGE_BACKEND=local
OPS0_STORAGE_PATH=.ops0/storage
OPS0_CONTAINER_REGISTRY=ghcr.io/ops0-mlops
OPS0_BUILD_CONTAINERS=false

# API Configuration (pour tests)
OPS0_API_KEY=dev-key-123
OPS0_PROJECT=ops0-dev
EOF

# 9. Installer le package en mode développement
echo "📦 Installing ops0 in development mode..."
pip install -e .

# 10. Vérifier l'installation
echo "✅ Verifying installation..."
python -c "import ops0; print(f'ops0 version: {ops0.__version__}')"

# 11. Créer un exemple de test
echo "🧪 Creating test example..."
mkdir -p examples/dev
cat > examples/dev/test_pipeline.py << 'EOF'
"""
Simple test pipeline for development verification
"""
import ops0
import pandas as pd

@ops0.step
def generate_data():
    """Generate sample data for testing"""
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [0, 1, 0, 1, 0]
    })
    ops0.storage.save("raw_data", data)
    print(f"✓ Generated {len(data)} rows of test data")
    return data

@ops0.step
def preprocess():
    """Preprocess the test data"""
    data = ops0.storage.load("raw_data")

    # Simple preprocessing
    processed = data.copy()
    processed['feature1_scaled'] = processed['feature1'] / processed['feature1'].max()

    ops0.storage.save("processed_data", processed)
    print(f"✓ Preprocessed data with {len(processed.columns)} features")
    return processed

@ops0.step
def simple_model():
    """Train a simple test model"""
    data = ops0.storage.load("processed_data")

    # Mock training
    accuracy = 0.85
    model_info = {
        "accuracy": accuracy,
        "features": list(data.columns),
        "rows_trained": len(data)
    }

    ops0.storage.save("model_results", model_info)
    print(f"✓ Model trained with {accuracy:.2%} accuracy")
    return model_info

if __name__ == "__main__":
    print("🚀 Running ops0 development test pipeline...")

    with ops0.pipeline("dev-test-pipeline"):
        # Define the pipeline steps
        generate_data()
        preprocess()
        simple_model()

        # Run locally
        print("\n📊 Executing pipeline locally...")
        results = ops0.run(mode="local")
        print(f"\n✅ Pipeline completed! Results: {results}")
EOF

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Test installation: python examples/dev/test_pipeline.py"
echo "   3. Run tests: pytest"
echo "   4. Start developing: code src/ops0/"
echo ""
echo "🔧 Development commands:"
echo "   • ops0 --help          # CLI help"
echo "   • pytest               # Run tests"
echo "   • black src/           # Format code"
echo "   • mypy src/            # Type checking"
echo "   • pre-commit run       # Run all checks"
echo ""