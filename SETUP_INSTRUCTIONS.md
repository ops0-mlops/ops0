# 🚀 ops0 Setup Instructions

## Quick Installation Steps

### 1. **Clone and Prepare Environment**

```bash
# Clone repository
git clone https://github.com/ops0-mlops/ops0.git
cd ops0

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Update pip
pip install --upgrade pip
```

### 2. **Install Dependencies**

```bash
# Development mode installation with all dependencies
pip install -e ".[dev,ml,cloud,monitoring]"

# Install development tools
pip install \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    ruff>=0.1.0 \
    mypy>=1.5.0 \
    pre-commit>=3.0.0

# Install pre-commit hooks
pre-commit install
```

### 3. **Validate Installation**

```bash
# Verify ops0 imports correctly
python -c "import ops0; print(f'ops0 version: {ops0.__version__}')"

# Run complete validation script
python scripts/validate_installation.py

# Test with development example
python examples/dev/test_pipeline.py
```

### 4. **First Tests**

```bash
# Run unit tests
pytest tests/unit/ -v

# Tests with coverage
pytest --cov=ops0 --cov-report=html

# Check code quality
make check  # or manually:
black --check src/
ruff check src/
mypy src/
```

## Using Makefile (Recommended)

```bash
# Complete setup
make setup

# Quick development cycle
make quick  # format + lint + test-unit

# Full cycle
make full   # format + lint + test + build + docs

# Specific tests
make test-unit
make test-integration
make test-cov

# Code quality
make format
make lint
make quality
```

## Created File Structure

```
ops0/
├── 📄 Makefile                      # Development commands
├── 📄 .gitignore                    # Git exclusions
├── 📄 .pre-commit-config.yaml       # Pre-commit hooks
├── 🔧 scripts/
│   └── validate_installation.py     # Validation
├── 🧪 tests/
│   ├── conftest.py                  # Pytest configuration
│   └── unit/
│       ├── test_decorators.py       # Decorator tests
│       └── test_storage.py          # Storage tests
├── 📚 examples/dev/
│   └── test_pipeline.py             # Test pipeline
└── 📖 SETUP_INSTRUCTIONS.md         # This file
```

## Recommended Development Workflow

### 1. **Daily Development**
```bash
# Activate environment
source venv/bin/activate

# Quick cycle during development
make quick

# Commit with automatic pre-commit
git add .
git commit -m "feat: new feature"
```

### 2. **Before Each Commit**
```bash
# Complete verification
make check

# Full tests
make test

# If everything is OK
git commit -m "fix: fix bug X"
```

### 3. **Adding New Features**
```bash
# Create branch
git checkout -b feature/my-new-feature

# Develop with TDD
# 1. Write failing test
# 2. Implement minimum to pass test  
# 3. Refactor
# 4. Repeat

# Automatic tests
make test-watch  # Re-run tests on changes

# Before pushing
make ci  # Simulate CI locally
```

## File Development Order

### **Phase 1: Core Framework (✅ Done)**
- [x] `src/ops0/__init__.py` - Entry point
- [x] `src/ops0/core/decorators.py` - @step/@pipeline decorators
- [x] `src/ops0/core/storage.py` - Storage layer
- [x] `src/ops0/core/graph.py` - Pipeline graph
- [x] `src/ops0/core/executor.py` - Local executor

### **Phase 2: Tests and CLI (🔲 To Do)**
- [ ] `src/ops0/cli/main.py` - Main CLI
- [ ] `tests/unit/test_graph.py` - Graph tests
- [ ] `tests/unit/test_executor.py` - Executor tests
- [ ] `tests/integration/test_pipeline_execution.py` - Integration tests

### **Phase 3: Advanced Runtime (🔲 To Do)**
- [ ] `src/ops0/runtime/containers.py` - Finalize containerization
- [ ] `src/ops0/runtime/monitoring.py` - Monitoring
- [ ] `src/ops0/integrations/sklearn.py` - ML integration

### **Phase 4: Production (🔲 To Do)**
- [ ] `src/ops0/cloud/` - Cloud deployment
- [ ] Complete documentation
- [ ] Advanced examples

## Debug Commands

```bash
# Debug with verbose
pytest -v -s tests/unit/test_decorators.py

# Debug with PDB
pytest --pdb tests/unit/test_decorators.py

# Debug ops0 development mode
export OPS0_LOG_LEVEL=DEBUG
export OPS0_ENV=development
python examples/dev/test_pipeline.py

# Profiling
python -m cProfile -o profile.stats examples/dev/test_pipeline.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## Troubleshooting

### ops0 Import Error
```bash
# Reinstall in dev mode
pip install -e .

# Check PYTHONPATH
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Failing Tests
```bash
# Clean and reinstall
make clean
make setup
make test
```

### Pre-commit Failures
```bash
# Force run corrections
pre-commit run --all-files

# Then commit
git add .
git commit -m "fix: pre-commit corrections"
```

## 🎯 Ready to Develop!

After these steps, you should have:

✅ Python environment configured  
✅ ops0 installed in development mode  
✅ Tests passing  
✅ Pre-commit hooks active  
✅ Functional CLI  
✅ Working example pipeline  

**Next steps:**
1. `make quick` to verify everything works
2. Look at `examples/dev/test_pipeline.py` to understand the API
3. Read `CONTRIBUTING.md` for conventions
4. Start developing! 🐍⚡

**Support:**
- Discord: https://discord.gg/ops0
- Issues: https://github.com/ops0-mlops/ops0/issues
- Docs: https://docs.ops0.xyz