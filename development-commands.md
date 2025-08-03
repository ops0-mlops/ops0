# 🚀 ops0 Development Commands

## Development Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/ops0-mlops/ops0.git
cd ops0

# 2. Make installation script executable
chmod +x setup-dev-environment.sh

# 3. Run installation script
./setup-dev-environment.sh

# 4. Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# 5. Verify installation
python -c "import ops0; print(f'ops0 version: {ops0.__version__}')"
```

## Daily Development Commands

### 🧪 Testing
```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# Tests with coverage
pytest --cov=ops0 --cov-report=html

# Specific tests
pytest tests/unit/test_decorators.py::TestStepDecorator::test_step_decorator_basic

# Watch mode (re-run on changes)
ptw tests/unit/
```

### 🎨 Code Quality
```bash
# Code formatting
black src/ops0/

# Import sorting
isort src/ops0/

# Linting with ruff
ruff check src/ops0/

# Type checking
mypy src/ops0/

# All pre-commit checks
pre-commit run --all-files
```

### 🔧 Development
```bash
# Development mode installation
pip install -e .

# Installation with optional dependencies
pip install -e ".[dev,ml,cloud]"

# CLI testing
ops0 --help
ops0 init test-project

# Example pipeline test
python examples/dev/test_pipeline.py
```

### 📦 Build and Distribution
```bash
# Build package
python -m build

# Test built package
pip install dist/ops0-*.whl

# Upload to PyPI (test)
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

## Recommended Development Workflow

### 1. New Feature
```bash
# Create branch
git checkout -b feature/new-feature

# Develop with TDD
# 1. Write the test
# 2. Make test fail
# 3. Implement feature
# 4. Make test pass
# 5. Refactor

# Check quality
pre-commit run --all-files

# Commit and push
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature
```

### 2. Bug Fix
```bash
# Create branch
git checkout -b fix/bug-description

# Write reproduction test
# Implement fix
# Verify test passes

# Regression tests
pytest

# Commit
git commit -m "fix: resolve bug in component"
```

### 3. Documentation
```bash
# Generate documentation
cd docs/
make html

# Serve locally
python -m http.server 8000 -d _build/html
```

## 🎯 Component-Specific Testing

### Core Framework
```bash
# Decorators
pytest tests/unit/test_decorators.py -v

# Storage
pytest tests/unit/test_storage.py -v

# Graph/Pipeline
pytest tests/unit/test_graph.py -v

# Executor
pytest tests/unit/test_executor.py -v
```

### Runtime
```bash
# Containers
pytest tests/unit/test_containers.py -v

# Monitoring  
pytest tests/unit/test_monitoring.py -v
```

### CLI
```bash
# CLI commands
pytest tests/unit/test_cli.py -v

# CLI integration
pytest tests/integration/test_cli_integration.py -v
```

## 🐛 Debugging

### Debug with pytest
```bash
# Stop on first failure
pytest -x

# Verbose with output
pytest -v -s

# Debug with pdb
pytest --pdb

# Debug on failure only
pytest --pdb-on-failure
```

### Debug ops0 code
```bash
# Debug mode
export OPS0_LOG_LEVEL=DEBUG
export OPS0_ENV=development

# Execute with debug
python -m pdb examples/dev/test_pipeline.py
```

## 📊 Performance and Profiling

### Test profiling
```bash
# Profiling with pytest-profiling
pip install pytest-profiling
pytest --profile

# Benchmarks
pytest tests/benchmarks/ --benchmark-only
```

### Code profiling
```bash
# cProfile
python -m cProfile -o profile.stats examples/dev/test_pipeline.py

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## 🔄 Local CI/CD

### Simulate CI locally
```bash
# All CI checks
./scripts/run-local-ci.sh

# Or manually:
black --check src/
ruff check src/
mypy src/
pytest
```

### Test installation
```bash
# In clean environment
python -m venv test-env
source test-env/bin/activate
pip install .
python -c "import ops0; print('Installation OK')"
```

## 📝 Conventions

### Commits
```
feat: new feature
fix: bug fix  
docs: documentation update
style: formatting, no logic change
refactor: refactoring without functionality change
test: add or modify tests
chore: maintenance tasks
```

### Branches
```
main: stable main branch
develop: development branch
feature/feature-name: new features
fix/bug-description: bug fixes
release/version: release preparation
```

### Tests
- One test per functionality
- Descriptive names: `test_step_decorator_with_type_hints`
- Arrange-Act-Assert pattern
- Mock external dependencies

## 🎯 Recommended Development Order

### Phase 1: Core (3-5 days)
1. ✅ Decorators (`@ops0.step`, `@ops0.pipeline`)
2. ✅ Storage layer (transparent save/load)
3. ✅ Graph building (dependency detection)
4. ✅ Local executor
5. ✅ Basic CLI

### Phase 2: Advanced (5-7 days)  
1. 🔲 Container auto-generation
2. 🔲 Distributed execution
3. 🔲 Monitoring/observability
4. 🔲 ML framework integrations
5. 🔲 Error handling/retry

### Phase 3: Production (3-5 days)
1. 🔲 Cloud deployment
2. 🔲 Scaling logic
3. 🔲 Security/auth
4. 🔲 Complete documentation
5. 🔲 Advanced examples

Happy coding! 🐍⚡