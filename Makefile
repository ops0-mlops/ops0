# ops0 Development Makefile
# Simplifies common development tasks

.PHONY: help setup test lint format clean build docs install dev-install

# Default target
help: ## Show this help message
	@echo "🐍 ops0 Development Commands"
	@echo "============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and Installation
setup: ## Setup development environment
	@echo "🚀 Setting up ops0 development environment..."
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -e ".[dev,ml,cloud,monitoring]"
	. venv/bin/activate && pre-commit install
	@echo "✅ Development environment ready!"
	@echo "Activate with: source venv/bin/activate"

install: ## Install ops0 in current environment
	pip install -e .

dev-install: ## Install with all development dependencies
	pip install -e ".[dev,ml,cloud,monitoring]"

# Code Quality
format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black src/ tests/ examples/
	isort src/ tests/ examples/
	@echo "✅ Code formatted!"

lint: ## Run all linting checks
	@echo "🔍 Running linting checks..."
	ruff check src/ tests/
	mypy src/
	bandit -r src/ --format=colorized
	@echo "✅ Linting complete!"

quality: format lint ## Run formatting and linting

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Testing
test: ## Run all tests
	@echo "🧪 Running tests..."
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	pytest tests/e2e/ -v

test-cov: ## Run tests with coverage report
	pytest --cov=ops0 --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode
	ptw tests/unit/ -- --testmon

# Development
dev: ## Start development mode (install + format + test)
	make dev-install
	make format
	make test-unit

check: ## Run all checks (format, lint, test)
	make format
	make lint
	make test

fix: ## Auto-fix common issues
	black src/ tests/ examples/
	isort src/ tests/ examples/
	ruff check src/ tests/ --fix

# Docker and Containerization
docker-build: ## Build development Docker image
	docker build -t ops0:dev -f deployment/docker/Dockerfile.dev .

docker-test: ## Run tests in Docker container
	docker run --rm ops0:dev pytest

# Documentation
docs: ## Build documentation
	@echo "📚 Building documentation..."
	cd docs && make html
	@echo "✅ Documentation built! Open docs/_build/html/index.html"

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	cd docs && make clean

# Building and Distribution
build: ## Build distribution packages
	@echo "📦 Building distribution packages..."
	python -m build
	@echo "✅ Build complete! Check dist/ directory"

build-clean: ## Clean build artifacts and rebuild
	make clean
	make build

publish-test: ## Publish to Test PyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish: ## Publish to PyPI (production)
	twine upload dist/*

# Examples and Development Testing
example: ## Run the development example pipeline
	python examples/dev/test_pipeline.py

example-ml: ## Run ML pipeline examples
	python examples/ml_pipelines/simple_ml.py

cli-test: ## Test CLI commands
	ops0 --help
	ops0 init test-project --dry-run

# Cleaning
clean: ## Remove all build artifacts and cache files
	@echo "🧹 Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

clean-all: clean ## Remove all artifacts including venv
	rm -rf venv/
	rm -rf .ops0/
	rm -rf logs/

# Performance and Profiling
profile: ## Profile the example pipeline
	python -m cProfile -o profile.stats examples/dev/test_pipeline.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

benchmark: ## Run performance benchmarks
	pytest tests/benchmarks/ --benchmark-only

# Security
security: ## Run security checks
	bandit -r src/ --format=colorized
	safety check

# Git hooks and CI simulation
hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

ci: ## Simulate CI pipeline locally
	@echo "🔄 Running CI pipeline locally..."
	make format
	make lint
	make test
	make build
	@echo "✅ CI pipeline complete!"

# Release preparation
release-prep: ## Prepare for release (version bump, changelog, etc.)
	@echo "🚀 Preparing release..."
	# Version bump would go here
	make clean
	make build
	make test
	@echo "✅ Release preparation complete!"

# Development utilities
deps-update: ## Update all dependencies
	pip-compile --upgrade requirements/dev.in
	pip-compile --upgrade requirements/prod.in

deps-install: ## Install dependencies from lock files
	pip-sync requirements/dev.txt

init-project: ## Initialize a new ops0 project (for testing)
	ops0 init test-project
	cd test-project && python pipeline.py

# Advanced development
type-check: ## Run comprehensive type checking
	mypy src/ --strict

complexity: ## Check code complexity
	radon cc src/ -a

dead-code: ## Find dead code
	vulture src/

# Database and data management (for future features)
db-migrate: ## Run database migrations (placeholder)
	@echo "🗄️ Database migrations not implemented yet"

# Container registry operations
registry-login: ## Login to container registry
	echo $$GHCR_TOKEN | docker login ghcr.io -u $$GHCR_USERNAME --password-stdin

registry-push: ## Push containers to registry
	docker push ghcr.io/ops0-mlops/ops0:latest

# Monitoring and observability
logs: ## View application logs
	tail -f logs/ops0.log

metrics: ## Show development metrics
	@echo "📊 Development Metrics:"
	@find src/ -name "*.py" | xargs wc -l | tail -1
	@echo "Tests:" && find tests/ -name "*.py" | wc -l
	@echo "Examples:" && find examples/ -name "*.py" | wc -l

# Help and information
info: ## Show development environment info
	@echo "🐍 ops0 Development Environment"
	@echo "=============================="
	@echo "Python version: $$(python --version)"
	@echo "Virtual env: $$VIRTUAL_ENV"
	@echo "ops0 version: $$(python -c 'import ops0; print(ops0.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not available')"

version: ## Show current version
	@python -c "from src.ops0.__about__ import __version__; print(f'ops0 version: {__version__}')"

# Quick development workflow
quick: ## Quick development cycle (format, lint, test-unit)
	@echo "⚡ Running quick development cycle..."
	make format
	make lint
	make test-unit
	@echo "✅ Quick cycle complete!"

# Full development workflow
full: ## Full development cycle (everything)
	@echo "🔄 Running full development cycle..."
	make format
	make lint
	make test
	make build
	make docs
	@echo "✅ Full cycle complete!"

# Default development command
dev-default: quick ## Default development command (alias for quick)