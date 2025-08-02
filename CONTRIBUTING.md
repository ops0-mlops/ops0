# Contributing to ops0 🤝

Thanks for your interest in contributing to ops0! We're building the future of Python-native MLOps, and every contribution makes a difference.

This guide will help you get started, whether you're fixing a bug, adding a feature, or improving documentation.

## 🚀 Quick Start for Contributors

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ops0.git
cd ops0

# 3. Set up development environment
./scripts/setup-dev.sh

# 4. Create a feature branch
git checkout -b feature/your-feature-name

# 5. Make your changes and test
pytest
pre-commit run --all-files

# 6. Push and create a Pull Request
git push origin feature/your-feature-name
```

## 🎯 Types of Contributions

### 🐛 Bug Fixes
Found a bug? Here's how to help:

```bash
# 1. Create a test that reproduces the bug
# tests/test_bug_reproduction.py
def test_bug_reproduction():
    """Test case that demonstrates the bug"""
    with ops0.pipeline():
        @ops0.step
        def failing_step():
            # Code that triggers the bug
            pass
    
    # This should pass after the fix
    assert ops0.run() == expected_result

# 2. Fix the bug
# 3. Ensure the test passes
# 4. Submit PR with "Fixes #issue_number"
```

### ✨ New Features
Want to add a feature? Let's talk first!

1. **💡 Propose**: Open a [Discussion](https://github.com/ops0-mlops/ops0/discussions) or [Issue](https://github.com/ops0-mlops/ops0/issues/new?template=feature_request.md)
2. **🏗️ Design**: We'll help you design the API
3. **🔨 Implement**: Write the code with tests
4. **📝 Document**: Add docs and examples

#### Feature Example: New Decorator
```python
# Example: Adding @ops0.cache decorator
@ops0.step
@ops0.cache(ttl="1h", key="user_{user_id}")
def expensive_computation(user_id, data):
    """Cache results for 1 hour per user"""
    return complex_calculation(user_id, data)
```

### 📚 Documentation
Documentation is as important as code:

- **📖 API Docs**: Improve docstrings and type hints
- **🎓 Tutorials**: Write step-by-step guides
- **💡 Examples**: Real-world pipeline examples
- **🔧 Troubleshooting**: Common issues and solutions

### 🎨 Examples & Templates
Help others get started faster:

```python
# examples/fraud_detection/pipeline.py
"""
Real-time fraud detection pipeline example
Shows: streaming data, ML inference, alerting
"""
import ops0
from sklearn.ensemble import IsolationForest

@ops0.step
@ops0.source.kafka(topic="transactions", format="json")
def stream_transactions():
    """Consume transaction stream"""
    pass

@ops0.step
@ops0.model.load("fraud-detector-v2")
def detect_fraud(transactions, model):
    """Real-time fraud detection"""
    features = extract_features(transactions)
    predictions = model.predict(features)
    return {"transactions": transactions, "predictions": predictions}

@ops0.step
@ops0.alert.slack(channel="#fraud-alerts")
def alert_on_fraud(results):
    """Send alerts for high-risk transactions"""
    high_risk = results["predictions"] > 0.8
    if high_risk.any():
        transactions = results["transactions"][high_risk]
        return f"🚨 {len(transactions)} high-risk transactions detected"
```

## 🛠 Development Setup

### Prerequisites
- Python 3.8+ 
- Docker (for integration tests)
- Git
- Pre-commit (installed automatically)

### Environment Setup

```bash
# Clone and enter directory
git clone https://github.com/ops0-mlops/ops0.git
cd ops0

# Automated setup (recommended)
./scripts/setup-dev.sh

# Manual setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev,test]"
pre-commit install
```

### Project Structure
```
ops0/
├── ops0/                   # Main package
│   ├── core/              # Core framework
│   │   ├── decorators.py  # @ops0.step, @ops0.pipeline
│   │   ├── graph.py       # Dependency analysis
│   │   ├── executor.py    # Local/remote execution
│   │   └── storage.py     # Transparent storage
│   ├── runtime/           # Production runtime
│   │   ├── containers.py  # Auto-containerization
│   │   ├── orchestrator.py# Distributed execution
│   │   └── monitoring.py  # Observability
│   ├── integrations/      # ML framework integrations
│   │   ├── sklearn.py     # Scikit-learn support
│   │   ├── pytorch.py     # PyTorch integration
│   │   └── tensorflow.py  # TensorFlow support
│   └── cli/               # Command line interface
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── docs/                 # Documentation
├── examples/             # Example pipelines
└── scripts/              # Development scripts
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit          # Fast unit tests
pytest tests/integration   # Integration tests
pytest tests/e2e           # End-to-end tests

# Run tests with coverage
pytest --cov=ops0 --cov-report=html

# Run specific test file
pytest tests/unit/test_decorators.py

# Run tests matching pattern
pytest -k "test_step_decorator"
```

### Code Quality

```bash
# Run all quality checks
pre-commit run --all-files

# Individual checks
black ops0/                    # Code formatting
isort ops0/                    # Import sorting
flake8 ops0/                   # Linting
mypy ops0/                     # Type checking
bandit -r ops0/                # Security checks
```

## 📋 Coding Standards

### Python Style
We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Line length: 88 characters (Black default)
# Use type hints everywhere
def process_data(
    data: pd.DataFrame, 
    config: Dict[str, Any]
) -> pd.DataFrame:
    """Process data with configuration.
    
    Args:
        data: Input DataFrame to process
        config: Processing configuration
        
    Returns:
        Processed DataFrame
        
    Raises:
        ValueError: If data is empty
    """
    if data.empty:
        raise ValueError("Data cannot be empty")
    
    return data.transform(config["transformation"])
```

### Decorators Design
Keep the decorator API clean and intuitive:

```python
# ✅ Good: Simple, clear purpose
@ops0.step
def process_data(data):
    return clean(data)

@ops0.step
@ops0.retry(max_attempts=3)
def flaky_external_api():
    return call_api()

# ❌ Avoid: Complex nested decorators
@ops0.step(
    retry=True,
    cache=True,
    monitor=True,
    parallel=True
)  # Too many concerns in one decorator
```

### Error Handling
Provide clear, actionable error messages:

```python
# ✅ Good: Specific, actionable error
raise ops0.InvalidPipelineError(
    f"Step '{step_name}' expects argument '{arg_name}' "
    f"but no previous step provides it. "
    f"Available outputs: {available_outputs}"
)

# ❌ Avoid: Generic, unhelpful error
raise Exception("Pipeline failed")
```

### Testing Guidelines

#### Unit Tests
Test individual components in isolation:

```python
# tests/unit/test_decorators.py
import pytest
from ops0.core.decorators import step

def test_step_decorator_basic():
    """Test basic @ops0.step functionality"""
    @step
    def sample_step(x: int) -> int:
        return x * 2
    
    assert sample_step.is_ops0_step is True
    assert sample_step.input_signature == {"x": int}
    assert sample_step.output_signature == int

def test_step_decorator_with_multiple_inputs():
    """Test step with multiple typed inputs"""
    @step
    def multi_input_step(a: str, b: int, c: float = 1.0) -> dict:
        return {"a": a, "b": b, "c": c}
    
    expected_signature = {
        "a": str,
        "b": int, 
        "c": (float, 1.0)  # type and default
    }
    assert multi_input_step.input_signature == expected_signature
```

#### Integration Tests
Test component interactions:

```python
# tests/integration/test_pipeline_execution.py
def test_local_pipeline_execution():
    """Test complete pipeline execution locally"""
    
    @ops0.step
    def generate_data() -> list:
        return [1, 2, 3, 4, 5]
    
    @ops0.step
    def process_data(data: list) -> list:
        return [x * 2 for x in data]
    
    @ops0.step  
    def summarize(processed: list) -> dict:
        return {"sum": sum(processed), "count": len(processed)}
    
    # Execute pipeline
    with ops0.pipeline("test-pipeline"):
        result = ops0.run(mode="local")
    
    assert result["sum"] == 30
    assert result["count"] == 5
```

#### End-to-End Tests
Test complete user workflows:

```python
# tests/e2e/test_deployment.py
@pytest.mark.slow
def test_full_deployment_workflow():
    """Test complete notebook-to-production workflow"""
    
    # Create temporary pipeline file
    pipeline_code = """
import ops0

@ops0.step
def hello_world():
    return {"message": "Hello from ops0!"}
"""
    
    with temp_pipeline_file(pipeline_code) as pipeline_path:
        # Test local execution
        result = ops0.run(pipeline_path, mode="local")
        assert result["message"] == "Hello from ops0!"
        
        # Test deployment (uses test environment)
        deployment = ops0.deploy(pipeline_path, env="test")
        assert deployment.status == "deployed"
        
        # Test remote execution
        remote_result = deployment.invoke()
        assert remote_result["message"] == "Hello from ops0!"
```

## 🔍 Pull Request Process

### Before Submitting
- [ ] Tests pass (`pytest`)
- [ ] Code quality checks pass (`pre-commit run --all-files`)
- [ ] Documentation updated (if needed)
- [ ] Changelog entry added (if user-facing change)

### PR Description Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All tests pass

## Screenshots (if applicable)
Add screenshots or GIFs for UI changes.
```

### Review Process
1. **Automated Checks**: CI runs tests and quality checks
2. **Code Review**: Team member reviews the code
3. **Testing**: Changes tested in staging environment
4. **Approval**: Maintainer approves and merges

## 🏗 Architecture Deep Dive

Understanding ops0's architecture helps you contribute effectively:

### Core Components

#### 1. AST Analysis Engine
```python
# ops0/core/analyzer.py
class PipelineAnalyzer:
    """Analyzes Python code to build execution graphs"""
    
    def analyze_function(self, func: Callable) -> StepMetadata:
        """Extract step metadata from function"""
        tree = ast.parse(inspect.getsource(func))
        
        # Extract input/output signatures
        signature = self._extract_signature(func)
        
        # Analyze dependencies
        dependencies = self._find_dependencies(tree)
        
        return StepMetadata(
            name=func.__name__,
            signature=signature,
            dependencies=dependencies,
            source_code=tree
        )
```

#### 2. Storage Abstraction
```python
# ops0/core/storage.py
class StorageLayer:
    """Transparent data passing between steps"""
    
    def save(self, key: str, data: Any, namespace: str = None):
        """Save data with automatic serialization"""
        serializer = self._choose_serializer(data)
        serialized = serializer.serialize(data)
        self.backend.store(f"{namespace}/{key}", serialized)
    
    def load(self, key: str, namespace: str = None) -> Any:
        """Load data with automatic deserialization"""
        data = self.backend.retrieve(f"{namespace}/{key}")
        return self._deserialize(data)
```

#### 3. Execution Engine
```python
# ops0/runtime/executor.py
class PipelineExecutor:
    """Orchestrates step execution"""
    
    def execute(self, pipeline: Pipeline, mode: str = "local"):
        """Execute pipeline in specified mode"""
        graph = self._build_execution_graph(pipeline)
        
        if mode == "local":
            return self._execute_local(graph)
        elif mode == "distributed":
            return self._execute_distributed(graph)
```

### Adding New Integrations

Example: Adding Pandas integration

```python
# ops0/integrations/pandas.py
import pandas as pd
from ops0.core.decorators import step
from ops0.core.storage import register_serializer

# Custom serializer for DataFrames
class PandasSerializer:
    def serialize(self, df: pd.DataFrame) -> bytes:
        return df.to_parquet()
    
    def deserialize(self, data: bytes) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(data))

# Register the serializer
register_serializer(pd.DataFrame, PandasSerializer())

# Pandas-specific decorators
def dataframe_step(func):
    """Decorator for DataFrame processing steps"""
    @step
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            # Add metadata
            result.attrs['ops0_step'] = func.__name__
            result.attrs['ops0_timestamp'] = time.time()
        return result
    return wrapper
```

## 📊 Performance Guidelines

### Benchmarking New Features
```python
# tests/benchmarks/test_performance.py
import pytest
import time
from ops0.benchmarks import measure_performance

@pytest.mark.benchmark
def test_step_execution_performance():
    """Ensure step execution stays under latency thresholds"""
    
    @ops0.step
    def simple_step(data):
        return data * 2
    
    data = list(range(1000))
    
    # Measure execution time
    with measure_performance() as timer:
        result = simple_step(data)
    
    # Assert performance requirements
    assert timer.elapsed < 0.1  # 100ms max
    assert len(result) == 1000
```

### Memory Optimization
```python
# Use generators for large datasets
@ops0.step
def process_large_dataset(data_path: str) -> Iterator[Dict]:
    """Process data in chunks to optimize memory"""
    for chunk in pd.read_csv(data_path, chunksize=1000):
        yield from chunk.to_dict('records')

# Automatic cleanup
@ops0.step
@ops0.cleanup_after(minutes=5)
def temporary_processing(data):
    """Automatically cleanup temporary resources"""
    temp_file = create_temp_file(data)
    result = process_file(temp_file)
    # ops0 handles cleanup automatically
    return result
```

## 🎓 Learning Resources

### For New Contributors
- [Python AST Tutorial](https://docs.python.org/3/library/ast.html)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Pytest Documentation](https://docs.pytest.org/)

### ops0-Specific Resources
- [Architecture Overview](https://docs.ops0.xyz/architecture)
- [Plugin Development Guide](https://docs.ops0.xyz/plugins)
- [Performance Optimization](https://docs.ops0.xyz/performance)

## 🎉 Recognition

### Contributor Levels
- **First-time**: Welcome package & Discord recognition
- **Regular**: Contributor badge & early access to features  
- **Core**: Commit access & decision-making participation
- **Maintainer**: Full project access & leadership role

### Hall of Fame
Contributors are recognized in:
- GitHub profile with contributor badge
- [Contributors page](https://ops0.xyz/contributors) on website
- Annual contributor spotlight blog posts
- Conference speaker opportunities

## 📞 Getting Help

### Quick Questions
- 💬 [Discord #contributors](https://discord.gg/ops0)
- 🐛 [GitHub Issues](https://github.com/ops0-mlops/ops0/issues)

### Longer Discussions  
- 💡 [GitHub Discussions](https://github.com/ops0-mlops/ops0/discussions)
- 📧 [Mailing List](mailto:contributors@ops0.xyz)

### Office Hours
- **Weekly**: Thursdays 2-3 PM PST
- **Format**: Discord voice chat
- **Topics**: Architecture, feature planning, Q&A

## 🤝 Code of Conduct

We're committed to providing a welcoming, inclusive environment for all contributors.

### Our Standards
- **Be respectful**: Treat everyone with kindness and respect
- **Be inclusive**: Welcome newcomers and different perspectives  
- **Be collaborative**: Help others learn and grow
- **Be constructive**: Focus on solutions, not problems

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing private information without consent
- Any behavior that would be inappropriate in a professional setting

### Reporting
If you experience or witness unacceptable behavior:
- **Email**: [conduct@ops0.xyz](mailto:conduct@ops0.xyz)
- **Discord**: Direct message any moderator
- **Anonymous**: [Report form](https://ops0.xyz/report)

All reports are handled confidentially and fairly.

## 📜 License

By contributing to ops0, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## 🙏 Thank You!

Every contribution, no matter how small, makes ops0 better for the entire ML community. Whether you're fixing a typo, adding a feature, or helping with documentation, you're helping data scientists around the world ship ML models faster.

Welcome to the ops0 family! 🐍⚡

---

<div align="center">

**Ready to contribute?**

[🍴 Fork the repo](https://github.com/ops0-mlops/ops0/fork) • [💬 Join Discord](https://discord.gg/ops0) • [📖 Read the docs](https://docs.ops0.xyz)

*Built with ❤️ by the ops0 community*

</div>