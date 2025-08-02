# ops0 🐍⚡

> **Write Python. Ship Production. Forget the Infrastructure.**

[![GitHub Stars](https://img.shields.io/github/stars/ops0-mlops/ops0?style=for-the-badge)](https://github.com/ops0-mlops/ops0)
[![Python Version](https://img.shields.io/pypi/pyversions/ops0?style=for-the-badge)](https://pypi.org/project/ops0/)
[![License](https://img.shields.io/github/license/ops0-mlops/ops0?style=for-the-badge)](LICENSE)

**ops0** is a Python-native ML pipeline orchestration framework that transforms your functions into production-ready, scalable pipelines with zero DevOps complexity.

## 🚀 Quick Start

Transform any Python function into a production ML pipeline in **under 5 minutes**:

```python
import ops0

@ops0.step
def preprocess(data):
    cleaned = data.dropna()
    return cleaned.scaled()

@ops0.step  
def train_model(processed_data):
    model = RandomForestClassifier()
    model.fit(processed_data.X, processed_data.y)
    return model

@ops0.step
def predict(model, new_data):
    return model.predict(new_data)

# Deploy to production
ops0.deploy()  # 🚀 That's it!
```

**Result**: Your pipeline is live at `https://your-pipeline.ops0.xyz` with automatic scaling, monitoring, and fault tolerance.

## 🎯 Why ops0?

| Traditional MLOps | ops0 |
|---|---|
| 📋 Write hundreds of lines of YAML | 🐍 Pure Python decorators |
| 🧩 Manually define DAGs and dependencies | 🧠 Automatic dependency detection |
| 🔧 Configure infrastructure, workers, queues | 📦 Invisible containerization |
| 🤯 Learn platform-specific syntax | ⚡ If you know Python, you know ops0 |
| ⏰ Days to deploy a simple model | 🚀 Production in 5 minutes |

## 📦 Installation

```bash
# Install ops0
pip install ops0

# Initialize a new project
ops0 init my-ml-pipeline
cd my-ml-pipeline

# Run locally
ops0 run --local

# Deploy to production  
ops0 deploy
```

## 🛠 Core Features

### 🐍 **Pure Python Experience**
No YAML, no configs, no new syntax. Just Python functions with decorators.

```python
@ops0.step
def my_step(input_data):
    # Your existing Python code works unchanged
    result = some_ml_function(input_data)
    return result
```

### 🧠 **Automatic Orchestration**
ops0 analyzes your function signatures to build optimal execution graphs.

```python
@ops0.step
def step_a(data):
    return process(data)

@ops0.step
def step_b(result_from_step_a):  # ops0 knows this depends on step_a
    return transform(result_from_step_a)

@ops0.step  
def step_c(data_from_a, data_from_b):  # Runs in parallel automatically
    return combine(data_from_a, data_from_b)
```

### 📦 **Transparent Storage**
Share data between steps without thinking about serialization.

```python
@ops0.step
def generate_features():
    features = expensive_computation()
    ops0.storage.save("features", features)  # Any Python object
    
@ops0.step
def train():
    features = ops0.storage.load("features")  # Automatic deserialization
    return train_model(features)
```

### 🐳 **Invisible Infrastructure**
Every step becomes an isolated, scalable container automatically.

- **Auto-containerization**: Analyzes imports, builds optimized Docker images
- **Smart scaling**: Each step scales independently based on load
- **Zero config**: No Dockerfiles, no Kubernetes manifests

### 📊 **Built-in Observability**
Production-grade monitoring without setup.

```python
@ops0.step
@ops0.monitor(alert_on_latency=">500ms")
def critical_step(data):
    return process(data)
```

- Real-time pipeline monitoring
- Automatic error detection and retry
- Performance metrics and alerts
- Visual DAG representation

## 🏃‍♂️ Examples

### Fraud Detection Pipeline

```python
import ops0
from sklearn.ensemble import IsolationForest

@ops0.step
def fetch_transactions():
    # Your data loading logic
    return load_from_database()

@ops0.step
def extract_features(transactions):
    features = calculate_risk_features(transactions)
    ops0.storage.save("features", features)
    return features

@ops0.step
def detect_anomalies():
    features = ops0.storage.load("features")
    model = IsolationForest()
    anomalies = model.fit_predict(features)
    return {"anomalies": anomalies, "model": model}

@ops0.step
def alert_on_fraud(results):
    if results["anomalies"].sum() > 10:
        ops0.notify.slack("High fraud activity detected!")
    return results

# Deploy the entire pipeline
ops0.deploy(name="fraud-detector")
```

### Real-time ML Training

```python
@ops0.step
@ops0.schedule("0 2 * * *")  # Run daily at 2 AM
def retrain_model():
    data = fetch_latest_data()
    model = train_updated_model(data)
    ops0.models.deploy(model, "production")  # Blue-green deployment
    return {"model_version": model.version, "accuracy": model.score}

@ops0.step
@ops0.trigger.on_data_change("/data/users")  # React to new data
def update_features():
    return recalculate_user_features()
```

## 🔧 Development Workflow

### Local Development
```bash
# Test your pipeline locally
ops0 run --local

# Debug specific steps
ops0 debug --step train_model

# Validate pipeline structure
ops0 validate
```

### Production Deployment
```bash
# Deploy to staging
ops0 deploy --env staging

# Run tests against staging
ops0 test --env staging

# Promote to production
ops0 promote staging production

# Monitor pipeline health
ops0 status --follow
```

## 🎛 Configuration

### Environment Variables
```bash
export OPS0_API_KEY="your-api-key"
export OPS0_PROJECT="my-ml-project"
export OPS0_ENVIRONMENT="production"
```

### Project Configuration (`ops0.toml`)
```toml
[project]
name = "fraud-detector"
version = "1.0.0"
description = "Real-time fraud detection pipeline"

[deployment]
region = "us-west-2"
auto_scaling = true
max_workers = 10

[monitoring]
alerts = ["slack://fraud-team", "email://alerts@company.com"]
retention_days = 30
```

## 🤝 Contributing

We love contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

### Quick Development Setup

```bash
# Clone the repository
git clone https://github.com/ops0-mlops/ops0.git
cd ops0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
pre-commit run --all-files
```

### Ways to Contribute

- 🐛 **Bug Reports**: Found a bug? [Open an issue](https://github.com/ops0-mlops/ops0/issues/new/choose)
- 💡 **Feature Requests**: Have an idea? [Start a discussion](https://github.com/ops0-mlops/ops0/discussions)
- 📝 **Documentation**: Improve our docs or add examples
- 🔧 **Code**: Fix bugs, add features, or optimize performance
- 🎨 **Design**: Improve UX/UI of our CLI or web dashboard

## 📚 Documentation

- **[Getting Started](https://docs.ops0.xyz/getting-started)** - Your first pipeline in 5 minutes
- **[API Reference](https://docs.ops0.xyz/api)** - Complete function documentation  
- **[Examples](https://docs.ops0.xyz/examples)** - Real-world pipeline examples
- **[Best Practices](https://docs.ops0.xyz/best-practices)** - Production deployment tips
- **[Migration Guide](https://docs.ops0.xyz/migration)** - From Airflow, Prefect, or Kubeflow

## 🏗 Architecture

ops0 is built on three core principles:

1. **Convention over Configuration** - Smart defaults, minimal setup
2. **Progressive Disclosure** - Simple by default, powerful when needed  
3. **Python-First** - If it works in Python, it works in ops0

### Core Components

- **🧠 AST Analyzer**: Understands your code structure and dependencies
- **📦 Container Engine**: Builds optimized, isolated execution environments
- **⚡ Orchestrator**: Manages execution flow and scaling
- **💾 Storage Layer**: Handles data persistence and sharing
- **📊 Monitor**: Tracks performance and health

## 🔒 Security & Privacy

- **🔐 Zero Data Access**: ops0 never accesses your training data
- **🏠 On-Premise Option**: Deploy in your own infrastructure
- **🛡 SOC 2 Compliant**: Enterprise-grade security standards
- **🔍 Audit Logs**: Complete traceability of all operations

## 🚀 Roadmap

### Phase 1: Core Framework ✅
- [x] Python decorators and AST analysis
- [x] Automatic dependency detection  
- [x] Local execution and testing
- [x] Auto-containerization
- [x] Basic cloud deployment

### Phase 2: Production Features 🚧
- [ ] GPU support and auto-detection
- [ ] Native ML framework integrations (scikit-learn, PyTorch, TensorFlow)
- [ ] Advanced monitoring and alerting
- [ ] Pipeline versioning and rollbacks

### Phase 3: Enterprise & Scale 📋
- [ ] Multi-cloud deployment
- [ ] Enterprise security (SSO, RBAC)
- [ ] A/B testing for ML models
- [ ] Cost optimization and resource management

See our [Public Roadmap](https://github.com/orgs/ops0-mlops/projects/1) for detailed progress.

## 📊 Benchmarks

ops0 vs Traditional MLOps platforms:

| Metric | Traditional | ops0 |
|--------|-------------|------|
| **Setup Time** | 2-4 weeks | 5 minutes |
| **Lines of Config** | 200-500 YAML | 0 |
| **Learning Curve** | 2-3 months | Immediate |
| **Deployment Speed** | 30-60 minutes | 4-8 seconds |
| **Maintenance Overhead** | High | Zero |

## 📄 License

ops0 is released under the [MIT License](LICENSE). See [LICENSE](LICENSE) for details.

## 🙋‍♀️ Support & Community

- **💬 [Discord Community](https://discord.gg/ops0)** - Chat with the team and other users
- **📧 [Mailing List](https://groups.google.com/g/ops0-users)** - Stay updated on releases
- **🐙 [GitHub Discussions](https://github.com/ops0-mlops/ops0/discussions)** - Ask questions and share ideas
- **🐛 [Issue Tracker](https://github.com/ops0-mlops/ops0/issues)** - Report bugs and request features
- **📖 [Documentation](https://docs.ops0.xyz)** - Comprehensive guides and API docs

## 🎯 What's Next?

1. **⭐ Star this repo** to stay updated
2. **🚀 Try the [5-minute quickstart](https://docs.ops0.xyz/quickstart)**  
3. **💬 Join our [Discord](https://discord.gg/ops0)** to connect with the community
4. **🤝 [Contribute](CONTRIBUTING.md)** to help shape the future of MLOps

---

<div align="center">

**Built with ❤️ by the ops0 team and contributors**

[Website](https://ops0.xyz) • [Documentation](https://docs.ops0.xyz) • [Community](https://discord.gg/ops0) • [Twitter](https://twitter.com/ops0_ai)

*ops0 - where Python meets Production* 🐍⚡

</div>