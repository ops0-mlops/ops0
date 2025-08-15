# ops0 - Write Python, Ship Production 🚀

<p align="center">
  <strong>Transform Python functions into production ML pipelines with zero configuration</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#how-it-works">How it Works</a> •
  <a href="#examples">Examples</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

**ops0** is a Python-first MLOps framework that automatically converts your data science code into scalable, production-ready pipelines. No YAML, no Kubernetes manifests, no infrastructure headaches - just write Python and deploy.

## ✨ The ops0 Magic

```python
import ops0

@ops0.step
def preprocess(data):
    # Your existing Python code
    return data.dropna()

@ops0.step
def train(data):
    model = RandomForestClassifier()
    model.fit(data)
    ops0.save_model(model, "classifier")
    return {"accuracy": 0.95}

@ops0.pipeline
def ml_pipeline(input_path):
    data = preprocess(load_data(input_path))
    return train(data)

# That's it! Deploy to production:
ops0.deploy()
```

## 🚀 Quick Start

### Installation

```bash
pip install ops0
```

### Your First Pipeline

1. **Initialize a new project**
```bash
ops0 init my-ml-project
cd my-ml-project
```

2. **Write your pipeline** (`pipeline.py`)
```python
import ops0
import pandas as pd

@ops0.step
def load_data(path):
    return pd.read_csv(path)

@ops0.step
def process(df):
    # ops0 handles serialization between steps
    return df.fillna(0)

@ops0.step
def train_model(df):
    # Your ML code here
    model = train_your_model(df)
    ops0.save_model(model, "my_model")
    return {"status": "success"}

@ops0.pipeline
def training_pipeline(data_path):
    df = load_data(data_path)
    processed = process(df)
    return train_model(processed)
```

3. **Test locally**
```bash
ops0 run --local
```

4. **Deploy to AWS**
```bash
ops0 deploy
```

## 🎯 Features

### Zero Configuration
- No YAML files, no JSON configs
- Automatic dependency detection via AST analysis
- Smart resource allocation based on your code

### Production Ready
- Automatic containerization of each step
- Built-in error handling and retries
- Scalable from laptop to cloud

### Developer Friendly
- Test locally with one command
- Hot reload during development
- Clear error messages

### Cloud Native
- Deploy to AWS Lambda + Step Functions
- Automatic S3 storage management
- Pay only for what you use

## 🔧 How It Works

1. **Code Analysis**: ops0 uses AST parsing to understand your function dependencies
2. **DAG Construction**: Automatically builds an execution graph from your code
3. **Containerization**: Each `@step` becomes an isolated, scalable unit
4. **Orchestration**: Manages execution order, parallelization, and data flow
5. **Deployment**: Generates cloud infrastructure as code (AWS CDK)

## 📚 Examples

### Real-time Fraud Detection
```python
@ops0.step
def extract_features(transaction):
    # Feature engineering
    features = {
        'amount_zscore': calculate_zscore(transaction.amount),
        'merchant_risk': get_merchant_score(transaction.merchant_id),
        'time_features': extract_time_features(transaction.timestamp)
    }
    return features

@ops0.step
def predict_fraud(features):
    model = ops0.load_model("fraud_detector_v2")
    score = model.predict_proba(features)[0][1]
    return {'fraud_probability': score}

@ops0.step
def alert_if_suspicious(prediction):
    if prediction['fraud_probability'] > 0.9:
        ops0.notify.slack("🚨 High risk transaction detected!")
        return {'alerted': True}
    return {'alerted': False}

@ops0.pipeline
def fraud_detection(transaction):
    features = extract_features(transaction)
    prediction = predict_fraud(features)
    return alert_if_suspicious(prediction)
```

### Model Training Pipeline
```python
@ops0.step
def prepare_dataset(raw_data_path):
    df = pd.read_parquet(raw_data_path)
    # ops0 automatically handles large dataframes
    return train_test_split(df)

@ops0.step
def train_model(X_train, y_train):
    # ops0 detects ML libraries and allocates GPU if needed
    model = XGBClassifier(gpu_id=0)
    model.fit(X_train, y_train)
    ops0.save_model(model, "xgboost_model")
    return model

@ops0.step
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'f1': f1_score(y_test, predictions)
    }
    ops0.log_metrics(metrics)
    return metrics

@ops0.pipeline
def training_pipeline(data_path):
    X_train, X_test, y_train, y_test = prepare_dataset(data_path)
    model = train_model(X_train, y_train)
    return evaluate(model, X_test, y_test)
```

## 🛠 CLI Commands

```bash
ops0 init          # Initialize a new project
ops0 run           # Run pipeline locally
ops0 deploy        # Deploy to cloud
ops0 status        # Check pipeline status
ops0 logs          # View execution logs
ops0 rollback      # Rollback to previous version
```

## 🗺 Roadmap

### Phase 1: MVP (Current)
- ✅ Core decorators (@step, @pipeline)
- ✅ AST-based dependency analysis
- ✅ Local execution engine
- ✅ Basic storage abstraction
- ✅ CLI interface
- 🚧 AWS CDK generation

### Phase 2: Production Features
- ⏳ Automatic containerization
- ⏳ Distributed orchestration
- ⏳ Monitoring & alerting
- ⏳ Model registry
- ⏳ GPU support
- ⏳ Multi-cloud support

### Phase 3: Ecosystem
- ⏳ Step marketplace
- ⏳ Native ML framework integrations
- ⏳ VS Code extension
- ⏳ Web dashboard
- ⏳ A/B testing support

## 🤝 Contributing

We're building ops0 in public! Contributions are welcome:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

ops0 is inspired by the simplicity of Flask, the power of Airflow, and the developer experience of Vercel. We believe ML infrastructure should be invisible.

---

<p align="center">
  <strong>Stop configuring. Start shipping. 🚀</strong>
</p>

<p align="center">
  Made with ❤️ by developers who were tired of YAML files
</p>