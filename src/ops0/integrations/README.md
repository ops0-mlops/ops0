# ops0 ML Framework Integrations

Seamless integration with popular ML frameworks - zero configuration, automatic optimization, and transparent serialization.

## 🚀 Quick Start

```python
import ops0
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@ops0.step
def train_model(data: pd.DataFrame):
    # ops0 automatically detects sklearn and optimizes serialization
    model = RandomForestClassifier()
    model.fit(data.drop('target', axis=1), data['target'])
    return model  # Automatically serialized with joblib

@ops0.step
def predict(model, new_data: pd.DataFrame):
    # Model is automatically deserialized
    predictions = model.predict(new_data)
    return predictions
```

## 🎯 Available Integrations

### Core ML Frameworks

| Framework | Version | Capabilities | Auto-Detection |
|-----------|---------|--------------|----------------|
| **scikit-learn** | ≥0.24 | ✅ Training<br>✅ Inference<br>✅ Model Registry<br>✅ Auto-serialization<br>✅ Cross-validation | `from sklearn` |
| **PyTorch** | ≥1.9 | ✅ GPU Support<br>✅ Distributed Training<br>✅ TorchScript<br>✅ Mixed Precision | `import torch` |
| **TensorFlow** | ≥2.6 | ✅ GPU Support<br>✅ SavedModel Format<br>✅ TF Serving<br>✅ TensorFlow Lite | `import tensorflow` |
| **XGBoost** | ≥1.5 | ✅ GPU Acceleration<br>✅ Distributed Training<br>✅ SHAP Integration | `import xgboost` |
| **LightGBM** | ≥3.0 | ✅ Categorical Features<br>✅ GPU Support<br>✅ Optuna Integration | `import lightgbm` |
| **Hugging Face** | ≥4.20 | ✅ Model Hub<br>✅ Transformers<br>✅ Datasets<br>✅ Tokenizers | `from transformers` |

### Data Processing

| Framework | Version | Capabilities | Auto-Detection |
|-----------|---------|--------------|----------------|
| **pandas** | ≥1.3 | ✅ Parquet Serialization<br>✅ Memory Optimization<br>✅ Data Validation | `import pandas` |
| **NumPy** | ≥1.19 | ✅ Memory Mapping<br>✅ Compressed Storage<br>✅ Type Optimization | `import numpy` |

## 📋 Feature Matrix

| Feature | sklearn | PyTorch | TensorFlow | XGBoost | LightGBM | HuggingFace |
|---------|---------|---------|------------|---------|----------|-------------|
| Auto-serialization | ✅ joblib | ✅ torch.save | ✅ SavedModel | ✅ JSON | ✅ Text | ✅ save_pretrained |
| GPU Detection | ❌ | ✅ Auto | ✅ Auto | ✅ Manual | ✅ Compile-time | ✅ Auto |
| Distributed | ❌ | ✅ DDP | ✅ Strategy | ✅ Dask | ✅ Native | ✅ Trainer |
| Model Registry | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Explanations | ✅ SHAP | ⚠️ Limited | ⚠️ Limited | ✅ SHAP | ✅ SHAP | ⚠️ Limited |
| Optimization | ✅ | ✅ TorchScript | ✅ TFLite | ✅ | ✅ | ✅ Quantization |

## 🔧 Usage Examples

### Automatic Framework Detection

```python
@ops0.step
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # ops0 detects pandas and uses Parquet serialization
    return df.fillna(0).drop_duplicates()

@ops0.step
def train_pytorch_model(data: torch.Tensor) -> torch.nn.Module:
    # ops0 detects PyTorch and handles GPU placement
    model = MyNeuralNetwork()
    # ... training code ...
    return model  # Serialized with torch.save
```

### Manual Integration Control

```python
from ops0.integrations import get_integration

# Get specific integration
sklearn_integration = get_integration('sklearn')

# Wrap model for enhanced features
wrapped_model = sklearn_integration.wrap_model(my_model)

# Use integration features
cv_results = sklearn_integration.cross_validate(
    model, X, y, 
    cv=5, 
    scoring=['accuracy', 'f1_macro']
)
```

### scikit-learn Integration

```python
@ops0.step
def train_sklearn_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])
    
    # ops0 handles the entire pipeline serialization
    return pipeline

@ops0.step
def auto_ml():
    from ops0.integrations import get_integration
    
    sklearn = get_integration('sklearn')
    
    # Auto-select model based on task
    model = sklearn.auto_select_model(task='classification')
    
    # Automatic hyperparameter tuning
    best_model = sklearn.hyperparameter_search(
        X, y,
        param_grid={
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None]
        }
    )
    
    return best_model
```

### PyTorch Integration

```python
@ops0.step
def train_torch_model(train_loader):
    import torch
    from ops0.integrations import get_integration
    
    torch_integration = get_integration('torch')
    
    # Model automatically placed on GPU if available
    model = MyModel()
    
    # Optimize for inference
    optimized_model = torch_integration.optimize_for_inference(
        model,
        example_input=torch.randn(1, 3, 224, 224),
        use_torchscript=True
    )
    
    # Get resource requirements
    resources = torch_integration.get_resource_requirements(model)
    print(f"Model needs: {resources}")
    
    return optimized_model

@ops0.step
def distributed_training():
    # ops0 automatically handles distributed setup
    torch_integration = get_integration('torch')
    
    # Setup distributed training
    torch_integration.distributed_setup(
        rank=ops0.get_rank(),
        world_size=ops0.get_world_size()
    )
    
    # Your training code works normally
    model = train_model()
    return model
```

### TensorFlow Integration

```python
@ops0.step
def train_tf_model():
    import tensorflow as tf
    from ops0.integrations import get_integration
    
    tf_integration = get_integration('tensorflow')
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile with XLA optimization
    model = tf_integration.optimize_for_inference(
        model,
        compile_optimization=True,
        convert_to_tflite=True,
        tflite_path='model.tflite'
    )
    
    # Export for TensorFlow Serving
    tf_integration.export_for_serving(
        model,
        export_path='./serving_model',
        version=1
    )
    
    return model

@ops0.step
def distributed_strategy():
    tf_integration = get_integration('tensorflow')
    
    # Get distributed strategy
    strategy = tf_integration.distributed_strategy('mirrored')
    
    with strategy.scope():
        model = create_model()
        model.compile(...)
    
    return model
```

### XGBoost Integration

```python
@ops0.step
def train_xgboost():
    from ops0.integrations import get_integration
    
    xgb_integration = get_integration('xgboost')
    
    # Create classifier with GPU support
    model = xgb_integration.create_classifier(
        n_estimators=100,
        tree_method='gpu_hist' if ops0.has_gpu() else 'hist'
    )
    
    # Hyperparameter tuning
    best_model = xgb_integration.hyperparameter_search(
        X, y,
        param_grid={
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300]
        }
    )
    
    # Get SHAP explanations
    explanations = xgb_integration.explain_predictions(
        best_model, X_test
    )
    
    return best_model, explanations
```

### Hugging Face Integration

```python
@ops0.step
def load_transformer():
    from ops0.integrations import get_integration
    
    hf_integration = get_integration('huggingface')
    
    # Load model from Hub
    model, tokenizer = hf_integration.load_task_model(
        'bert-base-uncased',
        task='text-classification'
    )
    
    # Create pipeline
    pipeline = hf_integration.create_pipeline(
        'sentiment-analysis',
        model=model,
        tokenizer=tokenizer
    )
    
    return pipeline

@ops0.step
def train_transformer(train_dataset, eval_dataset):
    hf_integration = get_integration('huggingface')
    
    # Create trainer with automatic optimization
    trainer = hf_integration.trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=3,
        fp16=True,  # Automatic mixed precision
    )
    
    # Train and push to Hub
    trainer.train()
    hf_integration.push_to_hub(model, 'my-org/my-model')
    
    return model
```

### Data Processing

```python
@ops0.step
def optimize_dataframe(df: pd.DataFrame):
    from ops0.integrations import get_integration
    
    pandas_integration = get_integration('pandas')
    
    # Optimize memory usage
    df_optimized = pandas_integration.optimize_dataframe(df)
    
    # Validate against schema
    validation = pandas_integration.validate_dataframe(
        df_optimized,
        schema={
            'required_columns': ['id', 'feature1', 'target'],
            'dtypes': {'id': 'int', 'target': 'float'},
            'ranges': {'feature1': {'min': 0, 'max': 100}}
        }
    )
    
    if not validation['valid']:
        raise ValueError(f"Validation failed: {validation['errors']}")
    
    return df_optimized

@ops0.step
def process_large_array(data: np.ndarray):
    numpy_integration = get_integration('numpy')
    
    # Optimize array (auto memory-mapping for large arrays)
    data_optimized = numpy_integration.optimize_array(
        data,
        use_memmap=True,
        allow_float32=True
    )
    
    # Create sliding windows efficiently
    windows = numpy_integration.sliding_window(
        data_optimized,
        window_size=100,
        step=50
    )
    
    return windows
```

## 🎨 Custom Integrations

Create your own integration by extending `BaseIntegration`:

```python
from ops0.integrations.base import (
    BaseIntegration, 
    ModelWrapper, 
    SerializationHandler,
    IntegrationCapability
)

class MyFrameworkSerializer(SerializationHandler):
    def serialize(self, obj):
        # Your serialization logic
        return serialized_bytes
    
    def deserialize(self, data):
        # Your deserialization logic
        return obj
    
    def get_format(self):
        return "myframework"

class MyFrameworkModelWrapper(ModelWrapper):
    def predict(self, data, **kwargs):
        # Your prediction logic
        return self.model.predict(data)
    
    def _extract_metadata(self):
        # Extract model metadata
        return ModelMetadata(
            framework='myframework',
            framework_version='1.0',
            model_type=type(self.model).__name__
        )

class MyFrameworkIntegration(BaseIntegration):
    @property
    def is_available(self):
        try:
            import myframework
            return True
        except ImportError:
            return False
    
    def _define_capabilities(self):
        return [
            IntegrationCapability.TRAINING,
            IntegrationCapability.INFERENCE,
            IntegrationCapability.SERIALIZATION
        ]
    
    def wrap_model(self, model):
        return MyFrameworkModelWrapper(model)
    
    def get_serialization_handler(self):
        return MyFrameworkSerializer()

# Register your integration
from ops0.integrations import register_integration
register_integration('myframework', MyFrameworkIntegration)
```

## 🔍 Advanced Features

### Resource Management

```python
@ops0.step
def gpu_aware_training(data):
    # ops0 automatically detects and allocates GPU resources
    model = create_model()
    
    # Get resource requirements
    for framework in ['sklearn', 'torch', 'tensorflow']:
        integration = get_integration(framework)
        if integration:
            resources = integration.get_resource_requirements(model)
            print(f"{framework} needs: {resources}")
    
    return model
```

### Model Registry

```python
@ops0.step
def register_models():
    from ops0.integrations import get_integration
    
    # Register any ML model
    sklearn_integration = get_integration('sklearn')
    model_id = sklearn_integration.register_model(
        model=my_sklearn_model,
        name='fraud-detector',
        tags={'version': '1.0', 'environment': 'production'}
    )
    
    # Load from registry
    loaded_model = sklearn_integration.load_model(model_id)
    
    return loaded_model
```

### Performance Profiling

```python
@ops0.step
def profile_models():
    integrations = ['pytorch', 'tensorflow', 'xgboost']
    
    for framework in integrations:
        integration = get_integration(framework)
        if integration:
            profile = integration.profile_model(
                model, 
                sample_data
            )
            print(f"{framework} inference: {profile['mean_inference_ms']}ms")
```

## 🚨 Troubleshooting

### Import Errors

```python
# Check available integrations
from ops0.integrations import list_available_integrations

available = list_available_integrations()
print(f"Available integrations: {list(available.keys())}")

# Install missing frameworks
# pip install scikit-learn torch tensorflow xgboost lightgbm transformers
```

### Serialization Issues

```python
# Force specific serialization format
from ops0.integrations import get_integration

pandas_integration = get_integration('pandas')
pandas_integration._default_format = 'pickle'  # Instead of parquet

# Or use manual serialization
from ops0.integrations import auto_serialize, auto_deserialize

serialized = auto_serialize(my_object)
deserialized = auto_deserialize(serialized, framework='sklearn')
```

### GPU Detection

```python
# Check GPU availability per framework
frameworks = ['pytorch', 'tensorflow', 'xgboost']

for fw in frameworks:
    integration = get_integration(fw)
    if integration:
        gpu_available = integration._check_gpu_available()
        print(f"{fw}: GPU {'available' if gpu_available else 'not available'}")
```

## 📚 Best Practices

1. **Let ops0 detect frameworks automatically** - It chooses optimal serialization
2. **Use integration features** - Cross-validation, hyperparameter search, etc.
3. **Check capabilities** - Not all frameworks support all features
4. **Profile your models** - Use built-in profiling for optimization
5. **Leverage GPU support** - ops0 handles device placement automatically

## 🔗 Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Hugging Face Documentation](https://huggingface.co/docs)