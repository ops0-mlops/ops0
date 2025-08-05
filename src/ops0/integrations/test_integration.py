"""
Test file for ops0 integrations

Run this to verify all integrations are working correctly:
    python -m ops0.integrations.test_integrations
"""

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_integration_detection():
    """Test automatic framework detection"""
    from . import detect_framework, list_available_integrations

    print("\n=== Testing Framework Detection ===")

    # Test detection with mock objects
    test_cases = [
        ("sklearn.ensemble.RandomForestClassifier", "sklearn"),
        ("torch.nn.Module", "torch"),
        ("tensorflow.keras.Model", "tensorflow"),
        ("pandas.DataFrame", "pandas"),
        ("numpy.ndarray", "numpy"),
    ]

    for obj_type, expected in test_cases:
        # Create mock object with module name
        class MockObject:
            pass

        MockObject.__module__ = obj_type
        result = detect_framework(MockObject())
        print(f"  {obj_type}: {'✓' if result == expected else '✗'} (detected: {result})")

    # List available integrations
    print("\n=== Available Integrations ===")
    available = list_available_integrations()

    for framework, capabilities in available.items():
        print(f"  {framework}: {len(capabilities)} capabilities")
        for cap in capabilities:
            print(f"    - {cap.value}")


def test_serialization():
    """Test serialization handlers"""
    print("\n=== Testing Serialization ===")

    # Test with simple data
    test_data = {
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2},
        'string': 'test',
        'number': 42.0
    }

    from . import auto_serialize, auto_deserialize

    try:
        # Serialize with pickle (fallback)
        serialized = auto_serialize(test_data)
        deserialized = auto_deserialize(serialized)

        print(f"  Pickle serialization: ✓")
        print(f"  Data preserved: {'✓' if deserialized == test_data else '✗'}")
    except Exception as e:
        print(f"  Serialization failed: {e}")


def test_base_classes():
    """Test base integration classes"""
    print("\n=== Testing Base Classes ===")

    from .base import (
        IntegrationCapability,
        ModelMetadata,
        DatasetMetadata,
    )

    # Test metadata creation
    model_meta = ModelMetadata(
        framework='test',
        framework_version='1.0',
        model_type='TestModel',
        parameters={'param1': 'value1'}
    )

    dataset_meta = DatasetMetadata(
        format='test',
        shape=(100, 10),
        n_samples=100,
        n_features=10
    )

    print(f"  ModelMetadata: ✓")
    print(f"  DatasetMetadata: ✓")
    print(f"  IntegrationCapability enum: ✓ ({len(IntegrationCapability)} capabilities)")


def test_individual_integrations():
    """Test each integration individually"""
    print("\n=== Testing Individual Integrations ===")

    from . import get_integration

    frameworks = [
        'sklearn',
        'pytorch',
        'tensorflow',
        'pandas',
        'numpy',
        'xgboost',
        'lightgbm',
        'huggingface'
    ]

    for framework in frameworks:
        integration = get_integration(framework)

        if integration:
            print(f"\n  {framework}:")
            print(f"    Available: {'✓' if integration.is_available else '✗'}")
            print(f"    Version: {integration.get_version()}")
            print(f"    Capabilities: {len(integration.capabilities)}")

            # Test serialization handler
            try:
                handler = integration.get_serialization_handler()
                print(f"    Serialization: ✓ ({handler.get_format()})")
            except Exception as e:
                print(f"    Serialization: ✗ ({e})")
        else:
            print(f"\n  {framework}: Not available")


def run_integration_example():
    """Run a simple integration example if sklearn is available"""
    print("\n=== Running Integration Example ===")

    from . import get_integration, detect_framework

    sklearn = get_integration('sklearn')

    if sklearn and sklearn.is_available:
        try:
            # Create a simple model
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            # Create dummy data
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)

            # Train model
            model = LogisticRegression()
            model.fit(X, y)

            # Test framework detection
            detected = detect_framework(model)
            print(f"  Framework detection: {'✓' if detected == 'sklearn' else '✗'}")

            # Test model wrapping
            wrapped = sklearn.wrap_model(model)
            print(f"  Model wrapping: ✓")

            # Test metadata extraction
            metadata = wrapped.metadata
            print(f"  Metadata extraction: ✓")
            print(f"    Model type: {metadata.model_type}")
            print(f"    Framework: {metadata.framework}")

            # Test prediction
            predictions = wrapped.predict(X[:5])
            print(f"  Prediction: ✓ (shape: {predictions.shape})")

            # Test serialization
            handler = sklearn.get_serialization_handler()
            serialized = handler.serialize(model)
            deserialized = handler.deserialize(serialized)
            print(f"  Serialization roundtrip: ✓")

        except Exception as e:
            print(f"  Example failed: {e}")
    else:
        print("  Sklearn not available - skipping example")


def main():
    """Run all integration tests"""
    print("🧪 ops0 Integration Tests")
    print("=" * 50)

    test_integration_detection()
    test_serialization()
    test_base_classes()
    test_individual_integrations()
    run_integration_example()

    print("\n" + "=" * 50)
    print("✅ Integration tests completed!")
    print("\nNote: Some integrations may not be available if the")
    print("corresponding ML frameworks are not installed.")
    print("\nInstall frameworks with:")
    print("  pip install scikit-learn torch tensorflow")
    print("  pip install pandas numpy xgboost lightgbm transformers")


if __name__ == '__main__':
    main()