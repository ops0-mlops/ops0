"""
ops0 ML Framework Integrations

Seamless integration with popular ML frameworks.
Zero configuration, automatic optimization, and transparent serialization.
"""

from typing import Dict, Type, Optional, Any
import logging

from .base import (
    BaseIntegration,
    ModelWrapper,
    DatasetWrapper,
    IntegrationCapability,
    SerializationHandler,
)

# Lazy imports for optional dependencies
_INTEGRATIONS: Dict[str, Type[BaseIntegration]] = {}
_LAZY_IMPORTS = {
    'sklearn': 'ScikitLearnIntegration',
    'torch': 'PyTorchIntegration',
    'tensorflow': 'TensorFlowIntegration',
    'pandas': 'PandasIntegration',
    'numpy': 'NumpyIntegration',
    'xgboost': 'XGBoostIntegration',
    'lightgbm': 'LightGBMIntegration',
    'huggingface': 'HuggingFaceIntegration',
}

logger = logging.getLogger(__name__)


def get_integration(framework: str) -> Optional[BaseIntegration]:
    """
    Get integration for a specific framework.

    Args:
        framework: Name of the ML framework

    Returns:
        Integration instance or None if not available
    """
    if framework in _INTEGRATIONS:
        return _INTEGRATIONS[framework]()

    # Try lazy loading
    if framework in _LAZY_IMPORTS:
        try:
            module_name = f".{framework}_integration"
            class_name = _LAZY_IMPORTS[framework]

            module = __import__(module_name, fromlist=[class_name], package='ops0.integrations')
            integration_class = getattr(module, class_name)

            _INTEGRATIONS[framework] = integration_class
            return integration_class()

        except ImportError as e:
            logger.debug(f"Integration for {framework} not available: {e}")
            return None

    return None


def detect_framework(obj: Any) -> Optional[str]:
    """
    Automatically detect ML framework from object.

    Args:
        obj: Object to analyze

    Returns:
        Framework name or None
    """
    obj_type = type(obj).__module__

    # Check module prefix
    framework_prefixes = {
        'sklearn': 'sklearn',
        'torch': 'torch',
        'tensorflow': 'tensorflow',
        'tf': 'tensorflow',
        'keras': 'tensorflow',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'transformers': 'huggingface',
    }

    for prefix, framework in framework_prefixes.items():
        if obj_type.startswith(prefix):
            return framework

    # Check specific attributes
    if hasattr(obj, 'fit') and hasattr(obj, 'predict'):
        # Likely a scikit-learn compatible model
        return 'sklearn'
    elif hasattr(obj, 'forward') and hasattr(obj, 'parameters'):
        # Likely a PyTorch model
        return 'torch'
    elif hasattr(obj, 'compile') and hasattr(obj, 'fit'):
        # Likely a Keras/TensorFlow model
        return 'tensorflow'

    return None


def auto_serialize(obj: Any) -> bytes:
    """
    Automatically serialize ML object using appropriate integration.

    Args:
        obj: Object to serialize

    Returns:
        Serialized bytes
    """
    framework = detect_framework(obj)
    if framework:
        integration = get_integration(framework)
        if integration:
            handler = integration.get_serialization_handler()
            return handler.serialize(obj)

    # Fallback to pickle
    import pickle
    return pickle.dumps(obj)


def auto_deserialize(data: bytes, framework: str = None) -> Any:
    """
    Automatically deserialize ML object.

    Args:
        data: Serialized data
        framework: Optional framework hint

    Returns:
        Deserialized object
    """
    if framework:
        integration = get_integration(framework)
        if integration:
            handler = integration.get_serialization_handler()
            return handler.deserialize(data)

    # Try to detect from data or fallback to pickle
    import pickle
    return pickle.loads(data)


def list_available_integrations() -> Dict[str, IntegrationCapability]:
    """
    List all available integrations and their capabilities.

    Returns:
        Dictionary of framework -> capabilities
    """
    available = {}

    for framework in _LAZY_IMPORTS:
        integration = get_integration(framework)
        if integration:
            available[framework] = integration.capabilities

    return available


# Export main classes and functions
__all__ = [
    # Base classes
    'BaseIntegration',
    'ModelWrapper',
    'DatasetWrapper',
    'IntegrationCapability',
    'SerializationHandler',

    # Functions
    'get_integration',
    'detect_framework',
    'auto_serialize',
    'auto_deserialize',
    'list_available_integrations',
]