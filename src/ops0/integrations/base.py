"""
ops0 Base Integration

Abstract base classes for ML framework integrations.
Provides a consistent interface for all ML frameworks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IntegrationCapability(Enum):
    """Capabilities that an integration can provide"""
    TRAINING = "training"
    INFERENCE = "inference"
    SERIALIZATION = "serialization"
    GPU_SUPPORT = "gpu_support"
    DISTRIBUTED = "distributed"
    AUTOML = "automl"
    VISUALIZATION = "visualization"
    EXPERIMENT_TRACKING = "experiment_tracking"
    MODEL_REGISTRY = "model_registry"
    DATA_VALIDATION = "data_validation"
    FEATURE_STORE = "feature_store"


@dataclass
class ModelMetadata:
    """Metadata for ML models"""
    framework: str
    framework_version: str
    model_type: str
    input_shape: Optional[Union[Tuple, Dict]] = None
    output_shape: Optional[Union[Tuple, Dict]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)


@dataclass
class DatasetMetadata:
    """Metadata for datasets"""
    format: str
    shape: Tuple
    dtypes: Dict[str, str] = field(default_factory=dict)
    size_bytes: int = 0
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    target_column: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class SerializationHandler(ABC):
    """Abstract serialization handler for ML objects"""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize ML object to bytes"""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to ML object"""
        pass

    @abstractmethod
    def get_format(self) -> str:
        """Get serialization format name"""
        pass


class ModelWrapper(ABC):
    """Abstract wrapper for ML models"""

    def __init__(self, model: Any, metadata: Optional[ModelMetadata] = None):
        self.model = model
        self.metadata = metadata or self._extract_metadata()

    @abstractmethod
    def _extract_metadata(self) -> ModelMetadata:
        """Extract metadata from model"""
        pass

    @abstractmethod
    def predict(self, data: Any, **kwargs) -> Any:
        """Run inference on data"""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        pass

    def to_ops0(self) -> Dict[str, Any]:
        """Convert to ops0 standard format"""
        return {
            'model': self.model,
            'metadata': self.metadata,
            'framework': self.metadata.framework,
        }


class DatasetWrapper(ABC):
    """Abstract wrapper for datasets"""

    def __init__(self, data: Any, metadata: Optional[DatasetMetadata] = None):
        self.data = data
        self.metadata = metadata or self._extract_metadata()

    @abstractmethod
    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from dataset"""
        pass

    @abstractmethod
    def to_numpy(self) -> Any:
        """Convert to numpy array"""
        pass

    @abstractmethod
    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        pass

    @abstractmethod
    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample n rows from dataset"""
        pass


class BaseIntegration(ABC):
    """Base class for all ML framework integrations"""

    def __init__(self):
        self.framework_name = self.__class__.__name__.replace('Integration', '').lower()
        self._serialization_handler = None
        self._capabilities = self._define_capabilities()

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the framework is installed and available"""
        pass

    @abstractmethod
    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define what capabilities this integration provides"""
        pass

    @property
    def capabilities(self) -> List[IntegrationCapability]:
        """Get integration capabilities"""
        return self._capabilities

    @abstractmethod
    def get_version(self) -> str:
        """Get framework version"""
        pass

    @abstractmethod
    def wrap_model(self, model: Any) -> ModelWrapper:
        """Wrap a model with ops0 interface"""
        pass

    @abstractmethod
    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap a dataset with ops0 interface"""
        pass

    @abstractmethod
    def get_serialization_handler(self) -> SerializationHandler:
        """Get serialization handler for this framework"""
        pass

    def optimize_for_inference(self, model: Any, **kwargs) -> Any:
        """Optimize model for inference (optional)"""
        logger.info(f"No inference optimization available for {self.framework_name}")
        return model

    def get_resource_requirements(self, model: Any) -> Dict[str, Any]:
        """Estimate resource requirements for model"""
        return {
            'cpu': '500m',
            'memory': '1Gi',
            'gpu': False,
        }

    def validate_environment(self) -> Dict[str, bool]:
        """Validate that the environment is properly set up"""
        checks = {
            'framework_available': self.is_available,
            'version_compatible': True,
        }

        if IntegrationCapability.GPU_SUPPORT in self.capabilities:
            checks['gpu_available'] = self._check_gpu_available()

        return checks

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available (override in subclasses)"""
        return False

    def register_model(self, model: Any, name: str, **kwargs) -> str:
        """Register model with ops0 registry"""
        from ops0.core.registry import registry

        wrapped = self.wrap_model(model)
        model_id = registry.register_model(
            name=name,
            model_data=model,
            framework=self.framework_name,
            framework_version=self.get_version(),
            model_type=wrapped.metadata.model_type,
            **kwargs
        )

        logger.info(f"Registered {self.framework_name} model '{name}' with ID: {model_id}")
        return model_id

    def load_model(self, model_id: str) -> Any:
        """Load model from ops0 registry"""
        from ops0.core.registry import registry

        model_data = registry.load_model(model_id)
        return model_data

    def to_onnx(self, model: Any, sample_input: Any = None) -> bytes:
        """Convert model to ONNX format (optional)"""
        raise NotImplementedError(f"ONNX conversion not implemented for {self.framework_name}")

    def explain_predictions(self, model: Any, data: Any, **kwargs) -> Dict[str, Any]:
        """Generate model explanations (optional)"""
        raise NotImplementedError(f"Model explanations not implemented for {self.framework_name}")

    def profile_inference(self, model: Any, data: Any) -> Dict[str, float]:
        """Profile model inference performance"""
        import time

        # Warm up
        wrapped = self.wrap_model(model)
        _ = wrapped.predict(data)

        # Time inference
        times = []
        for _ in range(10):
            start = time.time()
            _ = wrapped.predict(data)
            times.append(time.time() - start)

        import statistics
        return {
            'mean_ms': statistics.mean(times) * 1000,
            'std_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
        }