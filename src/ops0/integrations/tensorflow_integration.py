"""
ops0 TensorFlow Integration

TensorFlow and Keras integration with automatic optimization,
SavedModel support, and TensorFlow Serving compatibility.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import tempfile
import os

from .base import (
    BaseIntegration,
    ModelWrapper,
    DatasetWrapper,
    IntegrationCapability,
    SerializationHandler,
    ModelMetadata,
    DatasetMetadata,
)

logger = logging.getLogger(__name__)


class TensorFlowSerializer(SerializationHandler):
    """TensorFlow model serialization handler"""

    def serialize(self, obj: Any) -> bytes:
        """Serialize TensorFlow/Keras model"""
        import tensorflow as tf

        buffer = io.BytesIO()

        # Use SavedModel format for better compatibility
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")

            if isinstance(obj, tf.keras.Model):
                # Keras model
                obj.save(model_path, save_format='tf')
            else:
                # Raw TensorFlow model
                tf.saved_model.save(obj, model_path)

            # Package as tar.gz
            import tarfile
            with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
                tar.add(model_path, arcname='model')

        return buffer.getvalue()

    def deserialize(self, data: bytes) -> Any:
        """Deserialize TensorFlow/Keras model"""
        import tensorflow as tf
        import tarfile

        buffer = io.BytesIO(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tar.gz
            with tarfile.open(fileobj=buffer, mode='r:gz') as tar:
                tar.extractall(tmpdir)

            model_path = os.path.join(tmpdir, 'model')

            # Try loading as Keras model first
            try:
                return tf.keras.models.load_model(model_path)
            except:
                # Fallback to SavedModel
                return tf.saved_model.load(model_path)

    def get_format(self) -> str:
        return "tensorflow-savedmodel"


class TensorFlowModelWrapper(ModelWrapper):
    """Wrapper for TensorFlow/Keras models"""

    def _extract_metadata(self) -> ModelMetadata:
        """Extract metadata from TensorFlow model"""
        import tensorflow as tf

        model_type = type(self.model).__name__

        # Handle Keras models
        if isinstance(self.model, tf.keras.Model):
            # Get input/output shapes
            input_shape = None
            output_shape = None

            if hasattr(self.model, 'input_shape'):
                input_shape = self.model.input_shape
            elif hasattr(self.model, 'inputs'):
                input_shape = [inp.shape.as_list() for inp in self.model.inputs]

            if hasattr(self.model, 'output_shape'):
                output_shape = self.model.output_shape
            elif hasattr(self.model, 'outputs'):
                output_shape = [out.shape.as_list() for out in self.model.outputs]

            # Count parameters
            total_params = self.model.count_params() if hasattr(self.model, 'count_params') else 0

            # Get optimizer info if compiled
            optimizer_config = None
            if hasattr(self.model, 'optimizer') and self.model.optimizer:
                optimizer_config = {
                    'name': self.model.optimizer.__class__.__name__,
                    'config': self.model.optimizer.get_config(),
                }

            return ModelMetadata(
                framework='tensorflow',
                framework_version=tf.__version__,
                model_type=model_type,
                input_shape=input_shape,
                output_shape=output_shape,
                parameters={
                    'total_params': total_params,
                    'layers': len(self.model.layers) if hasattr(self.model, 'layers') else 0,
                    'optimizer': optimizer_config,
                },
            )
        else:
            # Raw TensorFlow model
            return ModelMetadata(
                framework='tensorflow',
                framework_version=tf.__version__,
                model_type=model_type,
            )

    def predict(self, data: Any, **kwargs) -> Any:
        """Run inference using TensorFlow model"""
        import tensorflow as tf
        import numpy as np

        # Convert to tensor if needed
        if not isinstance(data, tf.Tensor):
            data = tf.constant(data)

        # Run inference
        if hasattr(self.model, 'predict'):
            # Keras model
            output = self.model.predict(data, **kwargs)
        else:
            # Raw TF model
            output = self.model(data)

        # Convert to numpy if requested
        if kwargs.get('return_numpy', True):
            if isinstance(output, tf.Tensor):
                return output.numpy()
            else:
                return np.array(output)

        return output

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            'model_config': str(self.model.get_config()) if hasattr(self.model, 'get_config') else None,
        }

        if hasattr(self.model, 'summary'):
            import io
            stream = io.StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            params['summary'] = stream.getvalue()

        return params


class TensorFlowDatasetWrapper(DatasetWrapper):
    """Wrapper for TensorFlow datasets"""

    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from TensorFlow data"""
        import tensorflow as tf

        if isinstance(self.data, tf.Tensor):
            shape = self.data.shape.as_list()
            dtype_str = str(self.data.dtype)

            return DatasetMetadata(
                format='tf.Tensor',
                shape=tuple(shape),
                dtypes={'tensor': dtype_str},
                n_samples=shape[0] if len(shape) > 0 else 1,
                n_features=shape[1] if len(shape) > 1 else 1,
            )
        elif isinstance(self.data, tf.data.Dataset):
            # TensorFlow Dataset
            element_spec = self.data.element_spec

            # Try to get cardinality
            cardinality = tf.data.experimental.cardinality(self.data)
            n_samples = cardinality.numpy() if cardinality != tf.data.experimental.INFINITE_CARDINALITY else -1

            return DatasetMetadata(
                format='tf.data.Dataset',
                shape=(n_samples,),
                n_samples=n_samples,
                dtypes={'element_spec': str(element_spec)},
            )
        else:
            raise ValueError(f"Unsupported TensorFlow data type: {type(self.data)}")

    def to_numpy(self) -> Any:
        """Convert to numpy array"""
        import tensorflow as tf
        import numpy as np

        if isinstance(self.data, tf.Tensor):
            return self.data.numpy()
        elif isinstance(self.data, tf.data.Dataset):
            # Convert dataset to numpy (warning: loads all data into memory)
            arrays = []
            for batch in self.data:
                if isinstance(batch, tuple):
                    arrays.append(batch[0].numpy())
                else:
                    arrays.append(batch.numpy())
            return np.concatenate(arrays, axis=0)
        else:
            return np.array(self.data)

    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        import tensorflow as tf

        if isinstance(self.data, tf.Tensor):
            split_idx = int(len(self.data) * train_size)
            return self.data[:split_idx], self.data[split_idx:]
        elif isinstance(self.data, tf.data.Dataset):
            # Calculate split size
            total_size = tf.data.experimental.cardinality(self.data).numpy()
            train_size_n = int(total_size * train_size) if total_size > 0 else 1000

            train_dataset = self.data.take(train_size_n)
            test_dataset = self.data.skip(train_size_n)

            return train_dataset, test_dataset
        else:
            raise ValueError("Cannot split this data type")

    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample n items from dataset"""
        import tensorflow as tf

        if random_state is not None:
            tf.random.set_seed(random_state)

        if isinstance(self.data, tf.Tensor):
            indices = tf.random.shuffle(tf.range(tf.shape(self.data)[0]))[:n]
            return tf.gather(self.data, indices)
        elif isinstance(self.data, tf.data.Dataset):
            return self.data.shuffle(buffer_size=10000).take(n)


class TensorFlowIntegration(BaseIntegration):
    """Integration for TensorFlow and Keras"""

    @property
    def is_available(self) -> bool:
        """Check if TensorFlow is installed"""
        try:
            import tensorflow as tf
            return True
        except ImportError:
            return False

    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define TensorFlow capabilities"""
        caps = [
            IntegrationCapability.TRAINING,
            IntegrationCapability.INFERENCE,
            IntegrationCapability.SERIALIZATION,
            IntegrationCapability.GPU_SUPPORT,
            IntegrationCapability.MODEL_REGISTRY,
            IntegrationCapability.VISUALIZATION,
        ]

        # Check for distributed support
        try:
            import tensorflow as tf
            if hasattr(tf.distribute, 'MirroredStrategy'):
                caps.append(IntegrationCapability.DISTRIBUTED)
        except:
            pass

        return caps

    def get_version(self) -> str:
        """Get TensorFlow version"""
        try:
            import tensorflow as tf
            return tf.__version__
        except ImportError:
            return "not installed"

    def wrap_model(self, model: Any) -> ModelWrapper:
        """Wrap TensorFlow model"""
        return TensorFlowModelWrapper(model)

    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap TensorFlow dataset"""
        return TensorFlowDatasetWrapper(data)

    def get_serialization_handler(self) -> SerializationHandler:
        """Get TensorFlow serialization handler"""
        if self._serialization_handler is None:
            self._serialization_handler = TensorFlowSerializer()
        return self._serialization_handler

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except:
            return False

    def optimize_for_inference(self, model: Any, **kwargs) -> Any:
        """Optimize model for inference"""
        import tensorflow as tf

        if not isinstance(model, tf.keras.Model):
            logger.warning("Optimization only supports Keras models")
            return model

        # Compile with optimizations
        if kwargs.get('compile_optimization', True):
            model.compile(
                optimizer='adam',
                loss=model.loss if hasattr(model, 'loss') else None,
                metrics=model.metrics if hasattr(model, 'metrics') else None,
                jit_compile=True,  # XLA compilation
            )

        # Convert to TensorFlow Lite if requested
        if kwargs.get('convert_to_tflite', False):
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()

                # Save to file if path provided
                if 'tflite_path' in kwargs:
                    with open(kwargs['tflite_path'], 'wb') as f:
                        f.write(tflite_model)

                logger.info("Model converted to TensorFlow Lite")
            except Exception as e:
                logger.warning(f"TFLite conversion failed: {e}")

        return model

    def get_resource_requirements(self, model: Any) -> Dict[str, Any]:
        """Estimate resource requirements"""
        import tensorflow as tf

        if isinstance(model, tf.keras.Model):
            # Get model size
            total_params = model.count_params()
            param_bytes = total_params * 4  # Assuming float32

            # Estimate memory (conservative)
            memory_mb = max(512, (param_bytes // (1024 * 1024)) * 4)

            return {
                'cpu': '1000m' if total_params > 10_000_000 else '500m',
                'memory': f'{memory_mb}Mi',
                'gpu': self._check_gpu_available(),
                'model_size_mb': param_bytes // (1024 * 1024),
            }
        else:
            return super().get_resource_requirements(model)

    def create_dataset(self, data: Any, labels: Any = None, batch_size: int = 32, **kwargs) -> Any:
        """Create TensorFlow Dataset"""
        import tensorflow as tf

        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(data)

        # Apply transformations
        if kwargs.get('shuffle', True):
            dataset = dataset.shuffle(buffer_size=kwargs.get('buffer_size', 10000))

        dataset = dataset.batch(batch_size)

        if kwargs.get('prefetch', True):
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def distributed_strategy(self, strategy_type: str = 'mirrored') -> Any:
        """Get distributed training strategy"""
        import tensorflow as tf

        strategies = {
            'mirrored': tf.distribute.MirroredStrategy,
            'multi_worker': tf.distribute.MultiWorkerMirroredStrategy,
            'tpu': tf.distribute.TPUStrategy,
            'parameter_server': tf.distribute.ParameterServerStrategy,
        }

        if strategy_type not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_type}")

        return strategies[strategy_type]()

    def profile_model(self, model: Any, input_data: Any, steps: int = 100) -> Dict[str, Any]:
        """Profile model performance"""
        import tensorflow as tf
        import time

        if not isinstance(model, tf.keras.Model):
            return super().profile_model(model, input_data)

        # Warmup
        for _ in range(10):
            _ = model(input_data, training=False)

        # Profile
        times = []
        for _ in range(steps):
            start = time.time()
            _ = model(input_data, training=False)
            times.append(time.time() - start)

        import statistics

        # Memory profiling
        memory_info = {}
        if self._check_gpu_available():
            try:
                gpu_device = tf.config.list_physical_devices('GPU')[0]
                memory_info = tf.config.experimental.get_memory_info(gpu_device)
            except:
                pass

        return {
            'mean_inference_ms': statistics.mean(times) * 1000,
            'std_inference_ms': statistics.stdev(times) * 1000,
            'min_inference_ms': min(times) * 1000,
            'max_inference_ms': max(times) * 1000,
            'memory_info': memory_info,
        }

    def export_for_serving(self, model: Any, export_path: str, **kwargs) -> None:
        """Export model for TensorFlow Serving"""
        import tensorflow as tf

        if isinstance(model, tf.keras.Model):
            # Save in SavedModel format for TF Serving
            version = kwargs.get('version', 1)
            export_path = os.path.join(export_path, str(version))

            tf.saved_model.save(model, export_path)
            logger.info(f"Model exported for TensorFlow Serving: {export_path}")
        else:
            raise ValueError("Only Keras models can be exported for serving")