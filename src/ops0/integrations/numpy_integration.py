"""
ops0 NumPy Integration

Numerical computing with NumPy - efficient array operations,
automatic memory mapping for large arrays, and optimized serialization.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import io

from .base import (
    BaseIntegration,
    DatasetWrapper,
    IntegrationCapability,
    SerializationHandler,
    DatasetMetadata,
)

logger = logging.getLogger(__name__)


class NumpySerializer(SerializationHandler):
    """NumPy array serialization handler"""

    def __init__(self, allow_pickle: bool = False, compression: bool = True):
        """
        Initialize serializer.

        Args:
            allow_pickle: Allow pickling of objects (less secure)
            compression: Use compression for serialization
        """
        self.allow_pickle = allow_pickle
        self.compression = compression

    def serialize(self, obj: Any) -> bytes:
        """Serialize numpy array"""
        import numpy as np

        buffer = io.BytesIO()

        if self.compression:
            np.savez_compressed(buffer, array=obj)
        else:
            np.save(buffer, obj, allow_pickle=self.allow_pickle)

        return buffer.getvalue()

    def deserialize(self, data: bytes) -> Any:
        """Deserialize numpy array"""
        import numpy as np

        buffer = io.BytesIO(data)

        if self.compression:
            loaded = np.load(buffer, allow_pickle=self.allow_pickle)
            return loaded['array']
        else:
            return np.load(buffer, allow_pickle=self.allow_pickle)

    def get_format(self) -> str:
        return "numpy-compressed" if self.compression else "numpy"


class NumpyDatasetWrapper(DatasetWrapper):
    """Wrapper for NumPy arrays"""

    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from numpy array"""
        import numpy as np

        if not isinstance(self.data, np.ndarray):
            # Try to convert to numpy array
            self.data = np.asarray(self.data)

        shape = self.data.shape
        dtype_str = str(self.data.dtype)
        size_bytes = self.data.nbytes

        # Calculate statistics for numeric arrays
        stats = {}
        if np.issubdtype(self.data.dtype, np.number) and self.data.size > 0:
            stats = {
                'mean': float(np.mean(self.data)),
                'std': float(np.std(self.data)),
                'min': float(np.min(self.data)),
                'max': float(np.max(self.data)),
                'percentiles': {
                    '25': float(np.percentile(self.data, 25)),
                    '50': float(np.percentile(self.data, 50)),
                    '75': float(np.percentile(self.data, 75)),
                }
            }

        return DatasetMetadata(
            format='numpy.ndarray',
            shape=shape,
            dtypes={'array': dtype_str},
            size_bytes=size_bytes,
            n_samples=shape[0] if len(shape) > 0 else 1,
            n_features=shape[1] if len(shape) > 1 else 1,
            stats=stats,
        )

    def to_numpy(self) -> Any:
        """Already numpy array"""
        return self.data

    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        import numpy as np

        if len(self.data) == 0:
            return np.array([]), np.array([])

        # Random shuffle indices
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)

        split_idx = int(len(indices) * train_size)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        return self.data[train_indices], self.data[test_indices]

    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample n items from array"""
        import numpy as np

        if random_state is not None:
            np.random.seed(random_state)

        indices = np.random.choice(len(self.data), size=min(n, len(self.data)), replace=False)
        return self.data[indices]


class NumpyIntegration(BaseIntegration):
    """Integration for NumPy numerical computing"""

    def __init__(self):
        super().__init__()
        self._memory_map_threshold = 100 * 1024 * 1024  # 100MB

    @property
    def is_available(self) -> bool:
        """Check if numpy is installed"""
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define numpy capabilities"""
        return [
            IntegrationCapability.SERIALIZATION,
            IntegrationCapability.DATA_VALIDATION,
        ]

    def get_version(self) -> str:
        """Get numpy version"""
        try:
            import numpy as np
            return np.__version__
        except ImportError:
            return "not installed"

    def wrap_model(self, model: Any) -> None:
        """NumPy doesn't have models"""
        raise NotImplementedError("NumPy integration doesn't support models")

    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap numpy array"""
        return NumpyDatasetWrapper(data)

    def get_serialization_handler(self) -> SerializationHandler:
        """Get numpy serialization handler"""
        if self._serialization_handler is None:
            self._serialization_handler = NumpySerializer(compression=True)
        return self._serialization_handler

    def create_memmap(self, shape: Tuple, dtype: Any = 'float32', mode: str = 'w+') -> Any:
        """Create memory-mapped array for large datasets"""
        import numpy as np
        import tempfile

        # Create temporary file for memmap
        temp_file = tempfile.NamedTemporaryFile(delete=False)

        return np.memmap(
            temp_file.name,
            dtype=dtype,
            mode=mode,
            shape=shape
        )

    def optimize_array(self, arr: Any, **kwargs) -> Any:
        """Optimize array memory usage"""
        import numpy as np

        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        # Check if we should use memory mapping
        if arr.nbytes > self._memory_map_threshold and kwargs.get('use_memmap', True):
            logger.info(f"Converting large array ({arr.nbytes / 1024 ** 2:.1f} MB) to memory map")

            memmap = self.create_memmap(arr.shape, dtype=arr.dtype)
            memmap[:] = arr
            return memmap

        # Optimize dtype if possible
        if np.issubdtype(arr.dtype, np.integer):
            # Check if we can use a smaller integer type
            min_val, max_val = arr.min(), arr.max()

            if min_val >= 0:
                if max_val <= 255:
                    return arr.astype(np.uint8)
                elif max_val <= 65535:
                    return arr.astype(np.uint16)
                elif max_val <= 4294967295:
                    return arr.astype(np.uint32)
            else:
                if -128 <= min_val and max_val <= 127:
                    return arr.astype(np.int8)
                elif -32768 <= min_val and max_val <= 32767:
                    return arr.astype(np.int16)

        elif np.issubdtype(arr.dtype, np.floating):
            # Check if float32 is sufficient
            if arr.dtype == np.float64 and kwargs.get('allow_float32', True):
                # Check precision requirements
                if np.allclose(arr, arr.astype(np.float32), rtol=1e-6):
                    return arr.astype(np.float32)

        return arr

    def validate_array(self, arr: Any, schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate numpy array"""
        import numpy as np

        if not isinstance(arr, np.ndarray):
            try:
                arr = np.asarray(arr)
            except:
                return {'valid': False, 'errors': ['Cannot convert to numpy array']}

        errors = []
        warnings = []

        # Check for NaN/Inf values
        if np.issubdtype(arr.dtype, np.number):
            if np.any(np.isnan(arr)):
                warnings.append(f"Array contains {np.sum(np.isnan(arr))} NaN values")

            if np.any(np.isinf(arr)):
                warnings.append(f"Array contains {np.sum(np.isinf(arr))} infinite values")

        # Schema validation
        if schema:
            # Check shape
            if 'shape' in schema:
                expected_shape = schema['shape']
                if len(expected_shape) != len(arr.shape):
                    errors.append(f"Expected {len(expected_shape)} dimensions, got {len(arr.shape)}")
                else:
                    for i, (expected, actual) in enumerate(zip(expected_shape, arr.shape)):
                        if expected is not None and expected != actual:
                            errors.append(f"Dimension {i}: expected {expected}, got {actual}")

            # Check dtype
            if 'dtype' in schema:
                expected_dtype = np.dtype(schema['dtype'])
                if arr.dtype != expected_dtype:
                    errors.append(f"Expected dtype {expected_dtype}, got {arr.dtype}")

            # Check value ranges
            if 'min' in schema and np.any(arr < schema['min']):
                errors.append(f"Array contains values below minimum {schema['min']}")

            if 'max' in schema and np.any(arr > schema['max']):
                errors.append(f"Array contains values above maximum {schema['max']}")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'memory_usage_mb': arr.nbytes / 1024 ** 2,
                'is_c_contiguous': arr.flags['C_CONTIGUOUS'],
                'is_f_contiguous': arr.flags['F_CONTIGUOUS'],
            }
        }

    def array_operations(self) -> Dict[str, Any]:
        """Get optimized array operations"""
        import numpy as np

        return {
            'linalg': np.linalg,
            'fft': np.fft,
            'random': np.random,
            'ma': np.ma,  # Masked arrays
        }

    def parallel_apply(self, func: callable, arr: Any, axis: int = 0, **kwargs) -> Any:
        """Apply function in parallel using numpy"""
        import numpy as np

        # For simple operations, numpy is already optimized
        return np.apply_along_axis(func, axis, arr)

    def sliding_window(self, arr: Any, window_size: int, step: int = 1) -> Any:
        """Create sliding window view of array"""
        import numpy as np

        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        # Use numpy's stride tricks for efficient windowing
        from numpy.lib.stride_tricks import sliding_window_view

        try:
            # NumPy >= 1.20
            return sliding_window_view(arr, window_size)[::step]
        except AttributeError:
            # Fallback for older NumPy
            shape = (arr.shape[0] - window_size + 1, window_size)
            strides = (arr.strides[0] * step, arr.strides[0])

            return np.lib.stride_tricks.as_strided(
                arr,
                shape=shape,
                strides=strides
            )

    def save_compressed(self, arrays: Dict[str, Any], filepath: str) -> None:
        """Save multiple arrays in compressed format"""
        import numpy as np

        np.savez_compressed(filepath, **arrays)
        logger.info(f"Saved {len(arrays)} arrays to {filepath}")

    def load_compressed(self, filepath: str) -> Dict[str, Any]:
        """Load compressed arrays"""
        import numpy as np

        with np.load(filepath, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}