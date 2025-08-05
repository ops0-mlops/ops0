"""
ops0 PyTorch Integration

Deep learning with PyTorch - automatic GPU detection, distributed training,
and optimized inference support.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import io

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


class PyTorchSerializer(SerializationHandler):
    """PyTorch model serialization handler"""

    def serialize(self, obj: Any) -> bytes:
        """Serialize PyTorch model/tensor"""
        import torch

        buffer = io.BytesIO()

        # Handle different PyTorch objects
        if isinstance(obj, torch.nn.Module):
            # Save entire model
            torch.save({
                'model_state_dict': obj.state_dict(),
                'model_class': obj.__class__.__name__,
                'model_architecture': str(obj),
            }, buffer)
        else:
            # Save tensor or other objects
            torch.save(obj, buffer)

        return buffer.getvalue()

    def deserialize(self, data: bytes) -> Any:
        """Deserialize PyTorch model/tensor"""
        import torch

        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location='cpu')

    def get_format(self) -> str:
        return "pytorch"


class PyTorchModelWrapper(ModelWrapper):
    """Wrapper for PyTorch models"""

    def __init__(self, model: Any, metadata: Optional[ModelMetadata] = None):
        super().__init__(model, metadata)
        self._device = None

    def _extract_metadata(self) -> ModelMetadata:
        """Extract metadata from PyTorch model"""
        import torch

        model_type = type(self.model).__name__

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Get input/output shapes if possible
        input_shape = None
        output_shape = None

        # Try to infer from first layer
        if hasattr(self.model, 'modules'):
            modules = list(self.model.modules())
            if len(modules) > 1:
                first_layer = modules[1]
                if hasattr(first_layer, 'in_features'):
                    input_shape = (None, first_layer.in_features)
                elif hasattr(first_layer, 'in_channels'):
                    input_shape = (None, first_layer.in_channels, None, None)

        return ModelMetadata(
            framework='pytorch',
            framework_version=torch.__version__,
            model_type=model_type,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters={
                'total_params': total_params,
                'trainable_params': trainable_params,
                'non_trainable_params': total_params - trainable_params,
            },
        )

    def predict(self, data: Any, **kwargs) -> Any:
        """Run inference using PyTorch model"""
        import torch

        # Set model to eval mode
        self.model.eval()

        # Handle device placement
        device = kwargs.get('device', self._get_device())
        self.model = self.model.to(device)

        # Convert data to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        data = data.to(device)

        # Run inference
        with torch.no_grad():
            output = self.model(data)

        # Return as numpy by default
        if kwargs.get('return_numpy', True):
            return output.cpu().numpy()
        return output

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'state_dict': self.model.state_dict(),
            'architecture': str(self.model),
        }

    def _get_device(self) -> Any:
        """Get appropriate device for model"""
        import torch

        if self._device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            else:
                self._device = torch.device('cpu')

        return self._device


class PyTorchDatasetWrapper(DatasetWrapper):
    """Wrapper for PyTorch datasets and tensors"""

    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from PyTorch data"""
        import torch

        if isinstance(self.data, torch.Tensor):
            shape = tuple(self.data.shape)
            dtype_str = str(self.data.dtype)
            size_bytes = self.data.element_size() * self.data.nelement()

            return DatasetMetadata(
                format='torch.Tensor',
                shape=shape,
                dtypes={'tensor': dtype_str},
                size_bytes=size_bytes,
                n_samples=shape[0] if len(shape) > 0 else 1,
                n_features=shape[1] if len(shape) > 1 else 1,
            )
        elif hasattr(self.data, '__len__') and hasattr(self.data, '__getitem__'):
            # PyTorch Dataset
            length = len(self.data)
            sample = self.data[0] if length > 0 else None

            return DatasetMetadata(
                format='torch.utils.data.Dataset',
                shape=(length,),
                n_samples=length,
            )
        else:
            raise ValueError(f"Unsupported PyTorch data type: {type(self.data)}")

    def to_numpy(self) -> Any:
        """Convert to numpy array"""
        import torch
        import numpy as np

        if isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        else:
            # Try to convert dataset to tensor first
            if hasattr(self.data, '__len__'):
                samples = [self.data[i] for i in range(len(self.data))]
                if samples and isinstance(samples[0], torch.Tensor):
                    return torch.stack(samples).cpu().numpy()

            return np.array(self.data)

    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        import torch
        from torch.utils.data import random_split

        if hasattr(self.data, '__len__'):
            # Dataset-like object
            total_size = len(self.data)
            train_size_n = int(total_size * train_size)
            test_size_n = total_size - train_size_n

            return random_split(self.data, [train_size_n, test_size_n])
        elif isinstance(self.data, torch.Tensor):
            # Tensor
            split_idx = int(len(self.data) * train_size)
            return self.data[:split_idx], self.data[split_idx:]
        else:
            raise ValueError("Cannot split this data type")

    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample n items from dataset"""
        import torch

        if random_state is not None:
            torch.manual_seed(random_state)

        if isinstance(self.data, torch.Tensor):
            indices = torch.randperm(len(self.data))[:n]
            return self.data[indices]
        elif hasattr(self.data, '__len__'):
            indices = torch.randperm(len(self.data))[:n].tolist()
            return [self.data[i] for i in indices]


class PyTorchIntegration(BaseIntegration):
    """Integration for PyTorch deep learning framework"""

    @property
    def is_available(self) -> bool:
        """Check if PyTorch is installed"""
        try:
            import torch
            return True
        except ImportError:
            return False

    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define PyTorch capabilities"""
        caps = [
            IntegrationCapability.TRAINING,
            IntegrationCapability.INFERENCE,
            IntegrationCapability.SERIALIZATION,
            IntegrationCapability.GPU_SUPPORT,
            IntegrationCapability.MODEL_REGISTRY,
        ]

        # Check for distributed support
        try:
            import torch.distributed
            caps.append(IntegrationCapability.DISTRIBUTED)
        except:
            pass

        return caps

    def get_version(self) -> str:
        """Get PyTorch version"""
        try:
            import torch
            return torch.__version__
        except ImportError:
            return "not installed"

    def wrap_model(self, model: Any) -> ModelWrapper:
        """Wrap PyTorch model"""
        return PyTorchModelWrapper(model)

    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap PyTorch dataset"""
        return PyTorchDatasetWrapper(data)

    def get_serialization_handler(self) -> SerializationHandler:
        """Get PyTorch serialization handler"""
        if self._serialization_handler is None:
            self._serialization_handler = PyTorchSerializer()
        return self._serialization_handler

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def optimize_for_inference(self, model: Any, **kwargs) -> Any:
        """Optimize model for inference"""
        import torch

        # Set to eval mode
        model.eval()

        # Try TorchScript compilation
        try:
            if kwargs.get('use_torchscript', True):
                # Need example input for tracing
                example_input = kwargs.get('example_input')
                if example_input is not None:
                    model = torch.jit.trace(model, example_input)
                    logger.info("Model compiled with TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}")

        # Try ONNX export if requested
        if kwargs.get('export_onnx', False):
            try:
                self._export_onnx(model, kwargs.get('example_input'))
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

        return model

    def get_resource_requirements(self, model: Any) -> Dict[str, Any]:
        """Estimate resource requirements"""
        import torch

        # Calculate model size
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size

        # Estimate requirements
        memory_mb = max(512, total_size // (1024 * 1024) * 4)  # 4x model size

        return {
            'cpu': '1000m' if total_size > 100_000_000 else '500m',
            'memory': f'{memory_mb}Mi',
            'gpu': torch.cuda.is_available(),
            'model_size_mb': total_size // (1024 * 1024),
        }

    def create_data_loader(self, dataset: Any, batch_size: int = 32, **kwargs) -> Any:
        """Create PyTorch DataLoader"""
        import torch.utils.data as data

        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=kwargs.get('shuffle', True),
            num_workers=kwargs.get('num_workers', 0),
            pin_memory=kwargs.get('pin_memory', True),
            drop_last=kwargs.get('drop_last', False),
        )

    def distributed_setup(self, rank: int, world_size: int, **kwargs) -> None:
        """Setup distributed training"""
        import torch.distributed as dist

        backend = kwargs.get('backend', 'nccl' if self._check_gpu_available() else 'gloo')

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

        logger.info(f"Distributed training initialized: rank {rank}/{world_size}")

    def profile_model(self, model: Any, input_data: Any) -> Dict[str, Any]:
        """Profile model performance"""
        import torch
        import time

        # Warmup
        for _ in range(10):
            _ = model(input_data)

        # Profile forward pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.time()
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
        ) as prof:
            for _ in range(100):
                _ = model(input_data)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.time() - start_time

        # Get profiling results
        key_averages = prof.key_averages()

        return {
            'avg_forward_time_ms': (total_time / 100) * 1000,
            'profile_summary': str(key_averages),
            'top_operations': [
                {
                    'name': item.key,
                    'cpu_time_ms': item.cpu_time_total / 1000,
                    'cuda_time_ms': item.cuda_time_total / 1000 if torch.cuda.is_available() else 0,
                }
                for item in sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:5]
            ],
        }

    def _export_onnx(self, model: Any, example_input: Any) -> bytes:
        """Export model to ONNX format"""
        import torch
        import io

        buffer = io.BytesIO()

        torch.onnx.export(
            model,
            example_input,
            buffer,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )

        return buffer.getvalue()