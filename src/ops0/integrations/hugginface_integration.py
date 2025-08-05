"""
ops0 Hugging Face Integration

Transformer models and datasets - automatic model downloading,
optimized inference, and seamless integration with the Hugging Face ecosystem.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import json

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


class HuggingFaceSerializer(SerializationHandler):
    """Hugging Face model serialization handler"""

    def serialize(self, obj: Any) -> bytes:
        """Serialize Hugging Face model/tokenizer"""
        import tempfile
        import tarfile
        import io
        import os

        buffer = io.BytesIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model and tokenizer
            if hasattr(obj, 'save_pretrained'):
                obj.save_pretrained(tmpdir)
            else:
                raise ValueError("Object must have save_pretrained method")

            # Create tar archive
            with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, tmpdir)
                        tar.add(file_path, arcname=arcname)

        return buffer.getvalue()

    def deserialize(self, data: bytes) -> Any:
        """Deserialize Hugging Face model/tokenizer"""
        import tempfile
        import tarfile
        import io
        from transformers import AutoModel, AutoTokenizer

        buffer = io.BytesIO(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tar archive
            with tarfile.open(fileobj=buffer, mode='r:gz') as tar:
                tar.extractall(tmpdir)

            # Try to load as model
            try:
                return AutoModel.from_pretrained(tmpdir)
            except:
                # Try as tokenizer
                try:
                    return AutoTokenizer.from_pretrained(tmpdir)
                except:
                    raise ValueError("Could not load as model or tokenizer")

    def get_format(self) -> str:
        return "huggingface-pretrained"


class HuggingFaceModelWrapper(ModelWrapper):
    """Wrapper for Hugging Face models"""

    def __init__(self, model: Any, tokenizer: Any = None, metadata: Optional[ModelMetadata] = None):
        super().__init__(model, metadata)
        self.tokenizer = tokenizer

    def _extract_metadata(self) -> ModelMetadata:
        """Extract metadata from Hugging Face model"""
        import transformers

        model_type = type(self.model).__name__

        # Get model config
        config = {}
        if hasattr(self.model, 'config'):
            config = self.model.config.to_dict()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Model architecture info
        architecture = []
        if hasattr(self.model, 'config'):
            architecture = getattr(self.model.config, 'architectures', [])

        return ModelMetadata(
            framework='huggingface',
            framework_version=transformers.__version__,
            model_type=model_type,
            parameters={
                'total_params': total_params,
                'trainable_params': trainable_params,
                'config': config,
                'architectures': architecture,
            },
        )

    def predict(self, data: Any, **kwargs) -> Any:
        """Run inference using Hugging Face model"""
        import torch

        # Handle text input with tokenizer
        if self.tokenizer and isinstance(data, (str, list)):
            inputs = self.tokenizer(
                data,
                return_tensors='pt',
                padding=kwargs.get('padding', True),
                truncation=kwargs.get('truncation', True),
                max_length=kwargs.get('max_length', 512),
            )
        else:
            inputs = data

        # Move to device
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)

        # Process outputs based on task
        if kwargs.get('return_dict', True):
            return outputs
        else:
            # Return logits/embeddings
            if hasattr(outputs, 'logits'):
                return outputs.logits.cpu().numpy()
            elif hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state.cpu().numpy()
            else:
                return outputs

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            'config': self.model.config.to_dict() if hasattr(self.model, 'config') else {},
        }

        if self.tokenizer:
            params['tokenizer_config'] = {
                'vocab_size': self.tokenizer.vocab_size,
                'model_max_length': self.tokenizer.model_max_length,
            }

        return params


class HuggingFaceDatasetWrapper(DatasetWrapper):
    """Wrapper for Hugging Face datasets"""

    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from Hugging Face dataset"""
        from datasets import Dataset, DatasetDict

        if isinstance(self.data, Dataset):
            return DatasetMetadata(
                format='huggingface.Dataset',
                shape=(len(self.data), len(self.data.features)),
                n_samples=len(self.data),
                feature_columns=list(self.data.features.keys()),
                dtypes={k: str(v) for k, v in self.data.features.items()},
            )
        elif isinstance(self.data, DatasetDict):
            # Handle dataset dict
            splits = list(self.data.keys())
            total_samples = sum(len(split) for split in self.data.values())

            return DatasetMetadata(
                format='huggingface.DatasetDict',
                shape=(total_samples,),
                n_samples=total_samples,
                stats={'splits': splits, 'split_sizes': {k: len(v) for k, v in self.data.items()}},
            )
        else:
            raise ValueError(f"Unsupported Hugging Face data type: {type(self.data)}")

    def to_numpy(self) -> Any:
        """Convert to numpy array"""
        import numpy as np

        # This is complex for HF datasets as they can contain multiple fields
        if hasattr(self.data, 'to_pandas'):
            return self.data.to_pandas().values
        else:
            raise NotImplementedError("Complex conversion for this dataset type")

    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        from datasets import Dataset

        if isinstance(self.data, Dataset):
            split_data = self.data.train_test_split(train_size=train_size)
            return split_data['train'], split_data['test']
        else:
            raise ValueError("Can only split Dataset objects")

    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample n items from dataset"""
        if hasattr(self.data, 'shuffle'):
            shuffled = self.data.shuffle(seed=random_state)
            return shuffled.select(range(min(n, len(self.data))))
        else:
            raise NotImplementedError("Sampling not supported for this dataset type")


class HuggingFaceIntegration(BaseIntegration):
    """Integration for Hugging Face transformers and datasets"""

    @property
    def is_available(self) -> bool:
        """Check if transformers is installed"""
        try:
            import transformers
            import datasets
            return True
        except ImportError:
            return False

    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define Hugging Face capabilities"""
        caps = [
            IntegrationCapability.TRAINING,
            IntegrationCapability.INFERENCE,
            IntegrationCapability.SERIALIZATION,
            IntegrationCapability.GPU_SUPPORT,
            IntegrationCapability.MODEL_REGISTRY,
        ]

        return caps

    def get_version(self) -> str:
        """Get transformers version"""
        try:
            import transformers
            return transformers.__version__
        except ImportError:
            return "not installed"

    def wrap_model(self, model: Any) -> ModelWrapper:
        """Wrap Hugging Face model"""
        # Try to get associated tokenizer
        tokenizer = None
        if hasattr(model, 'name_or_path'):
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
            except:
                pass

        return HuggingFaceModelWrapper(model, tokenizer)

    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap Hugging Face dataset"""
        return HuggingFaceDatasetWrapper(data)

    def get_serialization_handler(self) -> SerializationHandler:
        """Get Hugging Face serialization handler"""
        if self._serialization_handler is None:
            self._serialization_handler = HuggingFaceSerializer()
        return self._serialization_handler

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def load_model(self, model_name: str, **kwargs) -> Tuple[Any, Any]:
        """Load model and tokenizer from Hugging Face Hub"""
        from transformers import AutoModel, AutoTokenizer

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            **kwargs
        )

        return model, tokenizer

    def load_task_model(self, model_name: str, task: str, **kwargs) -> Tuple[Any, Any]:
        """Load task-specific model from Hugging Face Hub"""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoModelForQuestionAnswering,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        task_models = {
            'text-classification': AutoModelForSequenceClassification,
            'token-classification': AutoModelForTokenClassification,
            'question-answering': AutoModelForQuestionAnswering,
            'text-generation': AutoModelForCausalLM,
            'text2text-generation': AutoModelForSeq2SeqLM,
        }

        if task not in task_models:
            raise ValueError(f"Unknown task: {task}. Choose from {list(task_models.keys())}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

        # Load model
        model_class = task_models[task]
        model = model_class.from_pretrained(model_name, **kwargs)

        return model, tokenizer

    def create_pipeline(self, task: str, model: Any = None, tokenizer: Any = None, **kwargs) -> Any:
        """Create Hugging Face pipeline"""
        from transformers import pipeline

        return pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=0 if self._check_gpu_available() else -1,
            **kwargs
        )

    def optimize_for_inference(self, model: Any, **kwargs) -> Any:
        """Optimize model for inference"""
        import torch

        # Quantization
        if kwargs.get('quantize', False):
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Model quantized to int8")

        # ONNX export
        if kwargs.get('export_onnx', False):
            from transformers import onnx
            # This is complex and model-specific
            logger.warning("ONNX export not implemented yet")

        # TorchScript
        if kwargs.get('torchscript', False):
            try:
                model = torch.jit.script(model)
                logger.info("Model converted to TorchScript")
            except:
                logger.warning("TorchScript conversion failed")

        return model

    def load_dataset(self, dataset_name: str, **kwargs) -> Any:
        """Load dataset from Hugging Face Hub"""
        from datasets import load_dataset

        return load_dataset(dataset_name, **kwargs)

    def trainer(
            self,
            model: Any,
            train_dataset: Any,
            eval_dataset: Any = None,
            tokenizer: Any = None,
            **kwargs
    ) -> Any:
        """Create Hugging Face Trainer"""
        from transformers import Trainer, TrainingArguments

        # Default training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './results'),
            num_train_epochs=kwargs.get('num_epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', 16),
            per_device_eval_batch_size=kwargs.get('eval_batch_size', 64),
            warmup_steps=kwargs.get('warmup_steps', 500),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir=kwargs.get('logging_dir', './logs'),
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset is not None else False,
            fp16=self._check_gpu_available(),
        )

        # Update with user kwargs
        for key, value in kwargs.items():
            if hasattr(training_args, key):
                setattr(training_args, key, value)

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=kwargs.get('compute_metrics'),
        )

    def push_to_hub(self, model: Any, repo_id: str, **kwargs) -> None:
        """Push model to Hugging Face Hub"""
        if hasattr(model, 'push_to_hub'):
            model.push_to_hub(repo_id, **kwargs)
            logger.info(f"Model pushed to hub: {repo_id}")
        else:
            raise ValueError("Model doesn't support push_to_hub")