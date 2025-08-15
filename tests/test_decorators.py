"""
Tests pour les décorateurs ops0 (@step et @pipeline).
Couvre les cas normaux, edge cases et erreurs.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import inspect
from typing import Any, Dict, List

from ops0.decorators import (
    step,
    pipeline,
    monitor,
    StepConfig,
    PipelineConfig,
    StepDecorator,
    PipelineDecorator
)
from ops0.executor import ExecutionContext, ExecutionMode


class TestStepDecorator:
    """Tests pour le décorateur @ops0.step"""

    def setup_method(self):
        """Reset des registres avant chaque test"""
        StepDecorator._steps.clear()
        PipelineDecorator._pipelines.clear()

    def test_simple_step_decoration(self):
        """Test qu'une fonction simple peut être décorée avec @step"""
        @step
        def simple_function(x: int) -> int:
            return x * 2

        # Vérifier que la fonction est toujours callable
        assert simple_function(5) == 10

        # Vérifier que la fonction est enregistrée
        assert "simple_function" in StepDecorator._steps

        # Vérifier les métadonnées
        config = StepDecorator.get("simple_function")
        assert config.name == "simple_function"
        assert config.memory == 512  # Default
        assert config.gpu is False
        assert config.retries == 3

    def test_step_with_configuration(self):
        """Test du décorateur @step avec configuration personnalisée"""
        @step(
            name="custom_step",
            memory=2048,
            timeout=600,
            gpu=True,
            retries=5,
            requirements=["torch>=1.10.0", "numpy>=1.20.0"]
        )
        def gpu_intensive_task(data):
            return data

        config = StepDecorator.get("custom_step")
        assert config.name == "custom_step"
        assert config.memory == 2048
        assert config.timeout == 600
        assert config.gpu is True
        assert config.retries == 5
        assert "torch>=1.10.0" in config.requirements

    def test_step_decorator_with_parentheses(self):
        """Test que @step() fonctionne comme @step"""
        @step()
        def with_parens():
            return 42

        assert with_parens() == 42
        assert "with_parens" in StepDecorator._steps

    def test_step_function_analysis(self):
        """Test que l'analyse automatique de la fonction fonctionne"""
        @step
        def ml_function(df):
            import pandas as pd
            import torch

            model = torch.nn.Linear(10, 1)
            return model(df)

        config = StepDecorator.get("ml_function")

        # L'analyseur devrait détecter l'usage de ML
        assert config.analysis is not None
        assert config.memory >= 2048  # Devrait être augmenté
        assert config.gpu is True  # Devrait détecter torch

    def test_step_decorator_preserves_function_metadata(self):
        """Test que le décorateur préserve les métadonnées de la fonction"""
        @step
        def documented_function(x: int, y: str = "default") -> Dict[str, Any]:
            """Cette fonction a une docstring."""
            return {"x": x, "y": y}

        # Vérifier que les métadonnées sont préservées
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "Cette fonction a une docstring."
        assert documented_function._ops0_type == "step"

        # Vérifier la signature
        sig = inspect.signature(documented_function)
        assert "x" in sig.parameters
        assert "y" in sig.parameters
        assert sig.parameters["y"].default == "default"

    def test_step_execution_context(self):
        """Test que le step utilise le contexte d'exécution"""
        @step
        def context_aware_step(data):
            ctx = ExecutionContext.current()
            if ctx and ctx.mode == ExecutionMode.LOCAL:
                return "local"
            return "production"

        # Sans contexte
        assert context_aware_step("test") == "production"

        # Avec contexte local
        with ExecutionContext(mode=ExecutionMode.LOCAL):
            assert context_aware_step("test") == "local"

    def test_step_decorator_error_handling(self):
        """Test que le décorateur gère les erreurs correctement"""
        # Tenter de décorer un non-callable
        with pytest.raises(TypeError):
            step(42)

        # Fonction qui lève une exception
        @step
        def failing_step():
            raise ValueError("Test error")

        # La fonction devrait toujours lever l'exception
        with pytest.raises(ValueError, match="Test error"):
            failing_step()

    def test_multiple_steps_registration(self):
        """Test que plusieurs steps peuvent être enregistrés"""
        @step
        def step1():
            return 1

        @step
        def step2():
            return 2

        @step(name="custom_name")
        def step3():
            return 3

        assert len(StepDecorator._steps) == 3
        assert "step1" in StepDecorator._steps
        assert "step2" in StepDecorator._steps
        assert "custom_name" in StepDecorator._steps

    def test_step_config_post_init(self):
        """Test que StepConfig applique les defaults intelligents"""
        # Créer une analyse mockée
        mock_analysis = Mock()
        mock_analysis.uses_ml_framework = True
        mock_analysis.uses_gpu = True
        mock_analysis.estimated_requirements = ["pandas", "torch"]

        # Créer une config avec mémoire faible
        config = StepConfig(
            name="test",
            func=lambda: None,
            memory=512,
            analysis=mock_analysis
        )

        # La mémoire devrait être augmentée
        assert config.memory == 2048
        assert config.gpu is True
        assert "pandas" in config.requirements
        assert "torch" in config.requirements


class TestPipelineDecorator:
    """Tests pour le décorateur @ops0.pipeline"""

    def setup_method(self):
        """Reset et création de steps de test"""
        StepDecorator._steps.clear()
        PipelineDecorator._pipelines.clear()

        # Créer des steps de test
        @step
        def load_data(path: str):
            return f"data from {path}"

        @step
        def process_data(data: str):
            return f"processed {data}"

        @step
        def save_results(results: str):
            return f"saved {results}"

    def test_simple_pipeline_decoration(self):
        """Test qu'une pipeline simple peut être créée"""
        @pipeline
        def simple_pipeline(input_path: str):
            data = load_data(input_path)
            processed = process_data(data)
            return save_results(processed)

        # Vérifier que la fonction est callable
        result = simple_pipeline("test.csv")
        assert result == "saved processed data from test.csv"

        # Vérifier l'enregistrement
        assert "simple_pipeline" in PipelineDecorator._pipelines
        config = PipelineDecorator._pipelines["simple_pipeline"]
        assert config.name == "simple_pipeline"
        assert len(config.steps) >= 1  # Devrait détecter les steps

    def test_pipeline_with_configuration(self):
        """Test du décorateur @pipeline avec configuration"""
        @pipeline(
            name="scheduled_pipeline",
            schedule="0 */2 * * *",
            description="Pipeline qui s'exécute toutes les 2 heures"
        )
        def my_pipeline():
            return "done"

        config = PipelineDecorator._pipelines["scheduled_pipeline"]
        assert config.name == "scheduled_pipeline"
        assert config.schedule == "0 */2 * * *"
        assert config.description == "Pipeline qui s'exécute toutes les 2 heures"

    def test_pipeline_step_detection(self):
        """Test que la pipeline détecte les steps utilisés"""
        @step
        def step_a():
            return "a"

        @step
        def step_b(x):
            return f"b({x})"

        @pipeline
        def detection_pipeline():
            a = step_a()
            return step_b(a)

        config = PipelineDecorator._pipelines["detection_pipeline"]

        # Devrait détecter step_a et step_b
        assert "step_a" in config.steps or "step_b" in config.steps

    def test_pipeline_with_docstring(self):
        """Test que la docstring est préservée et utilisée"""
        @pipeline
        def documented_pipeline():
            """
            Cette pipeline fait des choses importantes.

            Elle est bien documentée.
            """
            return "result"

        config = PipelineDecorator._pipelines["documented_pipeline"]
        assert config.description == documented_pipeline.__doc__

    def test_pipeline_metadata_preservation(self):
        """Test que les métadonnées de la fonction sont préservées"""
        @pipeline(name="test_pipe")
        def original_function(x: int, y: str = "default") -> str:
            """Original docstring"""
            return f"{x}-{y}"

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring"
        assert original_function._ops0_type == "pipeline"
        assert hasattr(original_function, "_ops0_pipeline")

    def test_pipeline_execution_context_setting(self):
        """Test que la pipeline définit le contexte d'exécution"""
        context_captured = []

        @pipeline
        def context_pipeline():
            ctx = ExecutionContext.current()
            if ctx:
                context_captured.append(ctx.pipeline_name)
            return "done"

        # Exécuter dans un contexte
        with ExecutionContext(mode=ExecutionMode.LOCAL) as ctx:
            result = context_pipeline()
            assert result == "done"
            # Le nom de la pipeline devrait être défini
            assert ctx.pipeline_name == "context_pipeline"

    def test_nested_pipeline_calls(self):
        """Test qu'une pipeline peut appeler d'autres pipelines"""
        @pipeline
        def sub_pipeline(x):
            return x * 2

        @pipeline
        def main_pipeline(x):
            intermediate = sub_pipeline(x)
            return intermediate + 1

        result = main_pipeline(5)
        assert result == 11

        # Les deux pipelines devraient être enregistrées
        assert "sub_pipeline" in PipelineDecorator._pipelines
        assert "main_pipeline" in PipelineDecorator._pipelines

    def test_pipeline_error_handling(self):
        """Test que les erreurs dans la pipeline sont propagées"""
        @pipeline
        def error_pipeline():
            raise RuntimeError("Pipeline error")

        with pytest.raises(RuntimeError, match="Pipeline error"):
            error_pipeline()

    @patch('ops0.parser.analyze_function')
    def test_pipeline_dag_building(self, mock_analyze):
        """Test que la pipeline construit un DAG"""
        # Mock de l'analyse
        mock_analysis = Mock()
        mock_analysis.called_functions = {"step_a", "step_b"}
        mock_analyze.return_value = mock_analysis

        @pipeline
        def dag_pipeline():
            pass

        # Vérifier que l'analyse a été appelée
        mock_analyze.assert_called_once()


class TestMonitorDecorator:
    """Tests pour le décorateur @monitor"""

    def test_monitor_decorator_adds_metadata(self):
        """Test que @monitor ajoute des métadonnées de monitoring"""
        @step
        @monitor(alert_on_latency="500ms", alert_on_error=True)
        def monitored_step():
            return "result"

        config = StepDecorator.get("monitored_step")
        assert hasattr(monitored_step, '_ops0_step')

        # Vérifier les métadonnées de monitoring
        step_config = monitored_step._ops0_step
        assert hasattr(step_config, 'monitoring')
        assert step_config.monitoring['alert_on_latency'] == "500ms"
        assert step_config.monitoring['alert_on_error'] is True

    def test_monitor_decorator_order(self):
        """Test que @monitor peut être appliqué avant ou après @step"""
        # Monitor après step
        @step
        @monitor(alert_on_error=False)
        def step_then_monitor():
            return 1

        # Monitor avant step
        @monitor(alert_on_latency="1s")
        @step
        def monitor_then_step():
            return 2

        # Les deux devraient fonctionner
        assert step_then_monitor() == 1
        assert monitor_then_step() == 2

    def test_monitor_without_step(self):
        """Test que @monitor sans @step ne fait rien"""
        @monitor(alert_on_error=True)
        def regular_function():
            return "not a step"

        # Devrait fonctionner normalement
        assert regular_function() == "not a step"

        # Mais pas de métadonnées ops0
        assert not hasattr(regular_function, '_ops0_step')


class TestIntegration:
    """Tests d'intégration des décorateurs"""

    def setup_method(self):
        StepDecorator._steps.clear()
        PipelineDecorator._pipelines.clear()

    def test_complete_ml_pipeline(self):
        """Test d'une pipeline ML complète avec tous les décorateurs"""
        # Définir les steps
        @step(memory=1024)
        def load_data(path: str) -> Dict[str, Any]:
            return {"data": f"loaded from {path}", "size": 1000}

        @step(memory=2048, gpu=True)
        @monitor(alert_on_latency="10s")
        def train_model(data: Dict[str, Any]) -> Dict[str, float]:
            return {"accuracy": 0.95, "loss": 0.05}

        @step
        def evaluate_model(model_metrics: Dict[str, float]) -> str:
            if model_metrics["accuracy"] > 0.9:
                return "model approved"
            return "model rejected"

        # Définir la pipeline
        @pipeline(
            name="ml_training",
            schedule="0 0 * * *",
            description="Daily model training"
        )
        def training_pipeline(data_path: str) -> str:
            data = load_data(data_path)
            metrics = train_model(data)
            return evaluate_model(metrics)

        # Exécuter la pipeline
        result = training_pipeline("s3://bucket/data.csv")
        assert result == "model approved"

        # Vérifier les enregistrements
        assert len(StepDecorator._steps) == 3
        assert "ml_training" in PipelineDecorator._pipelines

        # Vérifier les configurations
        train_config = StepDecorator.get("train_model")
        assert train_config.memory == 2048
        assert train_config.gpu is True
        assert hasattr(train_config, 'monitoring')

    def test_registry_isolation(self):
        """Test que les registres sont bien isolés entre steps et pipelines"""
        @step
        def isolated_step():
            return "step"

        @pipeline
        def isolated_pipeline():
            return "pipeline"

        # Vérifier l'isolation
        assert "isolated_step" in StepDecorator._steps
        assert "isolated_step" not in PipelineDecorator._pipelines
        assert "isolated_pipeline" in PipelineDecorator._pipelines
        assert "isolated_pipeline" not in StepDecorator._steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])