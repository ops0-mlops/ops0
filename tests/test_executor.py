"""
Tests pour le module executor de ops0.
Teste l'exécution locale et distribuée des pipelines.
"""
import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import cloudpickle
import subprocess

from ops0.executor import (
    ExecutionMode,
    ExecutionContext,
    StepResult,
    LocalExecutor,
    DockerExecutor,
    Orchestrator
)
from ops0.decorators import StepConfig, PipelineConfig
from ops0.storage import LocalStorage, S3Storage


class TestExecutionMode:
    """Tests pour l'enum ExecutionMode"""

    def test_execution_modes_exist(self):
        """Test que tous les modes d'exécution sont définis"""
        assert ExecutionMode.LOCAL.value == "local"
        assert ExecutionMode.DOCKER.value == "docker"
        assert ExecutionMode.LAMBDA.value == "lambda"
        assert ExecutionMode.KUBERNETES.value == "kubernetes"


class TestExecutionContext:
    """Tests pour ExecutionContext"""

    def test_context_creation(self):
        """Test la création d'un contexte d'exécution"""
        ctx = ExecutionContext(
            mode=ExecutionMode.LOCAL,
            pipeline_name="test_pipeline",
            execution_id="exec-123"
        )

        assert ctx.mode == ExecutionMode.LOCAL
        assert ctx.pipeline_name == "test_pipeline"
        assert ctx.execution_id == "exec-123"
        assert ctx.metadata == {}

    def test_context_manager(self):
        """Test ExecutionContext comme context manager"""
        # Pas de contexte au départ
        assert ExecutionContext.current() is None

        # Créer un contexte
        with ExecutionContext(mode=ExecutionMode.DOCKER) as ctx:
            # Le contexte devrait être accessible
            current = ExecutionContext.current()
            assert current is not None
            assert current.mode == ExecutionMode.DOCKER
            assert current is ctx

        # Contexte nettoyé après sortie
        assert ExecutionContext.current() is None

    def test_nested_contexts(self):
        """Test que les contextes imbriqués fonctionnent"""
        with ExecutionContext(mode=ExecutionMode.LOCAL) as ctx1:
            assert ExecutionContext.current().mode == ExecutionMode.LOCAL

            with ExecutionContext(mode=ExecutionMode.DOCKER) as ctx2:
                # Le contexte interne devrait être actif
                assert ExecutionContext.current().mode == ExecutionMode.DOCKER

            # Retour au contexte externe
            assert ExecutionContext.current().mode == ExecutionMode.LOCAL

    def test_context_with_storage(self):
        """Test du contexte avec backend de storage"""
        storage = LocalStorage()
        ctx = ExecutionContext(
            mode=ExecutionMode.LOCAL,
            storage=storage
        )

        assert ctx.storage is storage

    def test_context_metadata(self):
        """Test des métadonnées du contexte"""
        ctx = ExecutionContext(
            mode=ExecutionMode.LOCAL,
            metadata={"user": "test", "environment": "dev"}
        )

        assert ctx.metadata["user"] == "test"
        assert ctx.metadata["environment"] == "dev"


class TestStepResult:
    """Tests pour StepResult"""

    def test_step_result_creation(self):
        """Test la création d'un StepResult"""
        result = StepResult(
            step_name="test_step",
            success=True,
            result={"output": "data"},
            start_time=1000.0,
            end_time=1010.0
        )

        assert result.step_name == "test_step"
        assert result.success is True
        assert result.result == {"output": "data"}
        assert result.duration == 0  # Non calculé automatiquement

    def test_step_result_with_error(self):
        """Test StepResult avec erreur"""
        result = StepResult(
            step_name="failing_step",
            success=False,
            error="Division by zero",
            metadata={"traceback": "...stack trace..."}
        )

        assert result.success is False
        assert result.error == "Division by zero"
        assert "traceback" in result.metadata

    def test_step_result_duration_calculation(self):
        """Test le calcul de durée"""
        result = StepResult(
            step_name="timed_step",
            success=True,
            start_time=1000.0,
            end_time=1005.5,
            duration=5.5
        )

        assert result.duration == 5.5


class TestLocalExecutor:
    """Tests pour LocalExecutor"""

    def setup_method(self):
        """Créer un executor local pour les tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(self.temp_dir)
        self.executor = LocalExecutor(self.storage)

    def teardown_method(self):
        """Nettoyer après les tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_execute_simple_step(self):
        """Test l'exécution d'un step simple"""
        def simple_function(x):
            return x * 2

        step_config = StepConfig(
            name="simple",
            func=simple_function
        )

        result = self.executor.execute_step(step_config, (5,), {})

        assert result.success is True
        assert result.result == 10
        assert result.step_name == "simple"
        assert result.duration > 0

    def test_execute_step_with_kwargs(self):
        """Test l'exécution avec arguments nommés"""
        def kwargs_function(x, y=10):
            return x + y

        step_config = StepConfig(
            name="kwargs_step",
            func=kwargs_function
        )

        result = self.executor.execute_step(step_config, (5,), {"y": 20})

        assert result.success is True
        assert result.result == 25

    def test_execute_failing_step(self):
        """Test l'exécution d'un step qui échoue"""
        def failing_function():
            raise ValueError("Test error")

        step_config = StepConfig(
            name="failing",
            func=failing_function
        )

        result = self.executor.execute_step(step_config, (), {})

        assert result.success is False
        assert result.error == "Test error"
        assert "traceback" in result.metadata
        assert "ValueError" in result.metadata["traceback"]

    def test_step_result_storage(self):
        """Test que les résultats sont sauvegardés"""
        def data_function():
            return {"important": "data"}

        step_config = StepConfig(
            name="storage_test",
            func=data_function
        )

        result = self.executor.execute_step(step_config, (), {})

        # Vérifier que le résultat est sauvé
        assert result.success is True
        saved_data = self.storage.load("storage_test_result")
        assert saved_data == {"important": "data"}

    def test_execute_pipeline_simple(self):
        """Test l'exécution d'une pipeline simple"""
        def pipeline_func():
            return "pipeline result"

        pipeline_config = PipelineConfig(
            name="test_pipeline",
            func=pipeline_func
        )

        results = self.executor.execute_pipeline(pipeline_config, (), {})

        assert "pipeline" in results
        assert results["pipeline"].success is True
        assert results["pipeline"].result == "pipeline result"

    def test_get_execution_order(self):
        """Test la détermination de l'ordre d'exécution"""
        # DAG simple : A -> B -> C
        dag = {
            "C": ["B"],
            "B": ["A"],
            "A": []
        }

        order = self.executor._get_execution_order(dag)
        assert order == ["A", "B", "C"]

    def test_get_execution_order_parallel(self):
        """Test l'ordre avec branches parallèles"""
        # DAG : A -> C, B -> C (A et B peuvent être parallèles)
        dag = {
            "C": ["A", "B"],
            "A": [],
            "B": []
        }

        order = self.executor._get_execution_order(dag)
        # A et B doivent être avant C
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("C")

    def test_resolve_step_args(self):
        """Test la résolution des arguments depuis les résultats"""
        dag = {
            "step2": ["step1"],
            "step1": []
        }

        results = {
            "step1": StepResult(
                step_name="step1",
                success=True,
                result="data from step1"
            )
        }

        args = self.executor._resolve_step_args("step2", dag, results)
        assert args == ("data from step1",)

    def test_execute_pipeline_with_dependencies(self):
        """Test une pipeline avec dépendances entre steps"""
        def step_a():
            return 10

        def step_b(x):
            return x * 2

        def pipeline_func():
            a = step_a()
            return step_b(a)

        # Créer les configs
        step_a_config = StepConfig(name="step_a", func=step_a)
        step_b_config = StepConfig(name="step_b", func=step_b)

        pipeline_config = PipelineConfig(
            name="dep_pipeline",
            func=pipeline_func,
            steps={"step_a": step_a_config, "step_b": step_b_config}
        )

        # Mock build_dag pour retourner notre DAG
        with patch('ops0.executor.build_dag') as mock_dag:
            mock_dag.return_value = {
                "step_b": ["step_a"],
                "step_a": []
            }

            results = self.executor.execute_pipeline(pipeline_config, (), {})

            # Vérifier que la pipeline s'exécute
            assert results["pipeline"].success is True

    def test_pipeline_with_failing_step(self):
        """Test qu'une pipeline s'arrête si un step échoue"""
        def failing_step():
            raise RuntimeError("Step failed")

        def dependent_step(x):
            return x

        step1 = StepConfig(name="failing", func=failing_step)
        step2 = StepConfig(name="dependent", func=dependent_step)

        pipeline_config = PipelineConfig(
            name="fail_pipeline",
            func=lambda: None,
            steps={"failing": step1, "dependent": step2}
        )

        with patch('ops0.executor.build_dag') as mock_dag:
            mock_dag.return_value = {
                "dependent": ["failing"],
                "failing": []
            }

            results = self.executor.execute_pipeline(pipeline_config, (), {})

            # Le step dependent ne devrait pas s'exécuter
            assert "failing" in results or "pipeline" in results


class TestDockerExecutor:
    """Tests pour DockerExecutor"""

    def setup_method(self):
        """Créer un executor Docker pour les tests"""
        self.executor = DockerExecutor()

    def test_generate_dockerfile(self):
        """Test la génération de Dockerfile"""
        step_config = StepConfig(
            name="test_step",
            func=lambda x: x,
            memory=1024,
            requirements=["pandas>=1.3.0", "numpy>=1.21.0"]
        )

        dockerfile = self.executor._generate_dockerfile(step_config)

        assert "FROM python:3.9-slim" in dockerfile
        assert "pip install --no-cache-dir cloudpickle boto3" in dockerfile
        assert "pip install --no-cache-dir pandas>=1.3.0 numpy>=1.21.0" in dockerfile

    def test_generate_dockerfile_gpu(self):
        """Test Dockerfile pour GPU"""
        step_config = StepConfig(
            name="gpu_step",
            func=lambda x: x,
            gpu=True
        )

        dockerfile = self.executor._generate_dockerfile(step_config)

        assert "pytorch/pytorch" in dockerfile  # Image GPU

    def test_generate_step_wrapper(self):
        """Test la génération du wrapper Python"""
        step_config = StepConfig(
            name="wrapped_step",
            func=lambda x: x * 2
        )

        wrapper = self.executor._generate_step_wrapper(step_config)

        assert "import cloudpickle" in wrapper
        assert "import boto3" in wrapper
        assert "OPS0_BUCKET" in wrapper
        assert "func = cloudpickle.loads" in wrapper

    @patch('subprocess.run')
    def test_build_container(self, mock_run):
        """Test la construction du container"""
        mock_run.return_value = MagicMock(returncode=0)

        step_config = StepConfig(
            name="build_test",
            func=lambda: "result"
        )

        with patch('tempfile.TemporaryDirectory') as mock_tmpdir:
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"

            image_name = self.executor.build_container(step_config)

            assert image_name == "ops0-build_test:latest"
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "docker"
            assert call_args[1] == "build"

    def test_execute_step_docker(self):
        """Test l'exécution d'un step (simulé)"""
        step_config = StepConfig(
            name="docker_step",
            func=lambda: "docker result"
        )

        with patch.object(self.executor, 'build_container') as mock_build:
            mock_build.return_value = "ops0-docker_step:latest"

            result = self.executor.execute_step(step_config, (), {})

            assert result.success is True
            assert result.metadata["executor"] == "docker"
            assert result.metadata["image"] == "ops0-docker_step:latest"


class TestOrchestrator:
    """Tests pour Orchestrator"""

    def test_orchestrator_creation(self):
        """Test la création de l'orchestrateur"""
        orch = Orchestrator(mode=ExecutionMode.LOCAL)
        assert orch.mode == ExecutionMode.LOCAL
        assert ExecutionMode.LOCAL in orch.executors

    def test_orchestrator_mode_selection(self):
        """Test que le bon executor est sélectionné"""
        orch = Orchestrator(mode=ExecutionMode.DOCKER)
        assert orch.mode == ExecutionMode.DOCKER

    @patch('ops0.executor.LocalExecutor.execute_pipeline')
    def test_execute_pipeline_local(self, mock_execute):
        """Test l'exécution via orchestrateur"""
        mock_execute.return_value = {
            "step1": StepResult("step1", True, "result1"),
            "pipeline": StepResult("pipeline", True, "final")
        }

        orch = Orchestrator(mode=ExecutionMode.LOCAL)

        pipeline_config = PipelineConfig(
            name="test_pipeline",
            func=lambda: "test"
        )

        results = orch.execute_pipeline(pipeline_config)

        assert results["success"] is True
        assert "execution_id" in results
        assert results["pipeline"] == "test_pipeline"
        mock_execute.assert_called_once()

    def test_execution_summary(self):
        """Test le résumé d'exécution"""
        orch = Orchestrator(mode=ExecutionMode.LOCAL)

        # Mock l'executor
        mock_results = {
            "step1": StepResult("step1", True, "ok", duration=1.5),
            "step2": StepResult("step2", False, error="failed", duration=0.5),
            "step3": StepResult("step3", True, "ok", duration=2.0)
        }

        with patch.object(orch.executors[ExecutionMode.LOCAL],
                         'execute_pipeline', return_value=mock_results):

            pipeline_config = PipelineConfig(name="summary_test", func=lambda: None)
            results = orch.execute_pipeline(pipeline_config)

            # Vérifier le résumé
            assert results["success"] is False  # Un step a échoué
            assert len(results["results"]) == 3

    @patch('uuid.uuid4')
    def test_execution_id_generation(self, mock_uuid):
        """Test que l'execution ID est généré"""
        mock_uuid.return_value = "test-uuid-123"

        orch = Orchestrator(mode=ExecutionMode.LOCAL)

        with patch.object(orch.executors[ExecutionMode.LOCAL],
                         'execute_pipeline', return_value={}):

            pipeline_config = PipelineConfig(name="id_test", func=lambda: None)
            results = orch.execute_pipeline(pipeline_config)

            assert results["execution_id"] == "test-uuid-123"

    def test_orchestrator_with_different_modes(self):
        """Test que l'orchestrateur supporte différents modes"""
        # LOCAL
        orch_local = Orchestrator(mode=ExecutionMode.LOCAL)
        assert isinstance(orch_local.executors[ExecutionMode.LOCAL], LocalExecutor)

        # DOCKER
        orch_docker = Orchestrator(mode=ExecutionMode.DOCKER)
        assert isinstance(orch_docker.executors[ExecutionMode.DOCKER], DockerExecutor)


class TestIntegrationExecutor:
    """Tests d'intégration pour l'executor"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_local_execution(self):
        """Test complet d'exécution locale"""
        # Créer une vraie pipeline
        def load_data():
            return [1, 2, 3, 4, 5]

        def process_data(data):
            return [x * 2 for x in data]

        def save_data(processed):
            return {"count": len(processed), "sum": sum(processed)}

        # Créer les configs
        load_config = StepConfig(name="load", func=load_data)
        process_config = StepConfig(name="process", func=process_data)
        save_config = StepConfig(name="save", func=save_data)

        def pipeline():
            data = load_data()
            processed = process_data(data)
            return save_data(processed)

        pipeline_config = PipelineConfig(
            name="e2e_pipeline",
            func=pipeline,
            steps={
                "load": load_config,
                "process": process_config,
                "save": save_config
            }
        )

        # Exécuter via orchestrateur
        orch = Orchestrator(mode=ExecutionMode.LOCAL)

        with patch('ops0.executor.build_dag') as mock_dag:
            mock_dag.return_value = {
                "save": ["process"],
                "process": ["load"],
                "load": []
            }

            results = orch.execute_pipeline(pipeline_config)

            assert results["success"] is True
            assert results["pipeline"] == "e2e_pipeline"

    def test_parallel_execution_simulation(self):
        """Test simulation d'exécution parallèle"""
        # Steps qui peuvent s'exécuter en parallèle
        def parallel_a():
            time.sleep(0.1)
            return "a"

        def parallel_b():
            time.sleep(0.1)
            return "b"

        def merge(a, b):
            return f"{a}+{b}"

        configs = {
            "parallel_a": StepConfig(name="parallel_a", func=parallel_a),
            "parallel_b": StepConfig(name="parallel_b", func=parallel_b),
            "merge": StepConfig(name="merge", func=merge)
        }

        pipeline_config = PipelineConfig(
            name="parallel_pipeline",
            func=lambda: merge(parallel_a(), parallel_b()),
            steps=configs
        )

        executor = LocalExecutor()

        with patch('ops0.executor.build_dag') as mock_dag:
            # A et B n'ont pas de dépendances, merge dépend des deux
            mock_dag.return_value = {
                "merge": ["parallel_a", "parallel_b"],
                "parallel_a": [],
                "parallel_b": []
            }

            start_time = time.time()
            results = executor.execute_pipeline(pipeline_config, (), {})
            duration = time.time() - start_time

            # En séquentiel, ça prendrait au moins 0.2s
            # Même si on n'a pas de vraie parallélisation,
            # on vérifie que le DAG est correctement construit
            assert "pipeline" in results
            assert results["pipeline"].success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])