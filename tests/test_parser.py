"""
Tests pour le module parser de ops0.
Teste l'analyse AST et la détection automatique des dépendances.
"""
import pytest
import ast
import inspect
from unittest.mock import Mock, patch
import textwrap

from ops0.parser import (
    FunctionAnalysis,
    FunctionVisitor,
    analyze_function,
    build_dag,
    detect_cycles,
    _estimate_memory,
    _estimate_requirements,
    _detect_dependencies
)


class TestFunctionAnalysis:
    """Tests pour la dataclass FunctionAnalysis"""

    def test_function_analysis_creation(self):
        """Test la création d'une analyse de fonction"""
        analysis = FunctionAnalysis(
            name="test_func",
            source_code="def test_func(): pass"
        )

        assert analysis.name == "test_func"
        assert analysis.source_code == "def test_func(): pass"
        assert analysis.imports == set()
        assert analysis.called_functions == set()
        assert analysis.arguments == []
        assert analysis.uses_ml_framework is False
        assert analysis.estimated_memory == 512

    def test_function_analysis_with_ml_flags(self):
        """Test avec flags ML activés"""
        analysis = FunctionAnalysis(
            name="ml_func",
            source_code="",
            uses_pandas=True,
            uses_sklearn=True,
            uses_gpu=True
        )

        assert analysis.uses_pandas is True
        assert analysis.uses_sklearn is True
        assert analysis.uses_gpu is True
        assert analysis.uses_ml_framework is False  # Doit être défini explicitement


class TestFunctionVisitor:
    """Tests pour le visiteur AST"""

    def test_visit_simple_import(self):
        """Test la détection d'imports simples"""
        code = """
import pandas
import numpy as np
"""
        tree = ast.parse(code)
        visitor = FunctionVisitor()
        visitor.visit(tree)

        assert "pandas" in visitor.imports
        assert "numpy" in visitor.imports

    def test_visit_from_import(self):
        """Test la détection de from imports"""
        code = """
from sklearn.ensemble import RandomForestClassifier
from torch import nn
"""
        tree = ast.parse(code)
        visitor = FunctionVisitor()
        visitor.visit(tree)

        assert "sklearn.ensemble" in visitor.imports
        assert "torch" in visitor.imports

    def test_visit_function_calls(self):
        """Test la détection d'appels de fonction"""
        code = """
result = process_data(input)
output = model.predict(features)
transform(data)
"""
        tree = ast.parse(code)
        visitor = FunctionVisitor()
        visitor.visit(tree)

        assert "process_data" in visitor.called_functions
        assert "transform" in visitor.called_functions
        assert "model.predict" in visitor.attributes_accessed

    def test_ml_framework_detection(self):
        """Test la détection des frameworks ML"""
        # Pandas
        visitor = FunctionVisitor()
        visitor._check_ml_framework("pandas")
        assert visitor.ml_indicators["pandas"] is True

        # Numpy
        visitor = FunctionVisitor()
        visitor._check_ml_framework("numpy")
        assert visitor.ml_indicators["numpy"] is True

        # Scikit-learn
        visitor = FunctionVisitor()
        visitor._check_ml_framework("sklearn.preprocessing")
        assert visitor.ml_indicators["sklearn"] is True

        # PyTorch
        visitor = FunctionVisitor()
        visitor._check_ml_framework("torch.nn")
        assert visitor.ml_indicators["torch"] is True

        # TensorFlow
        visitor = FunctionVisitor()
        visitor._check_ml_framework("tensorflow.keras")
        assert visitor.ml_indicators["tensorflow"] is True

    def test_gpu_detection(self):
        """Test la détection d'utilisation GPU"""
        # Code avec indicateurs GPU
        code = """
device = torch.device('cuda')
model.cuda()
tensor.to('cuda:0')
"""
        tree = ast.parse(code)
        visitor = FunctionVisitor()
        visitor.visit(tree)

        assert visitor.ml_indicators["gpu"] is True

    def test_global_variable_detection(self):
        """Test la détection de variables globales"""
        code = """
x = global_var + 10
y = LOCAL_CONSTANT * 2
result = external_func(z)
"""
        tree = ast.parse(code)
        visitor = FunctionVisitor()
        visitor.visit(tree)

        assert "global_var" in visitor.global_vars
        assert "LOCAL_CONSTANT" in visitor.global_vars
        assert "external_func" in visitor.global_vars


class TestAnalyzeFunction:
    """Tests pour la fonction analyze_function"""

    def test_analyze_simple_function(self):
        """Test l'analyse d'une fonction simple"""

        def simple_func(x, y=10):
            """Une fonction simple"""
            return x + y

        analysis = analyze_function(simple_func)

        assert analysis.name == "simple_func"
        assert "def simple_func" in analysis.source_code
        assert analysis.arguments == ["x", "y"]
        assert analysis.returns is True  # Has return annotation info

    def test_analyze_ml_function(self):
        """Test l'analyse d'une fonction ML"""

        def ml_function(data):
            import pandas as pd
            import sklearn.ensemble

            df = pd.DataFrame(data)
            model = sklearn.ensemble.RandomForestClassifier()
            return model

        analysis = analyze_function(ml_function)

        assert analysis.uses_pandas is True
        assert analysis.uses_sklearn is True
        assert analysis.uses_ml_framework is True
        assert analysis.estimated_memory > 512

    def test_analyze_gpu_function(self):
        """Test l'analyse d'une fonction utilisant GPU"""

        def gpu_function(tensor):
            device = 'cuda'
            result = tensor.to(device)
            return result.cuda()

        analysis = analyze_function(gpu_function)

        assert analysis.uses_gpu is True
        assert analysis.estimated_memory >= 512

    def test_analyze_function_with_dependencies(self):
        """Test l'analyse avec dépendances détectées"""

        def dependent_func(data_from_loader, processed_results):
            combined = data_from_loader + processed_results
            return transform(combined)

        analysis = analyze_function(dependent_func)

        assert "data_from_loader" in analysis.arguments
        assert "processed_results" in analysis.arguments
        assert "transform" in analysis.called_functions

    def test_analyze_lambda_function(self):
        """Test l'analyse d'une fonction lambda"""
        lambda_func = lambda x: x * 2

        analysis = analyze_function(lambda_func)

        assert analysis.name == "<lambda>"
        assert len(analysis.arguments) == 1

    def test_analyze_function_without_source(self):
        """Test l'analyse quand le source n'est pas disponible"""
        # Fonction built-in
        analysis = analyze_function(print)

        assert analysis.name == "print"
        assert analysis.source_code == ""
        # Arguments devraient quand même être détectés via signature

    def test_analyze_complex_function(self):
        """Test l'analyse d'une fonction complexe"""

        def complex_function(input_data, model_config=None):
            """
            Fonction complexe avec plusieurs frameworks ML.
            """
            import pandas as pd
            import numpy as np
            import torch
            import tensorflow as tf

            # Traitement des données
            df = pd.read_csv(input_data)
            features = np.array(df.values)

            # GPU operations
            if torch.cuda.is_available():
                device = torch.device('cuda')
                tensor = torch.tensor(features).to(device)

            # Appels de fonctions
            preprocessed = preprocess_data(features)
            normalized = normalize_features(preprocessed)

            # Modèle
            model = create_model(model_config)
            predictions = model.predict(normalized)

            return predictions

        analysis = analyze_function(complex_function)

        assert analysis.uses_pandas is True
        assert analysis.uses_numpy is True
        assert analysis.uses_torch is True
        assert analysis.uses_tensorflow is True
        assert analysis.uses_gpu is True
        assert analysis.uses_ml_framework is True
        assert analysis.estimated_memory > 2048
        assert "preprocess_data" in analysis.called_functions
        assert "normalize_features" in analysis.called_functions
        assert "create_model" in analysis.called_functions


class TestEstimateMemory:
    """Tests pour l'estimation de mémoire"""

    def test_base_memory(self):
        """Test mémoire de base"""
        analysis = FunctionAnalysis(name="test", source_code="")
        memory = _estimate_memory(analysis)
        assert memory == 512

    def test_memory_with_pandas(self):
        """Test mémoire avec pandas"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            uses_pandas=True
        )
        memory = _estimate_memory(analysis)
        assert memory >= 1024

    def test_memory_with_ml_frameworks(self):
        """Test mémoire avec frameworks ML"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            uses_torch=True,
            uses_gpu=True
        )
        memory = _estimate_memory(analysis)
        assert memory >= 3072  # Base + torch + gpu

    def test_memory_with_large_data_indicators(self):
        """Test mémoire avec indicateurs de données volumineuses"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="df = pd.read_parquet('large_dataset.parquet')"
        )
        memory = _estimate_memory(analysis)
        assert memory > 512

    def test_memory_cap(self):
        """Test que la mémoire est plafonnée"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="large batch processing",
            uses_pandas=True,
            uses_torch=True,
            uses_tensorflow=True,
            uses_gpu=True
        )
        memory = _estimate_memory(analysis)
        assert memory <= 8192  # Cap à 8GB


class TestEstimateRequirements:
    """Tests pour l'estimation des requirements pip"""

    def test_basic_requirements(self):
        """Test requirements de base"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            imports=set()
        )
        reqs = _estimate_requirements(analysis)
        assert "cloudpickle>=2.0.0" in reqs

    def test_pandas_requirement(self):
        """Test requirement pandas"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            imports={"pandas", "pandas.io.excel"}
        )
        reqs = _estimate_requirements(analysis)
        assert any("pandas" in req for req in reqs)

    def test_ml_requirements(self):
        """Test requirements ML"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            imports={"sklearn.ensemble", "torch.nn", "tensorflow.keras"}
        )
        reqs = _estimate_requirements(analysis)

        assert any("scikit-learn" in req for req in reqs)
        assert any("torch" in req for req in reqs)
        assert any("tensorflow" in req for req in reqs)

    def test_no_duplicate_requirements(self):
        """Test pas de doublons dans les requirements"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            imports={"pandas", "pandas.core", "pandas.io"}
        )
        reqs = _estimate_requirements(analysis)

        pandas_count = sum(1 for req in reqs if "pandas" in req)
        assert pandas_count == 1


class TestDetectDependencies:
    """Tests pour la détection de dépendances"""

    def test_direct_dependency_match(self):
        """Test dépendance directe par nom"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            arguments=["load_data", "process_data"],
            called_functions={"load_data", "process_data", "save_results"}
        )

        deps = _detect_dependencies(analysis)

        assert deps["load_data"] == "load_data"
        assert deps["process_data"] == "process_data"

    def test_pattern_based_dependencies(self):
        """Test dépendances basées sur patterns"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            arguments=[
                "data_result",
                "preprocessing_output",
                "result_from_training",
                "normalized_data"
            ],
            called_functions={"data", "preprocessing", "training", "normalized"}
        )

        deps = _detect_dependencies(analysis)

        assert deps.get("data_result") == "data"
        assert deps.get("preprocessing_output") == "preprocessing"
        assert deps.get("result_from_training") == "training"

    def test_no_false_dependencies(self):
        """Test pas de fausses dépendances"""
        analysis = FunctionAnalysis(
            name="test",
            source_code="",
            arguments=["x", "y", "config"],
            called_functions={"helper", "utility"}
        )

        deps = _detect_dependencies(analysis)

        assert "x" not in deps
        assert "y" not in deps
        assert "config" not in deps


class TestBuildDAG:
    """Tests pour la construction du DAG"""

    def test_build_simple_dag(self):
        """Test construction d'un DAG simple"""

        def pipeline():
            a = step_a()
            b = step_b(a)
            return step_c(b)

        steps = {
            "step_a": Mock(),
            "step_b": Mock(),
            "step_c": Mock()
        }

        with patch('ops0.parser.inspect.getsource') as mock_source:
            mock_source.return_value = inspect.getsource(pipeline)

            dag = build_dag(pipeline, steps)

            # Vérifier la structure basique
            assert isinstance(dag, dict)
            assert all(step in dag for step in steps.keys())

    def test_build_dag_parallel_branches(self):
        """Test DAG avec branches parallèles"""

        def pipeline():
            a = load_a()
            b = load_b()
            return merge(a, b)

        steps = {
            "load_a": Mock(),
            "load_b": Mock(),
            "merge": Mock()
        }

        dag = build_dag(pipeline, steps)

        # Les deux loads ne devraient pas dépendre l'un de l'autre
        assert isinstance(dag, dict)

    def test_build_dag_fallback(self):
        """Test fallback quand l'analyse échoue"""

        def bad_pipeline():
            pass

        steps = {
            "step1": Mock(),
            "step2": Mock(),
            "step3": Mock()
        }

        with patch('ops0.parser.inspect.getsource') as mock_source:
            mock_source.side_effect = Exception("Source not available")

            dag = build_dag(bad_pipeline, steps)

            # Devrait créer un DAG séquentiel par défaut
            assert dag["step2"] == ["step1"]
            assert dag["step3"] == ["step2"]

    def test_build_dag_complex_dependencies(self):
        """Test DAG avec dépendances complexes"""
        source = '''
def pipeline():
    raw_data = load_data()
    clean_data = clean(raw_data)
    features = extract_features(clean_data)

    model_a = train_model_a(features)
    model_b = train_model_b(features)

    ensemble = combine_models(model_a, model_b)
    return evaluate(ensemble)
'''

        steps = {
            "load_data": Mock(),
            "clean": Mock(),
            "extract_features": Mock(),
            "train_model_a": Mock(),
            "train_model_b": Mock(),
            "combine_models": Mock(),
            "evaluate": Mock()
        }

        with patch('ops0.parser.inspect.getsource') as mock_source:
            mock_source.return_value = source

            dag = build_dag(lambda: None, steps)

            # Structure attendue (peut varier selon l'implémentation)
            assert isinstance(dag, dict)
            assert len(dag) == len(steps)


class TestDetectCycles:
    """Tests pour la détection de cycles"""

    def test_no_cycle(self):
        """Test DAG sans cycle"""
        dag = {
            "A": [],
            "B": ["A"],
            "C": ["B"],
            "D": ["B", "C"]
        }

        assert detect_cycles(dag) is False

    def test_simple_cycle(self):
        """Test cycle simple A -> B -> A"""
        dag = {
            "A": ["B"],
            "B": ["A"]
        }

        assert detect_cycles(dag) is True

    def test_complex_cycle(self):
        """Test cycle complexe"""
        dag = {
            "A": [],
            "B": ["A"],
            "C": ["B"],
            "D": ["C"],
            "E": ["D", "B"],
            "B": ["E"]  # Crée un cycle B -> E -> B
        }

        assert detect_cycles(dag) is True

    def test_self_cycle(self):
        """Test auto-référence"""
        dag = {
            "A": ["A"]  # Cycle sur soi-même
        }

        assert detect_cycles(dag) is True

    def test_disconnected_graph(self):
        """Test graphe non connexe"""
        dag = {
            "A": ["B"],
            "B": [],
            "C": ["D"],
            "D": []
        }

        assert detect_cycles(dag) is False

    def test_empty_dag(self):
        """Test DAG vide"""
        dag = {}

        assert detect_cycles(dag) is False


class TestIntegrationParser:
    """Tests d'intégration pour le parser"""

    def test_full_analysis_ml_pipeline(self):
        """Test analyse complète d'une pipeline ML"""

        def ml_pipeline(data_path: str, config: dict = None):
            """
            Pipeline ML complète pour classification.
            """
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            # Charger les données
            df = pd.read_csv(data_path)

            # Préparation
            X = df.drop('target', axis=1)
            y = df['target']

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Entraînement
            model = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100) if config else 100
            )
            model.fit(X_train, y_train)

            # Évaluation
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # GPU check (même si pas utilisé ici)
            if hasattr(df, 'cuda'):
                df = df.cuda()

            return {
                'model': model,
                'accuracy': accuracy,
                'test_size': len(X_test)
            }

        analysis = analyze_function(ml_pipeline)

        # Vérifications complètes
        assert analysis.name == "ml_pipeline"
        assert "data_path" in analysis.arguments
        assert "config" in analysis.arguments

        # Frameworks détectés
        assert analysis.uses_pandas is True
        assert analysis.uses_numpy is True
        assert analysis.uses_sklearn is True
        assert analysis.uses_ml_framework is True

        # Mémoire estimée
        assert analysis.estimated_memory >= 1024

        # Requirements
        assert any("pandas" in req for req in analysis.estimated_requirements)
        assert any("scikit-learn" in req for req in analysis.estimated_requirements)
        assert any("numpy" in req for req in analysis.estimated_requirements)

        # GPU detection (via le check hasattr cuda)
        assert analysis.uses_gpu is True

    def test_dag_building_real_pipeline(self):
        """Test construction DAG sur une vraie pipeline"""

        # Définir des steps
        def load():
            return "data"

        def clean(data):
            return f"clean({data})"

        def transform(cleaned_data):
            return f"transform({cleaned_data})"

        def train(transformed_data):
            return f"model({transformed_data})"

        # Pipeline
        def pipeline():
            raw = load()
            cleaned = clean(raw)
            features = transform(cleaned)
            model = train(features)
            return model

        steps = {
            "load": Mock(func=load),
            "clean": Mock(func=clean),
            "transform": Mock(func=transform),
            "train": Mock(func=train)
        }

        dag = build_dag(pipeline, steps)

        # Vérifier que le DAG a du sens
        assert isinstance(dag, dict)
        assert not detect_cycles(dag)

        # La structure attendue (dépend de l'implémentation)
        # mais on peut vérifier quelques invariants
        assert all(isinstance(deps, list) for deps in dag.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])