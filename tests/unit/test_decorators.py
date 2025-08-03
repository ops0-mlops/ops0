"""
Tests for ops0 decorators (@ops0.step and @ops0.pipeline)
"""
import pytest
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.ops0.core.decorators import step, pipeline, StepMetadata
from src.ops0.core.graph import PipelineGraph


class TestStepDecorator:
    """Test the @ops0.step decorator functionality"""

    def test_step_decorator_basic(self):
        """Test basic @ops0.step functionality"""

        @step
        def sample_step(x: int) -> int:
            return x * 2

        # Check that function is marked as ops0 step
        assert hasattr(sample_step, '_ops0_step')
        assert sample_step._ops0_step is True
        assert hasattr(sample_step, '_ops0_metadata')

        # Check that the original function is preserved
        assert hasattr(sample_step, '_original_func')
        assert sample_step._original_func.__name__ == 'sample_step'

    def test_step_metadata_extraction(self):
        """Test that step metadata is correctly extracted"""

        @step
        def test_step(a: str, b: int = 5) -> dict:
            """Test step docstring"""
            return {"a": a, "b": b}

        metadata = test_step._ops0_metadata
        assert isinstance(metadata, StepMetadata)
        assert metadata.name == 'test_step'
        assert metadata.func == test_step._original_func

    def test_step_with_storage_dependencies(self):
        """Test step that uses ops0.storage.load()"""

        @step
        def step_with_deps():
            # This should be detected as dependency
            from src.ops0.core.storage import storage
            data = storage.load("input_data")
            return data * 2

        metadata = step_with_deps._ops0_metadata
        # Dependencies would be detected by AST analyzer
        assert metadata.dependencies is not None

    def test_step_execution_in_pipeline(self):
        """Test step execution within pipeline context"""
        with PipelineGraph("test-pipeline") as pipeline:
            @step
            def test_step():
                return {"result": "success"}

            # Step should be registered in pipeline
            assert "test_step" in pipeline.steps
            step_node = pipeline.steps["test_step"]
            assert step_node.name == "test_step"

    def test_step_without_pipeline_context(self):
        """Test step execution outside pipeline context"""

        @step
        def standalone_step():
            return "standalone result"

        # Should work but not be registered anywhere
        result = standalone_step()
        assert result == "standalone result"

    def test_multiple_steps_dependency_detection(self):
        """Test multiple steps with dependencies"""
        with PipelineGraph("multi-step-test"):
            @step
            def step_one():
                from src.ops0.core.storage import storage
                storage.save("data1", [1, 2, 3])
                return "step1 done"

            @step
            def step_two():
                from src.ops0.core.storage import storage
                data = storage.load("data1")  # Depends on step_one
                storage.save("data2", data + [4, 5])
                return "step2 done"

            # Both steps should be registered
            pipeline = PipelineGraph.get_current()
            assert len(pipeline.steps) == 2
            assert "step_one" in pipeline.steps
            assert "step_two" in pipeline.steps


class TestPipelineDecorator:
    """Test the @ops0.pipeline decorator functionality"""

    def test_pipeline_context_manager(self):
        """Test pipeline as context manager"""
        with pipeline("test-context-pipeline") as p:
            assert isinstance(p, PipelineGraph)
            assert p.name == "test-context-pipeline"
            assert PipelineGraph.get_current() == p

        # After context, no current pipeline
        assert PipelineGraph.get_current() is None

    def test_pipeline_decorator_with_name(self):
        """Test @pipeline("name") decorator"""

        @pipeline("named-pipeline")
        def my_pipeline():
            @step
            def inner_step():
                return "inner result"

            return "pipeline result"

        assert hasattr(my_pipeline, '_ops0_pipeline')
        assert my_pipeline._ops0_pipeline is True

    def test_pipeline_decorator_without_name(self):
        """Test @pipeline decorator using function name"""

        @pipeline
        def auto_named_pipeline():
            @step
            def inner_step():
                return "inner result"

            return "pipeline result"

        assert hasattr(auto_named_pipeline, '_ops0_pipeline')

    def test_nested_pipeline_error(self):
        """Test that nested pipelines raise appropriate error"""
        with pipeline("outer-pipeline"):
            with pytest.raises(Exception):  # Should not allow nested pipelines
                with pipeline("inner-pipeline"):
                    pass

    def test_pipeline_step_registration(self):
        """Test that steps are properly registered in pipeline"""
        with pipeline("registration-test") as p:
            @step
            def first_step():
                return 1

            @step
            def second_step():
                return 2

            # Check steps are registered
            assert len(p.steps) == 2
            assert "first_step" in p.steps
            assert "second_step" in p.steps


class TestStepMetadata:
    """Test StepMetadata class functionality"""

    def test_metadata_creation(self):
        """Test creation of step metadata"""

        def sample_func(x: int, y: str = "default") -> bool:
            return True

        metadata = StepMetadata(sample_func)
        assert metadata.name == "sample_func"
        assert metadata.func == sample_func
        assert metadata.source_hash is not None
        assert len(metadata.source_hash) == 12  # Short hash

    def test_metadata_with_type_hints(self):
        """Test metadata extraction with type hints"""

        def typed_func(a: int, b: str, c: float = 1.0) -> dict:
            return {"a": a, "b": b, "c": c}

        metadata = StepMetadata(typed_func)
        # Input signature should be extracted
        assert metadata.inputs is not None
        # Output signature should be extracted
        assert metadata.outputs is not None

    def test_metadata_repr(self):
        """Test string representation of metadata"""

        def test_func():
            pass

        metadata = StepMetadata(test_func)
        repr_str = repr(metadata)
        assert "StepMetadata" in repr_str
        assert "test_func" in repr_str


# Integration tests
class TestDecoratorIntegration:
    """Integration tests for decorators working together"""

    def test_complete_pipeline_flow(self):
        """Test complete flow from pipeline definition to execution"""
        executed_steps = []

        with pipeline("integration-test") as p:
            @step
            def extract():
                executed_steps.append("extract")
                return [1, 2, 3, 4, 5]

            @step
            def transform():
                executed_steps.append("transform")
                return [x * 2 for x in [1, 2, 3, 4, 5]]

            @step
            def load():
                executed_steps.append("load")
                return "data loaded"

        # Pipeline should have all steps registered
        assert len(p.steps) == 3
        assert "extract" in p.steps
        assert "transform" in p.steps
        assert "load" in p.steps

        # Execution order should be determinable
        execution_order = p.build_execution_order()
        assert len(execution_order) > 0

    def test_step_dependency_chain(self):
        """Test steps with actual storage dependencies"""
        with pipeline("dependency-chain") as p:
            @step
            def step_a():
                from src.ops0.core.storage import storage
                storage.save("result_a", "data from A")
                return "A complete"

            @step
            def step_b():
                from src.ops0.core.storage import storage
                data_a = storage.load("result_a")
                storage.save("result_b", f"B processed {data_a}")
                return "B complete"

            @step
            def step_c():
                from src.ops0.core.storage import storage
                data_b = storage.load("result_b")
                return f"C final: {data_b}"

        # Verify dependency chain is detected
        execution_order = p.build_execution_order()

        # step_a should be in first level (no dependencies)
        assert "step_a" in execution_order[0]

        # step_b should be after step_a
        step_b_level = None
        for i, level in enumerate(execution_order):
            if "step_b" in level:
                step_b_level = i
                break
        assert step_b_level > 0  # Should not be in first level

    @patch('ops0.core.storage.storage')
    def test_step_with_mocked_storage(self, mock_storage):
        """Test step execution with mocked storage"""
        mock_storage.load.return_value = {"test": "data"}
        mock_storage.save.return_value = None

        @step
        def test_step():
            from src.ops0.core.storage import storage
            data = storage.load("test_key")
            storage.save("output_key", data)
            return data

        # Execute step
        result = test_step()

        # Verify storage calls
        mock_storage.load.assert_called_once_with("test_key")
        mock_storage.save.assert_called_once_with("output_key", {"test": "data"})
        assert result == {"test": "data"}


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])