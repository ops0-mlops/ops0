"""
ops0 Function Analyzer - AST-based analysis of Python functions.

This is the magic that makes ops0 understand your code automatically.
"""

import ast
import inspect
import hashlib
from typing import Set, Dict, Any, Callable, Type, List, Optional
import logging

logger = logging.getLogger(__name__)


class FunctionAnalyzer:
    """
    Analyzes Python functions using AST to extract dependencies and metadata.

    This is the magic that makes ops0 understand your code automatically.
    """

    def __init__(self, func: Callable):
        self.func = func
        try:
            self.source = inspect.getsource(func)
            self.tree = ast.parse(self.source)
            self.signature = inspect.signature(func)
        except (OSError, TypeError) as e:
            logger.warning(f"Could not analyze function {func.__name__}: {e}")
            self.source = ""
            self.tree = ast.Module(body=[], type_ignores=[])
            self.signature = None

    def get_dependencies(self) -> Set[str]:
        """
        Extract ops0.storage.load() calls to understand data dependencies.

        Returns:
            Set of storage keys this function depends on
        """
        dependencies = set()

        class DependencyVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Look for storage.load("key") patterns
                if self._is_storage_load(node):
                    key = self._extract_string_arg(node)
                    if key:
                        dependencies.add(key)

                # Look for ops0.storage.load("key") patterns
                elif self._is_ops0_storage_load(node):
                    key = self._extract_string_arg(node)
                    if key:
                        dependencies.add(key)

                self.generic_visit(node)

            def _is_storage_load(self, node: ast.Call) -> bool:
                """Check if call is storage.load()"""
                return (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == "storage" and
                        node.func.attr == "load")

            def _is_ops0_storage_load(self, node: ast.Call) -> bool:
                """Check if call is ops0.storage.load()"""
                return (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Attribute) and
                        isinstance(node.func.value.value, ast.Name) and
                        node.func.value.value.id == "ops0" and
                        node.func.value.attr == "storage" and
                        node.func.attr == "load")

            def _extract_string_arg(self, node: ast.Call) -> Optional[str]:
                """Extract string argument from function call"""
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Str):  # Python < 3.8
                        return arg.s
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):  # Python >= 3.8
                        return arg.value
                return None

        visitor = DependencyVisitor()
        visitor.visit(self.tree)
        return dependencies

    def get_input_signature(self) -> Dict[str, Any]:
        """Extract function parameters with type hints"""
        inputs = {}
        if not self.signature:
            return inputs

        for param_name, param in self.signature.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default = param.default if param.default != inspect.Parameter.empty else None
            inputs[param_name] = {
                "type": param_type,
                "default": default,
                "kind": param.kind.name
            }
        return inputs

    def get_output_signature(self) -> Type:
        """Extract return type hint"""
        if not self.signature:
            return Any
        return_annotation = self.signature.return_annotation
        return return_annotation if return_annotation != inspect.Signature.empty else Any

    def get_source_hash(self) -> str:
        """Generate hash of source code for reproducibility"""
        if not self.source:
            return "unknown"
        return hashlib.sha256(self.source.encode()).hexdigest()[:12]

    def get_storage_saves(self) -> Set[str]:
        """Extract ops0.storage.save() calls to understand data outputs"""
        saves = set()

        class SaveVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Look for storage.save("key", data) patterns
                if self._is_storage_save(node):
                    key = self._extract_string_arg(node)
                    if key:
                        saves.add(key)

                # Look for ops0.storage.save("key", data) patterns
                elif self._is_ops0_storage_save(node):
                    key = self._extract_string_arg(node)
                    if key:
                        saves.add(key)

                self.generic_visit(node)

            def _is_storage_save(self, node: ast.Call) -> bool:
                """Check if call is storage.save()"""
                return (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == "storage" and
                        node.func.attr == "save")

            def _is_ops0_storage_save(self, node: ast.Call) -> bool:
                """Check if call is ops0.storage.save()"""
                return (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Attribute) and
                        isinstance(node.func.value.value, ast.Name) and
                        node.func.value.value.id == "ops0" and
                        node.func.value.attr == "storage" and
                        node.func.attr == "save")

            def _extract_string_arg(self, node: ast.Call) -> Optional[str]:
                """Extract string argument from function call"""
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Str):  # Python < 3.8
                        return arg.s
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):  # Python >= 3.8
                        return arg.value
                return None

        visitor = SaveVisitor()
        visitor.visit(self.tree)
        return saves

    def get_imported_modules(self) -> Set[str]:
        """Extract imported modules"""
        imports = set()

        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.add(alias.name)

            def visit_ImportFrom(self, node):
                if node.module:
                    imports.add(node.module)

        visitor = ImportVisitor()
        visitor.visit(self.tree)
        return imports

    def get_function_calls(self) -> List[str]:
        """Extract function calls made within the function"""
        calls = []

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        calls.append(f"{node.func.value.id}.{node.func.attr}")
                self.generic_visit(node)

        visitor = CallVisitor()
        visitor.visit(self.tree)
        return calls

    def has_ml_frameworks(self) -> bool:
        """Check if function uses common ML frameworks"""
        ml_patterns = [
            "sklearn", "torch", "tensorflow", "tf", "keras",
            "pandas", "numpy", "scipy", "xgboost", "lightgbm"
        ]

        imports = self.get_imported_modules()
        calls = self.get_function_calls()

        return any(
            pattern in " ".join(imports) or pattern in " ".join(calls)
            for pattern in ml_patterns
        )

    def estimate_complexity(self) -> str:
        """Estimate function complexity based on AST analysis"""
        if not self.tree:
            return "unknown"

        complexity_score = 0

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.For, ast.While)):
                complexity_score += 2
            elif isinstance(node, (ast.If, ast.Try)):
                complexity_score += 1
            elif isinstance(node, ast.FunctionDef):
                complexity_score += 1

        if complexity_score <= 5:
            return "low"
        elif complexity_score <= 10:
            return "medium"
        else:
            return "high"

    def get_metadata(self) -> Dict[str, Any]:
        """Get comprehensive function metadata"""
        return {
            "name": self.func.__name__,
            "module": self.func.__module__ if hasattr(self.func, "__module__") else "unknown",
            "dependencies": list(self.get_dependencies()),
            "storage_saves": list(self.get_storage_saves()),
            "inputs": self.get_input_signature(),
            "output_type": str(self.get_output_signature()),
            "source_hash": self.get_source_hash(),
            "imports": list(self.get_imported_modules()),
            "function_calls": self.get_function_calls(),
            "has_ml_frameworks": self.has_ml_frameworks(),
            "complexity": self.estimate_complexity(),
            "docstring": self.func.__doc__,
        }