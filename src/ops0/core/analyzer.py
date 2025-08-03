import ast
import inspect
import hashlib
from typing import Set, Dict, Any, Callable, Type


class FunctionAnalyzer:
    """
    Analyzes Python functions using AST to extract dependencies and metadata.

    This is the magic that makes ops0 understand your code automatically.
    """

    def __init__(self, func: Callable):
        self.func = func
        self.source = inspect.getsource(func)
        self.tree = ast.parse(self.source)
        self.signature = inspect.signature(func)

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
                if (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == "storage" and
                        node.func.attr == "load"):

                    if node.args and isinstance(node.args[0], ast.Str):
                        dependencies.add(node.args[0].s)
                    elif node.args and isinstance(node.args[0], ast.Constant):
                        dependencies.add(node.args[0].value)

                self.generic_visit(node)

        visitor = DependencyVisitor()
        visitor.visit(self.tree)
        return dependencies

    def get_input_signature(self) -> Dict[str, Type]:
        """Extract function parameters with type hints"""
        inputs = {}
        for param_name, param in self.signature.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default = param.default if param.default != inspect.Parameter.empty else None
            inputs[param_name] = (param_type, default)
        return inputs

    def get_output_signature(self) -> Type:
        """Extract return type hint"""
        return_annotation = self.signature.return_annotation
        return return_annotation if return_annotation != inspect.Signature.empty else Any

    def get_source_hash(self) -> str:
        """Generate hash of source code for reproducibility"""
        return hashlib.sha256(self.source.encode()).hexdigest()[:12]

    def get_storage_saves(self) -> Set[str]:
        """Extract ops0.storage.save() calls to understand data outputs"""
        saves = set()

        class SaveVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Look for storage.save("key", data) patterns
                if (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == "storage" and
                        node.func.attr == "save"):

                    if node.args and isinstance(node.args[0], ast.Str):
                        saves.add(node.args[0].s)
                    elif node.args and isinstance(node.args[0], ast.Constant):
                        saves.add(node.args[0].value)

                self.generic_visit(node)

        visitor = SaveVisitor()
        visitor.visit(self.tree)
        return saves