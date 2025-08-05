"""
ops0 Function Analyzer

AST-based analysis of Python functions to extract pipeline metadata.
Automatically detects dependencies, inputs, outputs, and serialization needs.
"""

import ast
import inspect
import hashlib
from typing import Dict, List, Set, Any, Optional, Callable, Tuple
from dataclasses import dataclass

from .exceptions import StepError, ValidationError


@dataclass
class FunctionSignature:
    """Represents a function's input/output signature"""
    name: str
    parameters: Dict[str, type]
    return_type: Optional[type]
    defaults: Dict[str, Any]

    def __str__(self):
        params = []
        for name, param_type in self.parameters.items():
            type_str = getattr(param_type, '__name__', str(param_type))
            if name in self.defaults:
                params.append(f"{name}: {type_str} = {self.defaults[name]}")
            else:
                params.append(f"{name}: {type_str}")

        return_str = ""
        if self.return_type:
            return_type_str = getattr(self.return_type, '__name__', str(self.return_type))
            return_str = f" -> {return_type_str}"

        return f"{self.name}({', '.join(params)}){return_str}"


@dataclass
class StorageDependency:
    """Represents a storage dependency found in the function"""
    key: str
    operation: str  # 'load' or 'save'
    line_number: int
    variable_name: Optional[str] = None


class StorageCallVisitor(ast.NodeVisitor):
    """AST visitor to find ops0.storage calls"""

    def __init__(self):
        self.dependencies: List[StorageDependency] = []
        self.current_line = 0

    def visit_Call(self, node: ast.Call):
        """Visit function calls to detect storage operations"""
        self.current_line = node.lineno

        # Look for ops0.storage.load() or ops0.storage.save()
        if self._is_storage_call(node):
            operation, key_value = self._extract_storage_operation(node)
            if operation and key_value:
                # Try to get variable name for load operations
                var_name = None
                if operation == 'load' and hasattr(node, 'parent_assign'):
                    var_name = self._get_assigned_variable(node)

                self.dependencies.append(StorageDependency(
                    key=key_value,
                    operation=operation,
                    line_number=self.current_line,
                    variable_name=var_name
                ))

        self.generic_visit(node)

    def _is_storage_call(self, node: ast.Call) -> bool:
        """Check if this is a storage operation call"""
        # Pattern: ops0.storage.load() or storage.load()
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute):
                # ops0.storage.load
                if (isinstance(node.func.value.value, ast.Name) and
                    node.func.value.value.id == 'ops0' and
                    node.func.value.attr == 'storage'):
                    return node.func.attr in ['load', 'save']
            elif isinstance(node.func.value, ast.Name):
                # storage.load (after import)
                if (node.func.value.id == 'storage'):
                    return node.func.attr in ['load', 'save']
        return False

    def _extract_storage_operation(self, node: ast.Call) -> Tuple[Optional[str], Optional[str]]:
        """Extract operation type and key from storage call"""
        operation = node.func.attr

        # Get the key (first argument)
        if node.args and len(node.args) > 0:
            key_arg = node.args[0]
            if isinstance(key_arg, ast.Str):
                return operation, key_arg.s
            elif isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                return operation, key_arg.value

        return operation, None

    def _get_assigned_variable(self, node: ast.Call) -> Optional[str]:
        """Get the variable name if this call is part of an assignment"""
        # This would need parent tracking - simplified for now
        return None


class FunctionAnalyzer:
    """Analyzes Python functions to extract ops0 metadata"""

    def __init__(self, func: Callable):
        self.func = func
        self.source_code = self._get_source_code()
        self.ast_tree = self._parse_ast()
        self._signature = None
        self._dependencies = None
        self._source_hash = None

    def _get_source_code(self) -> str:
        """Get the source code of the function"""
        try:
            source = inspect.getsource(self.func)
            # Fix indentation issues
            lines = source.split('\n')
            if lines:
                # Find minimum indentation (excluding empty lines)
                min_indent = float('inf')
                for line in lines:
                    if line.strip():  # Skip empty lines
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)

                # Remove minimum indentation from all lines
                if min_indent < float('inf') and min_indent > 0:
                    fixed_lines = []
                    for line in lines:
                        if len(line) >= min_indent:
                            fixed_lines.append(line[min_indent:])
                        else:
                            fixed_lines.append(line)
                    return '\n'.join(fixed_lines)

            return source
        except OSError as e:
            raise StepError(
                f"Could not retrieve source code for function '{self.func.__name__}'",
                step_name=self.func.__name__,
                context={"error": str(e)}
            )

    def _parse_ast(self) -> ast.AST:
        """Parse the function source code into an AST"""
        try:
            return ast.parse(self.source_code)
        except SyntaxError as e:
            raise StepError(
                f"Syntax error in function '{self.func.__name__}'",
                step_name=self.func.__name__,
                context={"error": str(e), "line": e.lineno}
            )

    def get_input_signature(self) -> FunctionSignature:
        """Extract the function's input signature"""
        if self._signature is None:
            self._signature = self._analyze_signature()
        return self._signature

    def get_output_signature(self) -> Optional[type]:
        """Extract the function's return type"""
        signature = inspect.signature(self.func)
        return signature.return_annotation if signature.return_annotation != inspect.Signature.empty else None

    def get_dependencies(self) -> List[StorageDependency]:
        """Extract storage dependencies from the function"""
        if self._dependencies is None:
            self._dependencies = self._analyze_dependencies()
        return self._dependencies

    def get_source_hash(self) -> str:
        """Get a hash of the function source for caching"""
        if self._source_hash is None:
            self._source_hash = hashlib.sha256(self.source_code.encode()).hexdigest()[:16]
        return self._source_hash

    def _analyze_signature(self) -> FunctionSignature:
        """Analyze the function signature"""
        signature = inspect.signature(self.func)

        parameters = {}
        defaults = {}

        for param_name, param in signature.parameters.items():
            # Get parameter type
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            parameters[param_name] = param_type

            # Get default value
            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default

        return FunctionSignature(
            name=self.func.__name__,
            parameters=parameters,
            return_type=self.get_output_signature(),
            defaults=defaults
        )

    def _analyze_dependencies(self) -> List[StorageDependency]:
        """Analyze storage dependencies using AST"""
        visitor = StorageCallVisitor()
        visitor.visit(self.ast_tree)
        return visitor.dependencies

    def validate_step_function(self) -> bool:
        """Validate that the function can be used as an ops0 step"""
        try:
            # Check if function is callable
            if not callable(self.func):
                raise ValidationError("Step must be a callable function")

            # Check if we can analyze the function
            signature = self.get_input_signature()
            dependencies = self.get_dependencies()

            # Validate no conflicting parameter names with storage keys
            storage_keys = {dep.key for dep in dependencies if dep.operation == 'load'}
            param_names = set(signature.parameters.keys())

            conflicts = storage_keys.intersection(param_names)
            if conflicts:
                raise ValidationError(
                    f"Parameter names conflict with storage keys: {conflicts}",
                    context={"conflicts": list(conflicts)}
                )

            return True

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Step validation failed: {str(e)}")

    def get_resource_requirements(self) -> Dict[str, Any]:
        """Analyze function to determine resource requirements"""
        requirements = {
            "cpu": "100m",  # Default CPU
            "memory": "128Mi",  # Default memory
            "gpu": False,
            "disk_space": "1Gi"
        }

        # Simple heuristics based on function analysis
        source_lower = self.source_code.lower()

        # Check for GPU indicators
        gpu_indicators = ['torch', 'tensorflow', 'gpu', 'cuda', 'cupy']
        if any(indicator in source_lower for indicator in gpu_indicators):
            requirements["gpu"] = True
            requirements["memory"] = "2Gi"  # More memory for GPU workloads

        # Check for data processing indicators
        data_indicators = ['pandas', 'numpy', 'scipy', 'sklearn']
        if any(indicator in source_lower for indicator in data_indicators):
            requirements["memory"] = "512Mi"  # More memory for data processing

        # Check for ML model indicators
        ml_indicators = ['model', 'predict', 'train', 'fit']
        if any(indicator in source_lower for indicator in ml_indicators):
            requirements["cpu"] = "500m"  # More CPU for ML

        return requirements

    def get_serialization_hints(self) -> Dict[str, str]:
        """Suggest serialization formats based on function analysis"""
        hints = {}

        # Analyze return type and code to suggest serialization
        return_type = self.get_output_signature()
        source_lower = self.source_code.lower()

        # Common patterns
        if 'pandas' in source_lower or 'dataframe' in source_lower:
            hints['default'] = 'parquet'
        elif 'numpy' in source_lower or 'array' in source_lower:
            hints['default'] = 'numpy'
        elif 'torch' in source_lower or 'tensor' in source_lower:
            hints['default'] = 'torch'
        elif return_type and hasattr(return_type, '__name__'):
            if 'dict' in return_type.__name__.lower():
                hints['default'] = 'json'
            else:
                hints['default'] = 'pickle'
        else:
            hints['default'] = 'pickle'

        return hints