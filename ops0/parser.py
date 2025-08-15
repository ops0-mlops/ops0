"""
AST-based code analysis for automatic dependency detection and resource estimation.
Zero configuration through intelligent code analysis.
"""
import ast
import inspect
import re
from typing import Dict, List, Set, Any
from dataclasses import dataclass, field



@dataclass
class FunctionAnalysis:
    """Results of analyzing a function"""
    name: str
    source_code: str
    imports: Set[str] = field(default_factory=set)
    called_functions: Set[str] = field(default_factory=set)
    global_vars: Set[str] = field(default_factory=set)
    arguments: List[str] = field(default_factory=list)
    returns: bool = False
    uses_pandas: bool = False
    uses_numpy: bool = False
    uses_sklearn: bool = False
    uses_torch: bool = False
    uses_tensorflow: bool = False
    uses_ml_framework: bool = False
    uses_gpu: bool = False
    estimated_memory: int = 512  # MB
    estimated_requirements: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)  # arg_name -> function_name


class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function metadata"""

    def __init__(self):
        self.imports = set()
        self.called_functions = set()
        self.global_vars = set()
        self.attributes_accessed = set()
        self.ml_indicators = {
            'pandas': False,
            'numpy': False,
            'sklearn': False,
            'torch': False,
            'tensorflow': False,
            'gpu': False
        }

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
            self._check_ml_framework(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module)
            self._check_ml_framework(node.module)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Extract function calls
        if isinstance(node.func, ast.Name):
            self.called_functions.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls like pd.read_csv
            if isinstance(node.func.value, ast.Name):
                self.attributes_accessed.add(f"{node.func.value.id}.{node.func.attr}")

        # Check for GPU indicators
        if self._is_gpu_call(node):
            self.ml_indicators['gpu'] = True

        self.generic_visit(node)

    def visit_Name(self, node):
        # Track global variable usage
        if isinstance(node.ctx, ast.Load):
            self.global_vars.add(node.id)
        self.generic_visit(node)

    def _check_ml_framework(self, module_name: str):
        """Check if import indicates ML framework usage"""
        if 'pandas' in module_name:
            self.ml_indicators['pandas'] = True
        elif 'numpy' in module_name or module_name == 'np':
            self.ml_indicators['numpy'] = True
        elif 'sklearn' in module_name or 'scikit' in module_name:
            self.ml_indicators['sklearn'] = True
        elif 'torch' in module_name or 'pytorch' in module_name:
            self.ml_indicators['torch'] = True
        elif 'tensorflow' in module_name or module_name == 'tf':
            self.ml_indicators['tensorflow'] = True

    def _is_gpu_call(self, node) -> bool:
        """Check if the call indicates GPU usage"""
        gpu_indicators = [
            'cuda', 'gpu', 'GPU', 'to_device', 'cuda()', '.cuda',
            'device="cuda"', 'device=\'cuda\'', 'gpu_id'
        ]

        node_str = ast.dump(node)
        return any(indicator in node_str for indicator in gpu_indicators)


def analyze_function(func: callable) -> FunctionAnalysis:
    """
    Analyze a function to extract dependencies and resource requirements.

    This is the core of ops0's zero-configuration approach.
    """
    try:
        source = inspect.getsource(func)
    except Exception:
        # If we can't get source, return minimal analysis
        return FunctionAnalysis(
            name=func.__name__,
            source_code="",
            arguments=list(inspect.signature(func).parameters.keys())
        )

    # Parse the function's AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FunctionAnalysis(
            name=func.__name__,
            source_code=source,
            arguments=list(inspect.signature(func).parameters.keys())
        )

    # Visit the AST
    visitor = FunctionVisitor()
    visitor.visit(tree)

    # Extract function arguments
    sig = inspect.signature(func)
    arguments = list(sig.parameters.keys())

    # Build analysis result
    analysis = FunctionAnalysis(
        name=func.__name__,
        source_code=source,
        imports=visitor.imports,
        called_functions=visitor.called_functions,
        global_vars=visitor.global_vars,
        arguments=arguments,
        returns=sig.return_annotation != sig.empty,
        uses_pandas=visitor.ml_indicators['pandas'],
        uses_numpy=visitor.ml_indicators['numpy'],
        uses_sklearn=visitor.ml_indicators['sklearn'],
        uses_torch=visitor.ml_indicators['torch'],
        uses_tensorflow=visitor.ml_indicators['tensorflow'],
        uses_gpu=visitor.ml_indicators['gpu']
    )

    # Set ML framework flag
    analysis.uses_ml_framework = any([
        analysis.uses_sklearn,
        analysis.uses_torch,
        analysis.uses_tensorflow
    ])

    # Estimate memory requirements based on framework usage
    analysis.estimated_memory = _estimate_memory(analysis)

    # Estimate pip requirements
    analysis.estimated_requirements = _estimate_requirements(analysis)

    # Detect dependencies between functions
    analysis.dependencies = _detect_dependencies(analysis)

    return analysis


def _estimate_memory(analysis: FunctionAnalysis) -> int:
    """Estimate memory requirements based on code analysis"""
    base_memory = 512  # MB

    # Add memory based on framework usage
    if analysis.uses_pandas:
        base_memory += 512
    if analysis.uses_numpy:
        base_memory += 256
    if analysis.uses_sklearn:
        base_memory += 512
    if analysis.uses_torch or analysis.uses_tensorflow:
        base_memory += 2048
    if analysis.uses_gpu:
        base_memory += 1024

    # Check for indicators of large data processing
    large_data_indicators = [
        'read_csv', 'read_parquet', 'load_dataset',
        'DataFrame', 'large', 'big', 'batch'
    ]

    if any(ind in analysis.source_code for ind in large_data_indicators):
        base_memory = int(base_memory * 1.5)

    return min(base_memory, 8192)  # Cap at 8GB


def _estimate_requirements(analysis: FunctionAnalysis) -> List[str]:
    """Estimate pip requirements based on imports"""
    requirements = []

    # Map common imports to pip packages
    import_to_package = {
        'pandas': 'pandas>=1.3.0',
        'numpy': 'numpy>=1.21.0',
        'sklearn': 'scikit-learn>=1.0.0',
        'scikit': 'scikit-learn>=1.0.0',
        'torch': 'torch>=1.10.0',
        'tensorflow': 'tensorflow>=2.7.0',
        'xgboost': 'xgboost>=1.5.0',
        'lightgbm': 'lightgbm>=3.3.0',
        'matplotlib': 'matplotlib>=3.5.0',
        'seaborn': 'seaborn>=0.11.0',
        'plotly': 'plotly>=5.0.0',
        'requests': 'requests>=2.26.0',
        'boto3': 'boto3>=1.20.0',
        'sqlalchemy': 'sqlalchemy>=1.4.0'
    }

    for imp in analysis.imports:
        # Extract base module name
        base_module = imp.split('.')[0]
        if base_module in import_to_package:
            req = import_to_package[base_module]
            if req not in requirements:
                requirements.append(req)

    # Always include cloudpickle for serialization
    requirements.append('cloudpickle>=2.0.0')

    return requirements


def _detect_dependencies(analysis: FunctionAnalysis) -> Dict[str, str]:
    """
    Detect dependencies between functions based on argument names.

    This enables automatic DAG construction without explicit configuration.
    """
    dependencies = {}

    # Simple heuristic: if an argument name matches a called function name
    # or a pattern like "result_from_X", it likely depends on function X
    for arg in analysis.arguments:
        # Direct match
        if arg in analysis.called_functions:
            dependencies[arg] = arg

        # Pattern matching for common naming conventions
        patterns = [
            (r'(.+)_result$', 1),  # X_result -> X
            (r'(.+)_output$', 1),  # X_output -> X
            (r'result_from_(.+)$', 1),  # result_from_X -> X
            (r'(.+)_data$', 1),  # X_data -> X
            (r'processed_(.+)$', 1),  # processed_X -> X
        ]

        for pattern, group in patterns:
            match = re.match(pattern, arg)
            if match:
                potential_func = match.group(group)
                if potential_func in analysis.called_functions:
                    dependencies[arg] = potential_func

    return dependencies


def detect_cycles(dag: Dict[str, List[str]]) -> bool:
    """Détecter les dépendances circulaires"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in dag}

    def has_cycle(node):
        if color[node] == GRAY:
            return True
        if color[node] == BLACK:
            return False

        color[node] = GRAY
        for neighbor in dag.get(node, []):
            if neighbor in color and has_cycle(neighbor):
                return True
        color[node] = BLACK
        return False

    return any(has_cycle(node) for node in dag if color[node] == WHITE)


def build_dag(pipeline_func: callable, steps: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Build a DAG from a pipeline function by analyzing function calls and dependencies.

    Returns a dict mapping step names to their dependencies.
    """
    analysis = analyze_function(pipeline_func)
    dag = {}

    # Initialize all steps with empty dependencies
    for step_name in steps:
        dag[step_name] = []

    # Analyze the pipeline function body to determine execution order
    # This is a simplified version - a full implementation would do deeper AST analysis
    try:
        source = inspect.getsource(pipeline_func)
        tree = ast.parse(source)

        # Find assignment statements to track data flow
        assignments = {}  # variable_name -> function_that_produced_it

        class PipelineVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    func_name = node.value.func.id
                    if func_name in steps:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                assignments[target.id] = func_name
                self.generic_visit(node)

        visitor = PipelineVisitor()
        visitor.visit(tree)

        # Now find dependencies by looking at function arguments
        class DependencyVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in steps:
                    func_name = node.func.id
                    # Check arguments
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id in assignments:
                            dependency = assignments[arg.id]
                            if dependency not in dag[func_name]:
                                dag[func_name].append(dependency)
                self.generic_visit(node)

        dep_visitor = DependencyVisitor()
        dep_visitor.visit(tree)

    except Exception:
        # Fallback to simple sequential execution
        step_list = list(steps.keys())
        for i in range(1, len(step_list)):
            dag[step_list[i]] = [step_list[i - 1]]

    return dag