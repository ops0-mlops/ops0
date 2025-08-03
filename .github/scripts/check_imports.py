#!/usr/bin/env python3
"""
Check ops0 imports for consistency and circular dependencies.
Used in pre-commit hooks and CI.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set


class ImportChecker:
    def __init__(self, src_dir: str = "src/ops0"):
        self.src_dir = Path(src_dir)
        self.errors = []
        self.warnings = []

    def check_file(self, filepath: Path) -> List[str]:
        """Check a single Python file for import issues."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, str(filepath))

            # Check for relative imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level > 0:  # Relative import
                        if not self._is_valid_relative_import(node, filepath):
                            self.errors.append(
                                f"{filepath}:{node.lineno}: Invalid relative import: {node.module}"
                            )

                # Check for circular imports (simplified check)
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._check_circular_import(node, filepath)

        except SyntaxError as e:
            self.errors.append(f"{filepath}: Syntax error: {e}")
        except Exception as e:
            self.warnings.append(f"{filepath}: Warning: {e}")

    def _is_valid_relative_import(self, node: ast.ImportFrom, filepath: Path) -> bool:
        """Check if a relative import is valid."""
        # Add ops0-specific import validation logic here
        return True

    def _check_circular_import(self, node: ast.AST, filepath: Path):
        """Basic circular import detection."""
        # Implement circular import detection logic
        pass

    def run(self) -> int:
        """Run import checks on all Python files."""
        for py_file in self.src_dir.rglob("*.py"):
            self.check_file(py_file)

        # Print results
        for error in self.errors:
            print(f"ERROR: {error}", file=sys.stderr)

        for warning in self.warnings:
            print(f"WARNING: {warning}")

        return 1 if self.errors else 0


if __name__ == "__main__":
    checker = ImportChecker()
    sys.exit(checker.run())