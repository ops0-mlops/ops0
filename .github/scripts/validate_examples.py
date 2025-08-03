#!/usr/bin/env python3
"""
Validate ops0 pipeline examples to ensure they work correctly.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


class ExampleValidator:
    def __init__(self, examples_dir: str = "examples"):
        self.examples_dir = Path(examples_dir)
        self.failed_examples = []

    def find_example_files(self) -> List[Path]:
        """Find all example Python files."""
        example_files = []
        for pattern in ["**/pipeline.py", "**/example.py", "**/*_example.py"]:
            example_files.extend(self.examples_dir.rglob(pattern))
        return example_files

    def validate_example(self, example_file: Path) -> Tuple[bool, str]:
        """Validate a single example file."""
        try:
            # Set test environment
            env = os.environ.copy()
            env["OPS0_ENV"] = "test"
            env["PYTHONPATH"] = str(Path.cwd() / "src")

            # Try to validate the pipeline
            result = subprocess.run(
                [sys.executable, str(example_file), "--validate"],
                cwd=example_file.parent,
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return True, "Validation successful"
            else:
                return False, f"Validation failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            return False, "Validation timed out"
        except Exception as e:
            return False, f"Validation error: {e}"

    def run(self) -> int:
        """Run validation on all examples."""
        example_files = self.find_example_files()

        if not example_files:
            print("No example files found")
            return 0

        print(f"Validating {len(example_files)} example files...")

        for example_file in example_files:
            print(f"Validating {example_file}...")
            success, message = self.validate_example(example_file)

            if success:
                print(f"  ✅ {message}")
            else:
                print(f"  ❌ {message}")
                self.failed_examples.append(str(example_file))

        # Summary
        if self.failed_examples:
            print(f"\n❌ {len(self.failed_examples)} examples failed:")
            for example in self.failed_examples:
                print(f"  - {example}")
            return 1
        else:
            print(f"\n✅ All {len(example_files)} examples validated successfully!")
            return 0


if __name__ == "__main__":
    validator = ExampleValidator()
    sys.exit(validator.run())