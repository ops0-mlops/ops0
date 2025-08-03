#!/usr/bin/env python3
"""
Check version consistency across different files in the repository.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Optional


class VersionChecker:
    def __init__(self):
        self.version_files = {
            "src/ops0/__about__.py": r'__version__\s*=\s*["\']([^"\']+)["\']',
            "pyproject.toml": r'version\s*=\s*["\']([^"\']+)["\']',
        }
        self.versions = {}

    def extract_version(self, filepath: str, pattern: str) -> Optional[str]:
        """Extract version from a file using regex pattern."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            match = re.search(pattern, content)
            if match:
                return match.group(1)
            else:
                print(f"Warning: No version found in {filepath}")
                return None

        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    def run(self) -> int:
        """Check version consistency."""
        # Extract versions from all files
        for filepath, pattern in self.version_files.items():
            version = self.extract_version(filepath, pattern)
            if version:
                self.versions[filepath] = version

        if not self.versions:
            print("Error: No versions found in any file")
            return 1

        # Check consistency
        unique_versions = set(self.versions.values())

        if len(unique_versions) == 1:
            version = list(unique_versions)[0]
            print(f"✅ Version consistency check passed: {version}")
            return 0
        else:
            print("❌ Version inconsistency detected:")
            for filepath, version in self.versions.items():
                print(f"  {filepath}: {version}")
            return 1


if __name__ == "__main__":
    checker = VersionChecker()
    sys.exit(checker.run())