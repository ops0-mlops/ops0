"""
Setup configuration for ops0 - Zero-config MLOps platform
"""
from setuptools import setup, find_packages
from pathlib import Path

# Lire le README pour la description longue
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Lire les requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split("\n")

# Version
version = "0.1.0"

setup(
    name="ops0",
    version=version,
    author="ops0 Contributors",
    author_email="hello@ops0.xyz",
    description="Write Python, Ship Production - Zero-configuration MLOps platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ops0-mlops/ops0",
    project_urls={
        "Bug Tracker": "https://github.com/ops0-mlops/ops0/issues",
        "Documentation": "https://docs.ops0.xyz",
        "Source Code": "https://github.com/ops0-mlops/ops0",
    },

    # Packages
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,

    # Requirements
    python_requires=">=3.8",
    install_requires=requirements,

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-mock>=3.0",
            "black>=22.0",
            "isort>=5.0",
            "ruff>=0.0.280",
        ],
        "ml": [
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
        ],
        "aws": [
            "boto3>=1.26.0",
            "aws-cdk-lib>=2.100.0",
        ],
    },

    # Entry points - CRITICAL pour la commande CLI
    entry_points={
        "console_scripts": [
            "ops0=ops0.cli:main",
        ],
    },

    # Classificateurs PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],

    keywords="mlops, machine-learning, pipeline, deployment, automation, ml-engineering",
)