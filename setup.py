from setuptools import setup, find_packages

setup(
    name="ops0",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0",
        "cloudpickle>=2.0",
        "boto3>=1.26",
        "aws-cdk-lib>=2.100",
        "jinja2>=3.0",
    ],
    entry_points={
        "console_scripts": [
            "ops0=ops0.cli:main",
        ],
    },
    python_requires=">=3.8",
)