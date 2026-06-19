"""
Setup script for Hyperparameter Tuning Agent
"""
from setuptools import setup, find_packages

setup(
    name="hyperparameter-tuning-agent",
    version="1.0.0",
    description="Hyperparameter Tuning Agent for MITRA AI using Optuna",
    author="Epic-4 Team",
    packages=find_packages(),
    install_requires=[
        "optuna>=3.0.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "ray>=2.0.0",  # For parallel tuning
    ],
    entry_points={
        "console_scripts": [
            "hpt-agent=hyperparameter_tuning_agent.cli:main",
        ],
    },
    python_requires=">=3.8",
)