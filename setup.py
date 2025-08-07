"""
Setup configuration for SimulationSDK package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from __init__.py
version = "0.1.0"
with open("simulation_sdk/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="simulation-sdk",
    version=version,
    author="SimulationSDK Team",
    author_email="team@simulationsdk.example.com",
    description="A comprehensive Python framework for creating and evaluating conversational simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simulation-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "yaml": ["pyyaml>=6.0"],
        "openai": ["openai>=1.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-asyncio>=0.20",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "all": [
            "pyyaml>=6.0",
            "openai>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "simulation-sdk=simulation_sdk.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)