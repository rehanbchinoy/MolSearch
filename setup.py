#!/usr/bin/env python3
"""
Setup script for Molecular Search Pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="molsearch",
    version="0.1.0",
    description="Molecular similarity search using RDKit and Streamlit",
    author="Rehan Chinoy",
    packages=find_packages(),
    python_requires=">=3.11,<3.12",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "rdkit-pypi>=2023.3.1b1",
        "streamlit>=1.28.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black",
            "flake8",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    entry_points={
        "console_scripts": [
            "molsearch=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "chemistry",
        "molecular",
        "similarity",
        "search",
        "machine-learning",
        "ai",
        "drug-discovery",
        "cheminformatics",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/MolSearch-GPT/issues",
        "Source": "https://github.com/yourusername/MolSearch-GPT",
        "Documentation": "https://github.com/yourusername/MolSearch-GPT#readme",
    },
)
