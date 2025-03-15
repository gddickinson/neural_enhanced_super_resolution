#!/usr/bin/env python3
"""
NESR - Neural Enhanced Super-Resolution
Setup script for installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nesr",
    version="0.1.0",
    author="NESR Team",
    author_email="info@example.com",
    description="Neural Enhanced Super-Resolution - A multi-model approach to image upscaling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/nesr",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "diffusers>=0.14.0",
        "transformers>=4.25.0",
        "opencv-python>=4.6.0",
        "Pillow>=9.3.0",
        "numpy>=1.23.0",
        "tqdm>=4.64.0",
        "realesrgan>=0.3.0",
        "basicsr>=1.4.2",
        "huggingface-hub>=0.12.0",
        "accelerate>=0.16.0",
        "PyQt5>=5.15.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "gui": ["PyQt5>=5.15.0", "qtmodern>=0.2.0", "pyqtgraph>=0.13.0"],
        "dev": ["pytest", "black", "isort", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "nesr=main:main",
        ],
    },
)
