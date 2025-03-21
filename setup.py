#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for the PumpFun trading system package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trade-event-system",
    version="0.1.0",
    author="PumpFun Team",
    author_email="example@example.com",
    description="Event-driven trading system with real-time data feeds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trade-event-system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "websockets>=10.0",
        "psycopg2-binary>=2.9.3",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-asyncio>=0.15.1",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "isort>=5.9.3",
            "flake8>=3.9.2",
        ],
    },
)