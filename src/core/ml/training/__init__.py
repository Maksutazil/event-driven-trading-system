#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Package

This package provides functionality for training models on historical data
from various sources, including PostgreSQL databases.
"""

from .model_trainer import ModelTrainer

__all__ = [
    'ModelTrainer',
] 