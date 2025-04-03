#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Adapters Package

This package provides adapters for different types of machine learning models,
allowing them to be used with the ModelManager interface.
"""

from .scikit_learn import ScikitLearnModel, ScikitLearnModelAdapter

__all__ = [
    'ScikitLearnModel',
    'ScikitLearnModelAdapter',
] 