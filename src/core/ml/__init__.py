#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Learning Module for the Event-Driven Trading System.

This module provides the necessary components for incorporating machine learning
into the trading system, including model management, feature transformation,
and model adapters.

Core components:
- Model: Interface for machine learning models
- FeatureTransformer: Interface for feature transformation
- ModelManager: Interface for managing models and making predictions
- DefaultModelManager: Implementation of the ModelManager interface
- MLErrorHandler: Centralized error handling for ML components
"""

# Import interfaces
from src.core.ml.interfaces import Model, FeatureTransformer, ModelManager

# Import default implementation
from src.core.ml.model_manager import DefaultModelManager

# Import transformers
from src.core.ml.transformers import StandardFeatureTransformer

# Import model adapters
from src.core.ml.adapters import ScikitLearnModel, ScikitLearnModelAdapter

# Import error handler
from src.core.ml.error_handler import MLErrorHandler

# Import exceptions
from src.core.ml.exceptions import (
    MLModuleError, 
    ModelError, ModelNotFoundError, ModelLoadError, ModelSaveError, ModelUpdateError, ModelPredictionError,
    TransformerError, TransformerNotFoundError, TransformerFitError, TransformerTransformError,
    InvalidFeatureError, MissingFeatureError, InvalidModelTypeError
)

__all__ = [
    # Interfaces
    'Model',
    'FeatureTransformer',
    'ModelManager',
    
    # Implementations
    'DefaultModelManager',
    
    # Transformers
    'StandardFeatureTransformer',
    
    # Model adapters
    'ScikitLearnModel',
    'ScikitLearnModelAdapter',
    
    # Error handling
    'MLErrorHandler',
    
    # Exceptions
    'MLModuleError',
    'ModelError',
    'ModelNotFoundError',
    'ModelLoadError',
    'ModelSaveError',
    'ModelUpdateError',
    'ModelPredictionError',
    'TransformerError',
    'TransformerNotFoundError',
    'TransformerFitError',
    'TransformerTransformError',
    'InvalidFeatureError',
    'MissingFeatureError',
    'InvalidModelTypeError'
] 