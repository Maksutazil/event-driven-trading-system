"""
Custom exceptions for the machine learning module.

This module defines exception classes that can be raised by the machine learning components.
They provide more specific error information than generic exceptions.
"""

class MLModuleError(Exception):
    """Base exception class for all ML module errors."""
    pass


class ModelError(MLModuleError):
    """Base exception class for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when attempting to access a model that does not exist."""
    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found", *args, **kwargs)


class ModelLoadError(ModelError):
    """Raised when there is an error loading a model."""
    def __init__(self, model_path, error_details=None, *args, **kwargs):
        self.model_path = model_path
        message = f"Failed to load model from '{model_path}'"
        if error_details:
            message += f": {error_details}"
        super().__init__(message, *args, **kwargs)


class ModelSaveError(ModelError):
    """Raised when there is an error saving a model."""
    def __init__(self, model_id, file_path, error_details=None, *args, **kwargs):
        self.model_id = model_id
        self.file_path = file_path
        message = f"Failed to save model '{model_id}' to '{file_path}'"
        if error_details:
            message += f": {error_details}"
        super().__init__(message, *args, **kwargs)


class ModelUpdateError(ModelError):
    """Raised when there is an error updating a model."""
    def __init__(self, model_id, error_details=None, *args, **kwargs):
        self.model_id = model_id
        message = f"Failed to update model '{model_id}'"
        if error_details:
            message += f": {error_details}"
        super().__init__(message, *args, **kwargs)


class ModelPredictionError(ModelError):
    """Raised when there is an error making a prediction with a model."""
    def __init__(self, model_id, error_details=None, *args, **kwargs):
        self.model_id = model_id
        message = f"Failed to make prediction with model '{model_id}'"
        if error_details:
            message += f": {error_details}"
        super().__init__(message, *args, **kwargs)


class TransformerError(MLModuleError):
    """Base exception class for transformer-related errors."""
    pass


class TransformerNotFoundError(TransformerError):
    """Raised when attempting to access a transformer that does not exist."""
    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id
        super().__init__(f"No transformer registered for model '{model_id}'", *args, **kwargs)


class TransformerFitError(TransformerError):
    """Raised when there is an error fitting a transformer."""
    def __init__(self, error_details=None, *args, **kwargs):
        message = "Failed to fit transformer"
        if error_details:
            message += f": {error_details}"
        super().__init__(message, *args, **kwargs)


class TransformerTransformError(TransformerError):
    """Raised when there is an error transforming features."""
    def __init__(self, error_details=None, *args, **kwargs):
        message = "Failed to transform features"
        if error_details:
            message += f": {error_details}"
        super().__init__(message, *args, **kwargs)


class InvalidFeatureError(MLModuleError):
    """Raised when there is an error with features."""
    def __init__(self, feature_name=None, error_details=None, *args, **kwargs):
        message = "Invalid feature"
        if feature_name:
            message += f" '{feature_name}'"
        if error_details:
            message += f": {error_details}"
        super().__init__(message, *args, **kwargs)


class MissingFeatureError(InvalidFeatureError):
    """Raised when a required feature is missing."""
    def __init__(self, feature_name, *args, **kwargs):
        super().__init__(feature_name, "Feature is missing", *args, **kwargs)


class InvalidModelTypeError(MLModuleError):
    """Raised when an invalid model type is specified."""
    def __init__(self, model_type, *args, **kwargs):
        self.model_type = model_type
        super().__init__(f"Invalid model type '{model_type}'. Expected 'classification' or 'regression'.", *args, **kwargs) 