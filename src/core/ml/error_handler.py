#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Error Handler Module

This module provides centralized error handling for the machine learning components.
It handles recovery from transient failures, detailed error reporting, and ensures
consistent error management across ML components.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, TypeVar, Type, List, Union
from datetime import datetime
from threading import RLock
from functools import wraps

from src.core.events import EventBus, Event, EventType
from src.core.ml.exceptions import (
    MLModuleError, ModelError, ModelNotFoundError, ModelLoadError, ModelSaveError, ModelUpdateError,
    ModelPredictionError, TransformerError, TransformerNotFoundError, TransformerFitError,
    TransformerTransformError, InvalidFeatureError, MissingFeatureError, InvalidModelTypeError
)

logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')

class MLErrorHandler:
    """
    Centralized error handler for ML components.
    
    This class provides:
    1. Error tracking and analytics
    2. Recovery strategies for different error types
    3. Error event publishing
    4. Retry mechanisms for transient failures
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, max_errors: int = 1000):
        """
        Initialize the error handler.
        
        Args:
            event_bus: Optional event bus for publishing error events
            max_errors: Maximum number of errors to store in history
        """
        self.event_bus = event_bus
        self.max_errors = max_errors
        
        # Error history
        self._error_history: List[Dict[str, Any]] = []
        self._error_counts: Dict[str, int] = {}
        self._error_lock = RLock()
        
        # Recovery handlers for specific error types
        self._recovery_handlers: Dict[Type[Exception], Callable] = {}
        
        # Register default recovery handlers
        self._register_default_recovery_handlers()
        
        logger.info("ML Error Handler initialized")
    
    def _register_default_recovery_handlers(self) -> None:
        """Register default recovery handlers for known error types."""
        # Model not found recovery
        self.register_recovery_handler(
            ModelNotFoundError, 
            lambda err, ctx: self._default_model_not_found_recovery(err, ctx)
        )
        
        # Missing feature recovery
        self.register_recovery_handler(
            MissingFeatureError,
            lambda err, ctx: self._default_missing_feature_recovery(err, ctx)
        )
        
        # Transformer error recovery
        self.register_recovery_handler(
            TransformerError,
            lambda err, ctx: self._default_transformer_error_recovery(err, ctx)
        )
    
    def register_recovery_handler(self, 
                                 error_type: Type[Exception], 
                                 handler: Callable[[Exception, Dict[str, Any]], Any]) -> None:
        """
        Register a recovery handler for a specific error type.
        
        Args:
            error_type: The type of exception to handle
            handler: Function that takes (error, context) and returns a recovery value or raises
        """
        self._recovery_handlers[error_type] = handler
        logger.debug(f"Registered recovery handler for {error_type.__name__}")
    
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any] = None,
                    publish_event: bool = True) -> None:
        """
        Process an error, log it, and attempt recovery if possible.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            publish_event: Whether to publish an error event
        """
        context = context or {}
        error_type = type(error).__name__
        component = context.get('component', 'unknown')
        operation = context.get('operation', 'unknown')
        model_id = context.get('model_id', 'unknown')
        token_id = context.get('token_id', 'unknown')
        
        # Create error record
        error_record = {
            'error_type': error_type,
            'message': str(error),
            'component': component,
            'operation': operation,
            'model_id': model_id,
            'token_id': token_id,
            'timestamp': datetime.now(),
            'traceback': traceback.format_exc(),
            'context': context
        }
        
        # Add to history with thread safety
        with self._error_lock:
            self._error_history.append(error_record)
            
            # Maintain max size
            if len(self._error_history) > self.max_errors:
                self._error_history = self._error_history[-self.max_errors:]
            
            # Update error count
            err_key = f"{component}:{operation}:{error_type}"
            self._error_counts[err_key] = self._error_counts.get(err_key, 0) + 1
        
        # Log the error
        logger.error(
            f"ML Error in {component}.{operation}: {error_type}: {str(error)}",
            extra={
                'model_id': model_id,
                'token_id': token_id,
                'component': component,
                'operation': operation
            }
        )
        
        # Publish error event if requested
        if publish_event and self.event_bus:
            self._publish_error_event(error_record)
    
    def _publish_error_event(self, error_record: Dict[str, Any]) -> None:
        """
        Publish an error event to the event bus.
        
        Args:
            error_record: The error details
        """
        if not self.event_bus:
            return
            
        try:
            # Create event data
            event_data = {
                'error_type': error_record['error_type'],
                'message': error_record['message'],
                'component': error_record['component'],
                'operation': error_record['operation'],
                'model_id': error_record['model_id'],
                'token_id': error_record['token_id'],
                'timestamp': time.time()
            }
            
            # Add additional context that might be useful
            for key in ['severity', 'is_recoverable', 'recommended_action']:
                if key in error_record['context']:
                    event_data[key] = error_record['context'][key]
            
            self.event_bus.publish(Event(
                event_type=EventType.ERROR,
                data=event_data,
                token_id=error_record['token_id'],
                source=f"ml.{error_record['component']}"
            ))
            
        except Exception as e:
            logger.error(f"Failed to publish error event: {e}", exc_info=True)
    
    def try_recover(self, 
                   error: Exception, 
                   context: Dict[str, Any] = None) -> Optional[Any]:
        """
        Attempt to recover from an error using registered recovery handlers.
        
        Args:
            error: The exception to recover from
            context: Additional context for recovery
            
        Returns:
            Recovery value if successful, None otherwise
            
        Raises:
            The original exception if recovery fails and re-raise is specified
        """
        context = context or {}
        error_type = type(error)
        
        # Check if we have a handler for this error type
        for exc_type, handler in self._recovery_handlers.items():
            if isinstance(error, exc_type):
                logger.info(f"Attempting recovery for {error_type.__name__}")
                try:
                    return handler(error, context)
                except Exception as recovery_error:
                    logger.error(
                        f"Recovery failed for {error_type.__name__}: {recovery_error}",
                        exc_info=True
                    )
                    break
        
        # If we should re-raise the error
        if context.get('re_raise', False):
            raise error
            
        return None
    
    def retry(self, 
             max_attempts: int = 3, 
             delay: float = 1.0, 
             backoff: float = 2.0,
             exceptions: Optional[List[Type[Exception]]] = None) -> Callable:
        """
        Decorator for retrying operations that might fail with transient errors.
        
        Args:
            max_attempts: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff: Multiplier for delay between retries
            exceptions: List of exception types to retry on (default: all MLModuleError types)
            
        Returns:
            Decorated function with retry logic
        """
        exceptions = exceptions or [MLModuleError]
        
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except tuple(exceptions) as e:
                        last_exception = e
                        
                        # Log the retry attempt
                        logger.warning(
                            f"Retry {attempt+1}/{max_attempts} for {func.__name__} due to {type(e).__name__}: {str(e)}"
                        )
                        
                        # If this is the last attempt, don't sleep
                        if attempt < max_attempts - 1:
                            time.sleep(current_delay)
                            current_delay *= backoff
                    except Exception as e:
                        # For non-retryable exceptions, just raise immediately
                        raise
                
                # If we get here, all retries failed
                if last_exception:
                    raise last_exception
                    
                # This shouldn't happen, but just in case
                raise RuntimeError(f"All {max_attempts} retry attempts failed without an exception")
                
            return wrapper
        return decorator
    
    @staticmethod
    def retry_static(max_attempts: int = 3, 
                     delay: float = 1.0, 
                     backoff: float = 2.0,
                     exceptions: Optional[List[Type[Exception]]] = None) -> Callable:
        """
        Static version of the retry decorator for retrying operations that might fail with transient errors.
        
        This can be used as a class decorator without an instance of MLErrorHandler.
        
        Args:
            max_attempts: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff: Multiplier for delay between retries
            exceptions: List of exception types to retry on (default: all MLModuleError types)
            
        Returns:
            Decorated function with retry logic
        """
        exceptions = exceptions or [MLModuleError]
        
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except tuple(exceptions) as e:
                        last_exception = e
                        
                        # Log the retry attempt
                        logger.warning(
                            f"Retry {attempt+1}/{max_attempts} for {func.__name__} due to {type(e).__name__}: {str(e)}"
                        )
                        
                        # If this is the last attempt, don't sleep
                        if attempt < max_attempts - 1:
                            time.sleep(current_delay)
                            current_delay *= backoff
                    except Exception as e:
                        # For non-retryable exceptions, just raise immediately
                        raise
                
                # If we get here, all retries failed
                if last_exception:
                    raise last_exception
                    
                # This shouldn't happen, but just in case
                raise RuntimeError(f"All {max_attempts} retry attempts failed without an exception")
                
            return wrapper
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors handled by this error handler.
        
        Returns:
            Dictionary with error statistics
        """
        with self._error_lock:
            stats = {
                'total_errors': len(self._error_history),
                'unique_error_types': len(set(err['error_type'] for err in self._error_history)),
                'error_counts': dict(self._error_counts),
                'recent_errors': self._error_history[-10:] if self._error_history else []
            }
            
            # Get counts by component
            component_counts = {}
            for err in self._error_history:
                component = err['component']
                component_counts[component] = component_counts.get(component, 0) + 1
                
            stats['component_error_counts'] = component_counts
            
            return stats
    
    def clear_error_history(self) -> None:
        """Clear the error history and counts."""
        with self._error_lock:
            self._error_history.clear()
            self._error_counts.clear()
    
    # Default recovery handlers for common errors
    
    def _default_model_not_found_recovery(self, 
                                         error: ModelNotFoundError, 
                                         context: Dict[str, Any]) -> Any:
        """
        Default recovery for ModelNotFoundError.
        
        Args:
            error: The ModelNotFoundError
            context: Additional context
            
        Returns:
            Default value or raises the exception
        """
        fallback_model = context.get('fallback_model')
        if fallback_model:
            logger.info(f"Using fallback model {fallback_model} instead of {error.model_id}")
            if 'fallback_operation' in context and callable(context['fallback_operation']):
                return context['fallback_operation'](fallback_model)
        
        # We can't recover, raise or return None
        if context.get('fail_silently', False):
            logger.warning(f"Model {error.model_id} not found, returning None as fallback")
            return None
            
        raise error
    
    def _default_missing_feature_recovery(self, 
                                         error: MissingFeatureError, 
                                         context: Dict[str, Any]) -> Any:
        """
        Default recovery for MissingFeatureError.
        
        Args:
            error: The MissingFeatureError
            context: Additional context
            
        Returns:
            Features dictionary with default values
        """
        if 'required_features' in context and 'available_features' in context:
            required = context['required_features']
            available = context['available_features']
            
            # Create a new features dict with defaults for missing ones
            result = dict(available)
            default_value = context.get('default_feature_value', 0.0)
            
            for feature in required:
                if feature not in available:
                    result[feature] = default_value
                    logger.warning(f"Using default value {default_value} for missing feature '{feature}'")
            
            return result
        
        # Can't recover without required context
        raise error
    
    def _default_transformer_error_recovery(self, 
                                          error: TransformerError, 
                                          context: Dict[str, Any]) -> Any:
        """
        Default recovery for TransformerError.
        
        Args:
            error: The TransformerError
            context: Additional context
            
        Returns:
            Raw features if provided, otherwise raises
        """
        if 'raw_features' in context:
            logger.warning("Transformer error, using raw features as fallback")
            return context['raw_features']
            
        raise error 