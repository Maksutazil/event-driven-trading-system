#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Error Handler Module

This module provides centralized error handling for the trading components.
It handles recovery from transient failures, detailed error reporting, and ensures
consistent error management across trading components.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, TypeVar, Type, List, Union
from datetime import datetime
from threading import RLock
from functools import wraps

from src.core.events import EventBus, Event, EventType

logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')

# Define trading-specific exceptions
class TradingError(Exception):
    """Base exception for all trading-related errors."""
    pass

class SignalGenerationError(TradingError):
    """Error occurred during signal generation."""
    pass

class TradeExecutionError(TradingError):
    """Error occurred during trade execution."""
    pass

class PositionManagementError(TradingError):
    """Error occurred during position management."""
    pass

class RiskCalculationError(TradingError):
    """Error occurred during risk calculation."""
    pass

class TokenMonitoringError(TradingError):
    """Error occurred during token monitoring."""
    pass

class PriceFetchError(TradingError):
    """Error occurred when fetching price data."""
    pass

class InvalidParameterError(TradingError):
    """Invalid parameter provided to a trading function."""
    pass

class TradingConfigError(TradingError):
    """Error in the trading system configuration."""
    pass

class TradingErrorHandler:
    """
    Centralized error handler for trading components.
    
    This class provides:
    1. Error tracking and analytics
    2. Recovery strategies for different error types
    3. Error event publishing
    4. Retry mechanisms for transient failures
    5. Graceful degradation
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
        
        logger.info("Trading Error Handler initialized")
    
    def _register_default_recovery_handlers(self) -> None:
        """Register default recovery handlers for known error types."""
        # Price fetch error recovery
        self.register_recovery_handler(
            PriceFetchError, 
            lambda err, ctx: self._default_price_fetch_recovery(err, ctx)
        )
        
        # Trade execution error recovery
        self.register_recovery_handler(
            TradeExecutionError,
            lambda err, ctx: self._default_trade_execution_recovery(err, ctx)
        )
        
        # Signal generation error recovery
        self.register_recovery_handler(
            SignalGenerationError,
            lambda err, ctx: self._default_signal_generation_recovery(err, ctx)
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
        token_id = context.get('token_id', 'unknown')
        position_id = context.get('position_id', 'unknown')
        
        # Create error record
        error_record = {
            'error_type': error_type,
            'message': str(error),
            'component': component,
            'operation': operation,
            'token_id': token_id,
            'position_id': position_id,
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
            f"Trading Error in {component}.{operation}: {error_type}: {str(error)}",
            extra={
                'token_id': token_id,
                'position_id': position_id,
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
                'token_id': error_record['token_id'],
                'position_id': error_record['position_id'],
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
                source=f"trading.{error_record['component']}"
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
            exceptions: List of exception types to retry on (default: all TradingError types)
            
        Returns:
            Decorated function with retry logic
        """
        exceptions = exceptions or [TradingError]
        
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
        
        This can be used as a class decorator without an instance of TradingErrorHandler.
        
        Args:
            max_attempts: Maximum number of retry attempts
            delay: Initial delay between retries in seconds
            backoff: Multiplier for delay between retries
            exceptions: List of exception types to retry on (default: all TradingError types)
            
        Returns:
            Decorated function with retry logic
        """
        exceptions = exceptions or [TradingError]
        
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
    
    def _default_price_fetch_recovery(self, 
                                     error: PriceFetchError, 
                                     context: Dict[str, Any]) -> Any:
        """
        Default recovery for PriceFetchError.
        
        Args:
            error: The PriceFetchError
            context: Additional context
            
        Returns:
            Last known price or estimated price
        """
        # Try to use last known price
        if 'last_known_price' in context and context['last_known_price'] > 0:
            logger.info(f"Recovered from price fetch error using last known price: {context['last_known_price']}")
            return context['last_known_price']
            
        # Try to use estimated price from context
        if 'estimated_price' in context and context['estimated_price'] > 0:
            logger.info(f"Recovered from price fetch error using estimated price: {context['estimated_price']}")
            return context['estimated_price']
            
        # Can't recover
        raise error
    
    def _default_trade_execution_recovery(self, 
                                         error: TradeExecutionError, 
                                         context: Dict[str, Any]) -> Any:
        """
        Default recovery for TradeExecutionError.
        
        Args:
            error: The TradeExecutionError
            context: Additional context
            
        Returns:
            Execution result or False
        """
        if context.get('critical', False):
            # For critical operations (like closing positions), we can't recover
            logger.error(f"Critical trade execution error, cannot recover: {error}")
            raise error
            
        # For non-critical operations, we can just return False to indicate failure
        logger.warning(f"Non-critical trade execution error, returning failure: {error}")
        return False
    
    def _default_signal_generation_recovery(self, 
                                          error: SignalGenerationError, 
                                          context: Dict[str, Any]) -> Any:
        """
        Default recovery for SignalGenerationError.
        
        Args:
            error: The SignalGenerationError
            context: Additional context
            
        Returns:
            Empty list of signals
        """
        logger.warning(f"Signal generation error, returning empty signals list: {error}")
        return [] 