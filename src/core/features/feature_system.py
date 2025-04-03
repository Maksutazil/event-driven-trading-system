#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature System Module

This module provides the FeatureSystem class which serves as the central
component for managing features, feature providers, and feature computation.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta

from .interfaces import Feature, FeatureProvider, FeatureManager, FeatureConsumer
from .cache import InMemoryFeatureCache
from ..events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class FeatureSystem(FeatureManager):
    """
    Feature System for managing features and feature providers.
    
    The FeatureSystem serves as the central component that:
    1. Maintains a registry of available features and providers
    2. Handles feature computation with dependency resolution
    3. Manages feature caching for improved performance
    4. Publishes feature update events for reactive systems
    
    This class fully implements the FeatureManager interface.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        """
        Initialize a new FeatureSystem instance.
        
        Args:
            event_bus: Optional EventBus for publishing feature updates
        """
        # Provider and feature registries
        self._features: Dict[str, Feature] = {}
        self._providers: Dict[str, FeatureProvider] = {}
        self._feature_to_provider: Dict[str, FeatureProvider] = {}
        
        # Consumer registry - maps feature names to sets of consumers
        self._consumers: Dict[str, Set[FeatureConsumer]] = {}
        
        # Dependency tracking
        self._dependencies: Dict[str, Set[str]] = {}  # feature_name -> set of dependencies
        self._dependents: Dict[str, Set[str]] = {}    # feature_name -> set of dependent features
        
        # Caching
        self._cache = InMemoryFeatureCache()
        
        # Event bus for publishing updates
        self._event_bus = event_bus
        
        # Locks for thread safety
        self._provider_lock = threading.RLock()
        self._consumer_lock = threading.RLock()
        self._dependency_lock = threading.RLock()
        
        # Track computation in progress to avoid duplicate calculations
        self._computing: Dict[str, Dict[str, threading.Event]] = {}  # token_id -> {feature_name -> event}
        self._computing_lock = threading.RLock()
        
        # Performance metrics
        self._metrics = {
            "feature_computations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "computation_times": {},
            "computation_errors": 0,
            "last_reset_time": time.time()
        }
        self._metrics_lock = threading.RLock()
        
        logger.info("Feature system initialized")
    
    def register_feature(self, feature: Feature) -> None:
        """
        Register a feature with the system.
        
        Args:
            feature: Feature instance to register
        """
        if not feature:
            logger.warning("Attempted to register None feature")
            return
            
        feature_name = feature.name
        if not feature_name:
            logger.warning("Attempted to register feature with no name")
            return
            
        with self._provider_lock:
            if feature_name in self._features:
                logger.warning(f"Feature {feature_name} already registered. Replacing.")
            
            # Register the feature
            self._features[feature_name] = feature
            
            # Update dependency graph
            with self._dependency_lock:
                # Track dependencies of this feature
                if feature_name not in self._dependencies:
                    self._dependencies[feature_name] = set()
                
                # Add each dependency
                for dep_name in feature.dependencies:
                    self._dependencies[feature_name].add(dep_name)
                    
                    # Update reverse dependencies
                    if dep_name not in self._dependents:
                        self._dependents[dep_name] = set()
                    
                    self._dependents[dep_name].add(feature_name)
            
            logger.info(f"Registered feature: {feature_name} with dependencies: {feature.dependencies}")
    
    @property
    def cache(self) -> InMemoryFeatureCache:
        """Get the feature cache."""
        return self._cache
    
    def register_provider(self, provider: FeatureProvider) -> None:
        """
        Register a feature provider.
        
        Args:
            provider: The feature provider to register
        """
        if not provider:
            logger.warning("Attempted to register None provider")
            return
            
        provider_name = provider.name
        if not provider_name:
            logger.warning("Attempted to register provider with no name")
            return
            
        with self._provider_lock:
            if provider_name in self._providers:
                logger.warning(f"Provider {provider_name} already registered. Replacing.")
            
            # Register the provider
            self._providers[provider_name] = provider
            
            # Register all features provided by this provider
            for feature_name in provider.provides:
                if feature_name in self._feature_to_provider:
                    existing_provider = self._feature_to_provider[feature_name]
                    logger.warning(f"Feature {feature_name} already provided by {existing_provider.name}. Replacing with {provider_name}.")
                
                # Map feature to provider
                self._feature_to_provider[feature_name] = provider
                
                # Initialize dependency tracking for features from this provider
                with self._dependency_lock:
                    if feature_name not in self._dependencies:
                        self._dependencies[feature_name] = set()
                    
                    # Add dependencies for this feature
                    deps = provider.get_dependencies(feature_name)
                    for dep_name in deps:
                        self._dependencies[feature_name].add(dep_name)
                        
                        # Update reverse dependencies
                        if dep_name not in self._dependents:
                            self._dependents[dep_name] = set()
                        
                        self._dependents[dep_name].add(feature_name)
            
            logger.info(f"Registered provider: {provider_name} with features: {provider.provides}")
    
    def unregister_provider(self, provider: FeatureProvider) -> None:
        """
        Unregister a feature provider.
        
        Args:
            provider: The feature provider to unregister
        """
        if not provider:
            logger.warning("Attempted to unregister None provider")
            return
            
        provider_name = provider.name
        if not provider_name or provider_name not in self._providers:
            logger.warning(f"Provider {provider_name} not registered")
            return
            
        with self._provider_lock, self._dependency_lock:
            # Remove provider from registry
            del self._providers[provider_name]
            
            # Remove feature-to-provider mappings
            for feature_name in list(self._feature_to_provider.keys()):
                if self._feature_to_provider[feature_name] == provider:
                    del self._feature_to_provider[feature_name]
                    logger.debug(f"Removed feature-to-provider mapping for {feature_name}")
                    
                    # Clean up dependency tracking
                    if feature_name in self._dependencies:
                        deps = self._dependencies[feature_name]
                        del self._dependencies[feature_name]
                        
                        # Update reverse dependencies
                        for dep_name in deps:
                            if dep_name in self._dependents and feature_name in self._dependents[dep_name]:
                                self._dependents[dep_name].remove(feature_name)
                                if not self._dependents[dep_name]:
                                    del self._dependents[dep_name]
            
            logger.info(f"Unregistered provider: {provider_name}")
    
    def register_consumer(self, consumer: FeatureConsumer) -> None:
        """
        Register a feature consumer.
        
        Args:
            consumer: The feature consumer to register
        """
        if not consumer:
            logger.warning("Attempted to register None consumer")
            return
            
        required_features = consumer.get_required_features()
        
        with self._consumer_lock:
            registered_count = 0
            # Register consumer for each required feature
            for feature_name in required_features:
                if feature_name not in self._consumers:
                    self._consumers[feature_name] = set()
                
                # Only add if not already registered
                if consumer not in self._consumers[feature_name]:
                    self._consumers[feature_name].add(consumer)
                    registered_count += 1
            
            if registered_count > 0:
                logger.info(f"Registered consumer for {registered_count} features: {required_features}")
    
    def unregister_consumer(self, consumer: FeatureConsumer) -> None:
        """
        Unregister a feature consumer.
        
        Args:
            consumer: The feature consumer to unregister
        """
        if not consumer:
            logger.warning("Attempted to unregister None consumer")
            return
            
        with self._consumer_lock:
            unregistered_count = 0
            # Unregister consumer from each feature
            for feature_name, consumers in list(self._consumers.items()):
                if consumer in consumers:
                    consumers.remove(consumer)
                    unregistered_count += 1
                    
                    # Clean up empty consumer sets
                    if not consumers:
                        del self._consumers[feature_name]
            
            if unregistered_count > 0:
                logger.info(f"Unregistered consumer from {unregistered_count} features")
    
    def get_feature(self, token_id: str, feature_name: str, use_cache: bool = True) -> Any:
        """
        Get a feature value for a token.
        
        If the feature is in the cache and use_cache is True, return the cached value.
        Otherwise, compute the feature and update the cache.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            use_cache: Whether to use cached values if available
            
        Returns:
            Any: Feature value
            
        Raises:
            ValueError: If the feature is not available
        """
        if not token_id:
            raise ValueError("token_id cannot be None or empty")
        
        if not feature_name:
            raise ValueError("feature_name cannot be None or empty")
        
        # Check cache first if requested
        if use_cache:
            cached_value = self._cache.get(token_id, feature_name)
            if cached_value is not None:
                with self._metrics_lock:
                    self._metrics["cache_hits"] += 1
                logger.debug(f"Cache hit for feature {feature_name} of token {token_id}")
                return cached_value
            else:
                with self._metrics_lock:
                    self._metrics["cache_misses"] += 1
        
        # Get computing event to coordinate threads computing the same feature
        computing_event = self._get_computing_event(token_id, feature_name)
        
        # If the event is set, another thread is already computing this feature
        if computing_event.is_set():
            logger.debug(f"Waiting for computation of {feature_name} for {token_id} by another thread")
            
            # Wait for the computation to complete with a timeout
            if computing_event.wait(timeout=30.0):
                # Check cache again after waiting
                if use_cache:
                    cached_value = self._cache.get(token_id, feature_name)
                    if cached_value is not None:
                        with self._metrics_lock:
                            self._metrics["cache_hits"] += 1
                        logger.debug(f"Cache hit after waiting for feature {feature_name} of {token_id}")
                        return cached_value
            else:
                # Timeout - we'll compute it ourselves
                logger.warning(f"Timeout waiting for computation of {feature_name} for {token_id}")
                self._clear_computing_event(token_id, feature_name)
        
        # Set the computing flag to indicate we're computing this feature
        computing_event.set()
        
        try:
            # Compute the feature
            start_time = time.time()
            value = self.compute_feature(token_id, feature_name)
            
            # Track computation time
            elapsed = time.time() - start_time
            with self._metrics_lock:
                self._metrics["feature_computations"] += 1
                if feature_name not in self._metrics["computation_times"]:
                    self._metrics["computation_times"][feature_name] = []
                times = self._metrics["computation_times"][feature_name]
                times.append(elapsed)
                # Keep only the last 100 timings
                if len(times) > 100:
                    times.pop(0)
            
            return value
            
        except Exception as e:
            with self._metrics_lock:
                self._metrics["computation_errors"] += 1
            logger.error(f"Error computing feature {feature_name} for {token_id}: {e}", exc_info=True)
            raise ValueError(f"Failed to compute feature {feature_name} for {token_id}: {e}")
            
        finally:
            # Clear the computing flag
            self._clear_computing_event(token_id, feature_name)
    
    def compute_feature(self, token_id: str, feature_name: str, data: Optional[Dict[str, Any]] = None) -> Any:
        """
        Compute a feature value for a token.
        
        This method computes the feature value and updates the cache,
        but does not trigger consumer updates.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            data: Optional data to use for computation. If None, fetch the required data.
            
        Returns:
            Any: Computed feature value
            
        Raises:
            ValueError: If the feature is not available
        """
        if not token_id:
            raise ValueError("token_id cannot be None or empty")
        
        if not feature_name:
            raise ValueError("feature_name cannot be None or empty")
        
        # Prepare context with token_id
        context = {} if data is None else data.copy() 
        if 'token_id' not in context:
            context['token_id'] = token_id
        
        # Try to compute from provider first
        if feature_name in self._feature_to_provider:
            value = self._compute_from_provider(token_id, feature_name, context)
            return value
        
        # Try to compute from registered feature
        elif feature_name in self._features:
            value = self._compute_from_feature(token_id, feature_name, context)
            return value
        
        # Feature not found
        else:
            available_features = self.get_available_features()
            raise ValueError(f"Feature {feature_name} not available. Available features: {available_features}")
    
    def _compute_from_provider(self, token_id: str, feature_name: str, context: Dict[str, Any]) -> Any:
        """
        Compute a feature from a provider.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            context: Context data for computation
            
        Returns:
            Computed feature value
            
        Raises:
            ValueError: If computation fails or provider doesn't return the feature
        """
        provider = self._feature_to_provider[feature_name]
        
        # Get dependencies
        with self._dependency_lock:
            dependencies = self._dependencies.get(feature_name, set())
        
        # Ensure dependencies are available in context
        for dep_name in dependencies:
            if dep_name not in context:
                try:
                    dep_value = self.get_feature(token_id, dep_name, use_cache=True)
                    context[dep_name] = dep_value
                    logger.debug(f"Added dependency {dep_name}={dep_value} for {feature_name}")
                except Exception as e:
                    logger.warning(f"Failed to get dependency {dep_name} for {feature_name}: {e}")
        
        # Get features from provider
        try:
            provider_features = provider.get_features(context)
        except Exception as e:
            raise ValueError(f"Provider {provider.name} failed to compute features: {e}")
        
        # Check if the feature was returned
        if feature_name not in provider_features:
            raise ValueError(f"Provider {provider.name} did not return required feature: {feature_name}")
        
        # Get the feature value
        value = provider_features[feature_name]
        
        # Update cache
        self._cache.set(token_id, feature_name, value)
        logger.debug(f"Computed and cached {feature_name}={value} for {token_id} from provider {provider.name}")
        
        return value
    
    def _compute_from_feature(self, token_id: str, feature_name: str, context: Dict[str, Any]) -> Any:
        """
        Compute a feature from a registered Feature instance.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            context: Context data for computation
            
        Returns:
            Computed feature value
            
        Raises:
            ValueError: If computation fails
        """
        feature = self._features[feature_name]
        
        # Ensure dependencies are available in context
        for dep_name in feature.dependencies:
            if dep_name not in context:
                try:
                    dep_value = self.get_feature(token_id, dep_name, use_cache=True)
                    context[dep_name] = dep_value
                    logger.debug(f"Added dependency {dep_name}={dep_value} for {feature_name}")
                except Exception as e:
                    logger.warning(f"Failed to get dependency {dep_name} for {feature_name}: {e}")
        
        # Compute the feature
        try:
            value = feature.compute(context)
        except Exception as e:
            raise ValueError(f"Feature {feature_name} computation failed: {e}")
        
        # Update cache
        self._cache.set(token_id, feature_name, value)
        logger.debug(f"Computed and cached {feature_name}={value} for {token_id} from feature object")
        
        return value
    
    def update_feature(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Update a feature value.
        
        This method updates the cache and notifies all registered consumers.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            value: New feature value
        """
        if not token_id or not feature_name:
            logger.warning(f"Cannot update feature with empty token_id or feature_name")
            return
        
        # Update cache
        self._cache.set(token_id, feature_name, value)
        
        # Invalidate dependent features
        self._invalidate_dependent_features(token_id, feature_name)
        
        # Notify consumers
        self._notify_consumers(token_id, feature_name, value)
        
        logger.debug(f"Updated feature {feature_name}={value} for {token_id}")
    
    def _invalidate_dependent_features(self, token_id: str, feature_name: str) -> None:
        """
        Invalidate dependent features when a feature is updated.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature that was updated
        """
        with self._dependency_lock:
            if feature_name not in self._dependents:
                return
            
            # Get all features that depend on this one (directly or indirectly)
            to_invalidate = set()
            to_process = list(self._dependents[feature_name])
            
            # Traverse dependency graph to find all affected features
            while to_process:
                dep_feature = to_process.pop(0)
                if dep_feature not in to_invalidate:
                    to_invalidate.add(dep_feature)
                    
                    # Add features that depend on this dependent
                    if dep_feature in self._dependents:
                        to_process.extend(self._dependents[dep_feature])
            
            # Invalidate each dependent feature
            for dep_feature in to_invalidate:
                self._cache.invalidate(token_id, dep_feature)
                logger.debug(f"Invalidated dependent feature {dep_feature} for {token_id}")
    
    def _notify_consumers(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Notify consumers of a feature update.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            value: New feature value
        """
        with self._consumer_lock:
            if feature_name in self._consumers:
                consumers = list(self._consumers[feature_name])
                
                # Notify each consumer outside the lock
                for consumer in consumers:
                    try:
                        consumer.on_feature_update(token_id, feature_name, value)
                    except Exception as e:
                        logger.error(f"Error notifying consumer of feature update: {e}", exc_info=True)
        
        # Publish event
        if self._event_bus:
            try:
                self._event_bus.publish(Event(
                    event_type=EventType.FEATURE_UPDATE,
                    data={
                        'token_id': token_id,
                        'feature_name': feature_name,
                        'value': value,
                        'timestamp': time.time()
                    }
                ))
                logger.debug(f"Published FEATURE_UPDATE event for {feature_name} of {token_id}")
            except Exception as e:
                logger.error(f"Error publishing feature update event: {e}", exc_info=True)
    
    def get_available_features(self) -> List[str]:
        """
        Get all available features.
        
        Returns:
            List[str]: List of all available feature names
        """
        with self._provider_lock:
            all_features = set()
            
            # Add features from providers
            all_features.update(self._feature_to_provider.keys())
            
            # Add directly registered features
            all_features.update(self._features.keys())
            
            return sorted(list(all_features))
    
    def get_features_for_token(self, token_id: str, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get multiple feature values for a token.
        
        Args:
            token_id: ID of the token
            features: Optional list of feature names to get.
                     If None, get all available features.
            
        Returns:
            Dict[str, Any]: Dictionary mapping feature names to values
        """
        if not token_id:
            logger.warning("Cannot get features for empty token_id")
            return {}
        
        # If no specific features requested, get all available
        if features is None:
            features = self.get_available_features()
        
        result = {}
        errors = []
        
        # Get each feature
        for feature_name in features:
            try:
                value = self.get_feature(token_id, feature_name)
                result[feature_name] = value
            except Exception as e:
                errors.append(f"{feature_name}: {str(e)}")
                logger.warning(f"Error getting feature {feature_name} for {token_id}: {e}")
        
        if errors and not result:
            logger.error(f"Failed to get any requested features for {token_id}. Errors: {'; '.join(errors)}")
        
        return result
    
    def _get_computing_event(self, token_id: str, feature_name: str) -> threading.Event:
        """
        Get or create a computing event for a token and feature.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
        
        Returns:
            threading.Event: Event that is set when computation is in progress
        """
        with self._computing_lock:
            if token_id not in self._computing:
                self._computing[token_id] = {}
            
            if feature_name not in self._computing[token_id]:
                self._computing[token_id][feature_name] = threading.Event()
            
            return self._computing[token_id][feature_name]
    
    def _clear_computing_event(self, token_id: str, feature_name: str) -> None:
        """
        Clear a computing event for a token and feature.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
        """
        with self._computing_lock:
            if token_id in self._computing and feature_name in self._computing[token_id]:
                self._computing[token_id][feature_name].clear()
    
    def compute_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all requested or available features for the given context.
        
        Args:
            context: Context information for feature computation.
                    Should contain 'token_id' at minimum.
            
        Returns:
            Dictionary mapping feature names to computed values
        """
        start_time = time.time()
        
        # Check if token_id is specified in context
        token_id = context.get('token_id')
        if not token_id:
            logger.warning("No token_id specified in context. Cannot compute features.")
            return {}
        
        # Determine which features to compute
        requested_features = context.get('features')
        features_to_compute = requested_features if requested_features else self.get_available_features()
        
        # Get all requested features
        result = self.get_features_for_token(token_id, features_to_compute)
        
        # Record performance
        elapsed = time.time() - start_time
        logger.debug(f"Computed {len(result)} features in {elapsed:.4f} seconds")
        
        return result
    
    def get_feature_dependencies(self, feature_name: str) -> List[str]:
        """
        Get the dependencies for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            List[str]: List of feature dependencies
        """
        with self._dependency_lock:
            if feature_name not in self._dependencies:
                return []
            
            return sorted(list(self._dependencies[feature_name]))
    
    def get_feature_dependents(self, feature_name: str) -> List[str]:
        """
        Get the features that depend on a feature.
        
        Args:
            feature_name: Name of the feature
        
        Returns:
            List[str]: List of dependent feature names
        """
        with self._dependency_lock:
            if feature_name not in self._dependents:
                return []
            
            return sorted(list(self._dependents[feature_name]))
    
    def clear_cache(self) -> None:
        """
        Clear the entire feature cache.
        """
        invalidated = self._cache.invalidate()
        logger.info(f"Cleared feature cache, invalidated {invalidated} entries")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the feature system.
        
        Returns:
            Dict[str, Any]: Dictionary of metrics
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
            
            # Calculate average computation times
            avg_times = {}
            for feature_name, times in metrics["computation_times"].items():
                if times:
                    avg_times[feature_name] = sum(times) / len(times)
            
            metrics["avg_computation_times"] = avg_times
            
            # Get cache stats
            metrics["cache_stats"] = self._cache.get_stats()
            
            # Calculate uptime
            metrics["uptime_seconds"] = time.time() - metrics["last_reset_time"]
            
            return metrics
    
    def reset_metrics(self) -> None:
        """
        Reset performance metrics for the feature system.
        """
        with self._metrics_lock:
            self._metrics = {
                "feature_computations": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "computation_times": {},
                "computation_errors": 0,
                "last_reset_time": time.time()
            }
        logger.info("Reset feature system metrics")
    
    def get_provider_for_feature(self, feature_name: str) -> Optional[str]:
        """
        Get the name of the provider that provides a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            str: Name of the provider, or None if the feature is not provided by any provider
        """
        with self._provider_lock:
            if feature_name in self._feature_to_provider:
                return self._feature_to_provider[feature_name].name
            return None
    
    def get_provider(self, provider_name: str) -> Optional[FeatureProvider]:
        """
        Get a provider by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            FeatureProvider: The provider, or None if not found
        """
        with self._provider_lock:
            return self._providers.get(provider_name) 