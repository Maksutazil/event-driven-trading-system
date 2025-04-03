#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Manager Module

This module provides the implementation of the FeatureManager interface,
which is responsible for managing features across the system.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Type

from ..events import EventBus, Event, EventType
from .interfaces import FeatureManager, FeatureProvider, FeatureConsumer, Feature
from .cache import InMemoryFeatureCache
from .registry import FeatureRegistry

logger = logging.getLogger(__name__)


class DefaultFeatureManager(FeatureManager):
    """
    Default implementation of the FeatureManager interface.
    
    This class manages feature providers, consumers, and computation, with
    support for caching and standardized feature naming.
    """
    
    def __init__(self, 
                event_bus: Optional[EventBus] = None,
                feature_registry: Optional[FeatureRegistry] = None,
                registry_config_path: Optional[str] = None):
        """
        Initialize the feature manager.
        
        Args:
            event_bus: Optional event bus for publishing feature updates
            feature_registry: Optional feature registry for standardized naming
            registry_config_path: Optional path to feature registry config
        """
        self._event_bus = event_bus
        self._providers: Dict[str, FeatureProvider] = {}
        self._consumers: Dict[str, FeatureConsumer] = {}
        self._features: Dict[str, Feature] = {}
        self._feature_to_provider: Dict[str, str] = {}
        self._cache = InMemoryFeatureCache()
        
        # Initialize or use provided feature registry
        if feature_registry:
            self._registry = feature_registry
        else:
            self._registry = FeatureRegistry(config_path=registry_config_path)
        
        # Performance metrics
        self._metrics = {
            'computation_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'provider_errors': 0,
            'feature_errors': 0,
            'total_compute_time': 0.0
        }
        
        logger.info("DefaultFeatureManager initialized")
    
    @property
    def registry(self) -> FeatureRegistry:
        """Get the feature registry."""
        return self._registry
    
    def register_provider(self, provider: FeatureProvider) -> None:
        """
        Register a feature provider.
        
        Args:
            provider: The feature provider to register
        """
        name = provider.name
        if name in self._providers:
            logger.warning(f"Provider '{name}' already registered, replacing")
        
        self._providers[name] = provider
        
        # Register with feature registry
        self._registry.register_provider(name, list(provider.provides))
        
        # Update feature-to-provider mapping
        for feature_name in provider.provides:
            try:
                std_name = self._registry.get_standard_name(feature_name)
                self._feature_to_provider[std_name] = name
            except KeyError:
                # Feature not in registry, use original name
                self._feature_to_provider[feature_name] = name
                logger.warning(f"Feature '{feature_name}' not found in registry but provided by '{name}'")
        
        logger.info(f"Registered provider '{name}' with {len(provider.provides)} features")
    
    def unregister_provider(self, provider: FeatureProvider) -> None:
        """
        Unregister a feature provider.
        
        Args:
            provider: The feature provider to unregister
        """
        name = provider.name
        if name not in self._providers:
            logger.warning(f"Provider '{name}' not registered")
            return
        
        # Remove feature-to-provider mappings
        provider_features = self._providers[name].provides
        for feature_name in provider_features:
            try:
                std_name = self._registry.get_standard_name(feature_name)
                if std_name in self._feature_to_provider and self._feature_to_provider[std_name] == name:
                    del self._feature_to_provider[std_name]
            except KeyError:
                # Feature not in registry, check original name
                if feature_name in self._feature_to_provider and self._feature_to_provider[feature_name] == name:
                    del self._feature_to_provider[feature_name]
        
        # Remove provider
        del self._providers[name]
        logger.info(f"Unregistered provider '{name}'")
    
    def register_consumer(self, consumer: FeatureConsumer) -> None:
        """
        Register a feature consumer.
        
        Args:
            consumer: The feature consumer to register
        """
        name = consumer.__class__.__name__
        if name in self._consumers:
            logger.warning(f"Consumer '{name}' already registered, replacing")
        
        self._consumers[name] = consumer
        
        # Register with feature registry
        self._registry.register_consumer(name, consumer.get_required_features())
        
        logger.info(f"Registered consumer '{name}' requiring {len(consumer.get_required_features())} features")
    
    def unregister_consumer(self, consumer: FeatureConsumer) -> None:
        """
        Unregister a feature consumer.
        
        Args:
            consumer: The feature consumer to unregister
        """
        name = consumer.__class__.__name__
        if name not in self._consumers:
            logger.warning(f"Consumer '{name}' not registered")
            return
        
        del self._consumers[name]
        logger.info(f"Unregistered consumer '{name}'")
    
    def register_feature(self, feature: Feature) -> None:
        """
        Register a feature in the manager.
        
        Args:
            feature: The feature to register
        """
        name = feature.name
        if name in self._features:
            logger.warning(f"Feature '{name}' already registered, replacing")
        
        self._features[name] = feature
        
        # Register with feature registry if not already registered
        try:
            self._registry.get_standard_name(name)
        except KeyError:
            logger.info(f"Adding feature '{name}' to registry")
            self._registry.register_feature(
                name=name,
                description=f"Feature provided by {feature.__class__.__name__}",
                group="computed",
                aliases=[]
            )
        
        logger.info(f"Registered feature '{name}' with dependencies: {feature.dependencies}")
    
    def get_feature(self, token_id: str, feature_name: str, use_cache: bool = True) -> Any:
        """
        Get a feature value for a token.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            use_cache: Whether to use cached values
            
        Returns:
            The feature value
        """
        start_time = time.time()
        
        try:
            # Standardize feature name
            try:
                std_feature_name = self._registry.get_standard_name(feature_name)
            except KeyError:
                logger.warning(f"Feature '{feature_name}' not found in registry, using as-is")
                std_feature_name = feature_name
            
            # Check cache if enabled
            if use_cache:
                cached_value = self._cache.get(token_id, std_feature_name)
                if cached_value is not None:
                    self._metrics['cache_hits'] += 1
                    logger.debug(f"Cache hit for {token_id}/{std_feature_name}")
                    return cached_value
                
                self._metrics['cache_misses'] += 1
            
            # Compute the feature
            value = self.compute_feature(token_id, std_feature_name)
            
            # Cache if successful
            if value is not None:
                self._cache.set(token_id, std_feature_name, value, datetime.now())
            
            # Publish feature update event if event bus is available
            if self._event_bus and value is not None:
                self._publish_feature_update(token_id, std_feature_name, value)
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting feature {feature_name} for token {token_id}: {e}", exc_info=True)
            self._metrics['feature_errors'] += 1
            return None
        finally:
            self._metrics['total_compute_time'] += time.time() - start_time
    
    def compute_feature(self, token_id: str, feature_name: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Compute a feature value for a token.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            context: Optional context data
            
        Returns:
            The computed feature value
        """
        start_time = time.time()
        self._metrics['computation_count'] += 1
        
        try:
            # Standardize feature name
            try:
                std_feature_name = self._registry.get_standard_name(feature_name)
            except KeyError:
                logger.warning(f"Feature '{feature_name}' not found in registry, using as-is")
                std_feature_name = feature_name
            
            # Initialize or use provided context
            if context is None:
                context = {'token_id': token_id}
            elif 'token_id' not in context:
                context['token_id'] = token_id
            
            # Try to compute from a feature instance
            if std_feature_name in self._features:
                feature = self._features[std_feature_name]
                
                # Ensure all dependencies are in the context
                for dep in feature.dependencies:
                    if dep not in context:
                        # Try to get dependency from cache or compute it
                        dep_value = self.get_feature(token_id, dep)
                        if dep_value is not None:
                            context[dep] = dep_value
                        else:
                            logger.warning(f"Could not resolve dependency '{dep}' for feature '{std_feature_name}'")
                
                # Compute the feature
                return feature.compute(context)
            
            # Try to compute from a provider
            if std_feature_name in self._feature_to_provider:
                provider_name = self._feature_to_provider[std_feature_name]
                provider = self._providers[provider_name]
                
                # Get all features from the provider
                features = provider.get_features(context)
                
                # Standardize feature names in the result
                std_features = self._registry.transform_feature_dict(features)
                
                # Return the requested feature if available
                if std_feature_name in std_features:
                    return std_features[std_feature_name]
                else:
                    logger.warning(f"Provider '{provider_name}' did not return feature '{std_feature_name}'")
            
            # Try original name with providers if standardization failed
            if feature_name != std_feature_name and feature_name in self._feature_to_provider:
                provider_name = self._feature_to_provider[feature_name]
                provider = self._providers[provider_name]
                
                # Get all features from the provider
                features = provider.get_features(context)
                
                # Return the requested feature if available
                if feature_name in features:
                    return features[feature_name]
            
            logger.warning(f"No provider found for feature '{std_feature_name}'")
            return None
            
        except Exception as e:
            logger.error(f"Error computing feature {feature_name} for token {token_id}: {e}", exc_info=True)
            self._metrics['feature_errors'] += 1
            return None
        finally:
            self._metrics['total_compute_time'] += time.time() - start_time
    
    def update_feature(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Update a feature value in the cache.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            value: New feature value
        """
        try:
            # Standardize feature name
            try:
                std_feature_name = self._registry.get_standard_name(feature_name)
            except KeyError:
                logger.warning(f"Feature '{feature_name}' not found in registry, using as-is")
                std_feature_name = feature_name
            
            # Update cache
            self._cache.set(token_id, std_feature_name, value, datetime.now())
            
            # Notify consumers
            self._notify_consumers(token_id, std_feature_name, value)
            
            # Publish feature update event if event bus is available
            if self._event_bus:
                self._publish_feature_update(token_id, std_feature_name, value)
                
            logger.debug(f"Updated feature {std_feature_name} for token {token_id}")
        except Exception as e:
            logger.error(f"Error updating feature {feature_name} for token {token_id}: {e}", exc_info=True)
    
    def get_available_features(self) -> List[str]:
        """
        Get a list of all available features.
        
        Returns:
            List of feature names
        """
        features = set()
        
        # Add features from registry
        features.update(self._registry.list_features())
        
        # Add features from providers
        for provider in self._providers.values():
            features.update(provider.provides)
        
        # Add computed features
        features.update(self._features.keys())
        
        return list(features)
    
    def get_features_for_token(self, token_id: str, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get multiple features for a token.
        
        Args:
            token_id: ID of the token
            features: Optional list of features to get (all if None)
            
        Returns:
            Dictionary of feature values
        """
        result = {}
        
        # Use all available features if none specified
        if features is None:
            features = self.get_available_features()
        
        # Standardize feature names
        std_features = []
        for feature in features:
            try:
                std_features.append(self._registry.get_standard_name(feature))
            except KeyError:
                logger.warning(f"Feature '{feature}' not found in registry, using as-is")
                std_features.append(feature)
        
        # Get each feature
        for feature in std_features:
            value = self.get_feature(token_id, feature)
            if value is not None:
                result[feature] = value
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        logger.info("Feature cache cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = dict(self._metrics)
        metrics['cache_size'] = self._cache.get_stats()['entry_count']
        return metrics
    
    def get_providers(self) -> List[FeatureProvider]:
        """
        Get a list of all registered feature providers.
        
        Returns:
            List of feature providers
        """
        return list(self._providers.values())
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = {
            'computation_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'provider_errors': 0,
            'feature_errors': 0,
            'total_compute_time': 0.0
        }
        logger.info("Feature metrics reset")
    
    def _notify_consumers(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Notify consumers of a feature update.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            value: New feature value
        """
        for consumer in self._consumers.values():
            try:
                required_features = consumer.get_required_features()
                if feature_name in required_features:
                    consumer.on_feature_update(token_id, feature_name, value)
            except Exception as e:
                logger.error(f"Error notifying consumer {consumer.__class__.__name__}: {e}", exc_info=True)
    
    def _publish_feature_update(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Publish a feature update event.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            value: Feature value
        """
        if not self._event_bus:
            return
            
        event_data = {
            'token_id': token_id,
            'feature_name': feature_name,
            'value': value,
            'timestamp': time.time()
        }
        
        try:
            self._event_bus.publish(Event(
                event_type=EventType.FEATURE_UPDATE,
                data=event_data,
                source="FeatureManager",
                token_id=token_id
            ))
        except Exception as e:
            logger.error(f"Error publishing feature update event: {e}", exc_info=True)
    
    def validate_feature_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency of features across providers and the registry.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'unregistered_provider_features': [],
            'unregistered_consumer_features': [],
            'unregistered_computed_features': [],
            'provider_conflicts': [],
            'total_features': len(self._registry.list_features()),
            'is_consistent': True
        }
        
        # Check providers
        for provider_name, provider in self._providers.items():
            unregistered = self._registry.validate_feature_list(list(provider.provides))
            if unregistered:
                results['unregistered_provider_features'].extend([(provider_name, f) for f in unregistered])
                results['is_consistent'] = False
        
        # Check consumers
        for consumer_name, consumer in self._consumers.items():
            unregistered = self._registry.validate_feature_list(consumer.get_required_features())
            if unregistered:
                results['unregistered_consumer_features'].extend([(consumer_name, f) for f in unregistered])
                results['is_consistent'] = False
        
        # Check computed features
        for feature_name, feature in self._features.items():
            try:
                self._registry.get_standard_name(feature_name)
            except KeyError:
                results['unregistered_computed_features'].append(feature_name)
                results['is_consistent'] = False
        
        # Check provider conflicts
        feature_providers: Dict[str, List[str]] = {}
        for provider_name, provider in self._providers.items():
            for feature_name in provider.provides:
                try:
                    std_name = self._registry.get_standard_name(feature_name)
                    if std_name not in feature_providers:
                        feature_providers[std_name] = []
                    feature_providers[std_name].append(provider_name)
                except KeyError:
                    pass  # Already reported as unregistered
        
        for feature_name, providers in feature_providers.items():
            if len(providers) > 1:
                results['provider_conflicts'].append((feature_name, providers))
                results['is_consistent'] = False
        
        return results 