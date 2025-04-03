#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Interfaces

This module defines the interfaces for features, feature providers, 
feature consumers, and feature management in the system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class Feature(ABC):
    """
    Interface for computed features.
    
    A feature represents a computed value that may depend on other features.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the standard name of the feature.
        
        Returns:
            The feature name
        """
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """
        Get the names of feature dependencies.
        
        Returns:
            List of feature names that this feature depends on
        """
        pass
    
    @abstractmethod
    def compute(self, context: Dict[str, Any]) -> Any:
        """
        Compute the feature value.
        
        Args:
            context: Dictionary containing dependencies and other context data
            
        Returns:
            Computed feature value
        """
        pass


class FeatureProvider(ABC):
    """
    Interface for feature providers.
    
    A feature provider supplies values for one or more features.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            Provider name
        """
        pass
    
    @property
    @abstractmethod
    def provides(self) -> Set[str]:
        """
        Get the set of features provided.
        
        Returns:
            Set of feature names provided by this provider
        """
        pass
    
    @abstractmethod
    def get_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get values for all provided features.
        
        Args:
            context: Dictionary containing context data, such as token_id
            
        Returns:
            Dictionary mapping feature names to values
        """
        pass


class FeatureConsumer(ABC):
    """
    Interface for feature consumers.
    
    A feature consumer processes feature values when they are updated.
    """
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Get the features required by this consumer.
        
        Returns:
            List of required feature names
        """
        pass
    
    @abstractmethod
    def on_feature_update(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Handle a feature update.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the updated feature
            value: New feature value
        """
        pass


class FeatureManager(ABC):
    """
    Interface for feature managers.
    
    A feature manager coordinates feature providers, consumers, and computation.
    """
    
    @abstractmethod
    def register_provider(self, provider: FeatureProvider) -> None:
        """
        Register a feature provider.
        
        Args:
            provider: The feature provider to register
        """
        pass
    
    @abstractmethod
    def unregister_provider(self, provider: FeatureProvider) -> None:
        """
        Unregister a feature provider.
        
        Args:
            provider: The feature provider to unregister
        """
        pass
    
    @abstractmethod
    def register_consumer(self, consumer: FeatureConsumer) -> None:
        """
        Register a feature consumer.
        
        Args:
            consumer: The feature consumer to register
        """
        pass
    
    @abstractmethod
    def unregister_consumer(self, consumer: FeatureConsumer) -> None:
        """
        Unregister a feature consumer.
        
        Args:
            consumer: The feature consumer to unregister
        """
        pass
    
    @abstractmethod
    def register_feature(self, feature: Feature) -> None:
        """
        Register a feature in the manager.
        
        Args:
            feature: The feature to register
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def update_feature(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Update a feature value in the cache.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            value: New feature value
        """
        pass
    
    @abstractmethod
    def get_available_features(self) -> List[str]:
        """
        Get a list of all available features.
        
        Returns:
            List of feature names
        """
        pass
    
    @abstractmethod
    def get_features_for_token(self, token_id: str, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get multiple features for a token.
        
        Args:
            token_id: ID of the token
            features: Optional list of features to get (all if None)
            
        Returns:
            Dictionary of feature values
        """
        pass
    
    @abstractmethod
    def clear_cache(self) -> None:
        """
        Clear the feature cache.
        """
        pass


class FeatureCache(ABC):
    """
    Interface for feature caches.
    
    A feature cache stores feature values to avoid repeated computation.
    """
    
    @abstractmethod
    def get(self, token_id: str, feature_name: str) -> Any:
        """
        Get a cached feature value.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            
        Returns:
            Cached value, or None if not in cache
        """
        pass
    
    @abstractmethod
    def set(self, token_id: str, feature_name: str, value: Any, timestamp: Optional[datetime] = None) -> None:
        """
        Set a feature value in the cache.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            value: Feature value to cache
            timestamp: Optional timestamp
        """
        pass
    
    @abstractmethod
    def invalidate(self, token_id: str, feature_name: Optional[str] = None) -> None:
        """
        Invalidate a feature or all features for a token.
        
        Args:
            token_id: ID of the token
            feature_name: Optional name of the feature to invalidate.
                         If None, invalidate all features for the token.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear the cache.
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        pass 