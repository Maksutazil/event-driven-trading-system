#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Registry Module

This module provides a central registry for feature names to ensure consistent naming
across all components of the system, including feature providers, transformers, and ML models.

The registry maintains standard feature names, aliases, descriptions, and metadata to 
facilitate feature discovery and cross-component compatibility.
"""

import logging
import json
import os
from typing import Dict, List, Set, Any, Optional, Callable
from threading import RLock

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Central registry for feature names and metadata.
    
    This class maintains a mapping of standard feature names, allowing components
    to reference features consistently. It supports feature aliases, descriptions,
    and grouping to improve discoverability and compatibility.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the feature registry.
        
        Args:
            config_path: Optional path to a JSON configuration file with predefined features
        """
        # Core registry data
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}
        self._groups: Dict[str, Set[str]] = {}
        self._providers: Dict[str, Set[str]] = {}
        self._consumers: Dict[str, Set[str]] = {}
        
        # Thread safety
        self._lock = RLock()
        
        # Load predefined features if config path is provided
        if config_path and os.path.exists(config_path):
            self.load_from_config(config_path)
            
        # Initialize with core features if registry is empty
        if not self._registry:
            self.initialize_default_features()
            
        logger.info(f"Feature registry initialized with {len(self._registry)} features")
    
    def register_feature(self, 
                        name: str, 
                        description: str, 
                        group: str = 'general', 
                        aliases: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a new feature in the registry.
        
        Args:
            name: Standard name for the feature
            description: Description of what the feature represents
            group: Group the feature belongs to (e.g., 'price', 'volume', 'indicator')
            aliases: Alternative names for the feature
            metadata: Additional metadata about the feature
            
        Returns:
            True if feature was registered, False if it already exists
        """
        with self._lock:
            # Check if feature already exists
            if name in self._registry:
                logger.warning(f"Feature '{name}' already registered")
                return False
                
            # Add feature to registry
            self._registry[name] = {
                'description': description,
                'group': group,
                'aliases': aliases or [],
                'metadata': metadata or {}
            }
            
            # Add to group
            if group not in self._groups:
                self._groups[group] = set()
            self._groups[group].add(name)
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases:
                        logger.warning(f"Alias '{alias}' already registered for feature '{self._aliases[alias]}'")
                    else:
                        self._aliases[alias] = name
                        
            logger.debug(f"Registered feature '{name}' in group '{group}'")
            return True
    
    def get_standard_name(self, feature_name: str) -> str:
        """
        Get the standard name for a feature, resolving aliases.
        
        Args:
            feature_name: Feature name or alias
            
        Returns:
            Standard feature name
            
        Raises:
            KeyError: If the feature name or alias is not registered
        """
        with self._lock:
            # If it's a standard name, return it
            if feature_name in self._registry:
                return feature_name
                
            # If it's an alias, return the standard name
            if feature_name in self._aliases:
                return self._aliases[feature_name]
                
            # Not found
            raise KeyError(f"Feature '{feature_name}' not found in registry")
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """
        Get information about a feature.
        
        Args:
            feature_name: Feature name or alias
            
        Returns:
            Feature information dictionary
            
        Raises:
            KeyError: If the feature name or alias is not registered
        """
        with self._lock:
            standard_name = self.get_standard_name(feature_name)
            info = dict(self._registry[standard_name])
            info['name'] = standard_name
            return info
    
    def list_features(self, group: Optional[str] = None) -> List[str]:
        """
        List registered features, optionally filtered by group.
        
        Args:
            group: Optional group to filter by
            
        Returns:
            List of standard feature names
        """
        with self._lock:
            if group:
                return list(self._groups.get(group, set()))
            return list(self._registry.keys())
    
    def list_groups(self) -> List[str]:
        """
        List all feature groups.
        
        Returns:
            List of group names
        """
        with self._lock:
            return list(self._groups.keys())
    
    def register_provider(self, provider_name: str, features: List[str]) -> None:
        """
        Register a feature provider and the features it provides.
        
        Args:
            provider_name: Name of the feature provider
            features: List of features (standard names or aliases) the provider provides
        """
        with self._lock:
            standard_features = set()
            for feature in features:
                try:
                    standard_features.add(self.get_standard_name(feature))
                except KeyError:
                    logger.warning(f"Provider '{provider_name}' provides unregistered feature '{feature}'")
            
            self._providers[provider_name] = standard_features
            logger.debug(f"Registered provider '{provider_name}' for {len(standard_features)} features")
    
    def register_consumer(self, consumer_name: str, features: List[str]) -> None:
        """
        Register a feature consumer and the features it requires.
        
        Args:
            consumer_name: Name of the feature consumer
            features: List of features (standard names or aliases) the consumer requires
        """
        with self._lock:
            standard_features = set()
            for feature in features:
                try:
                    standard_features.add(self.get_standard_name(feature))
                except KeyError:
                    logger.warning(f"Consumer '{consumer_name}' requires unregistered feature '{feature}'")
            
            self._consumers[consumer_name] = standard_features
            logger.debug(f"Registered consumer '{consumer_name}' for {len(standard_features)} features")
    
    def get_providers_for_feature(self, feature_name: str) -> List[str]:
        """
        Get providers that provide a specific feature.
        
        Args:
            feature_name: Feature name or alias
            
        Returns:
            List of provider names
        """
        with self._lock:
            try:
                standard_name = self.get_standard_name(feature_name)
                return [p for p, features in self._providers.items() if standard_name in features]
            except KeyError:
                return []
    
    def get_consumers_for_feature(self, feature_name: str) -> List[str]:
        """
        Get consumers that require a specific feature.
        
        Args:
            feature_name: Feature name or alias
            
        Returns:
            List of consumer names
        """
        with self._lock:
            try:
                standard_name = self.get_standard_name(feature_name)
                return [c for c, features in self._consumers.items() if standard_name in features]
            except KeyError:
                return []
    
    def transform_feature_dict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a dictionary of features to use standard names.
        
        Args:
            features: Dictionary with feature names as keys
            
        Returns:
            New dictionary with standardized feature names
        """
        with self._lock:
            result = {}
            for name, value in features.items():
                try:
                    standard_name = self.get_standard_name(name)
                    result[standard_name] = value
                except KeyError:
                    # Keep the original name if not registered
                    result[name] = value
            return result
    
    def load_from_config(self, config_path: str) -> int:
        """
        Load features from a JSON configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Number of features loaded
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            count = 0
            features = config.get('features', [])
            for feature in features:
                success = self.register_feature(
                    name=feature['name'],
                    description=feature.get('description', ''),
                    group=feature.get('group', 'general'),
                    aliases=feature.get('aliases', []),
                    metadata=feature.get('metadata', {})
                )
                if success:
                    count += 1
            
            logger.info(f"Loaded {count} features from {config_path}")
            return count
        except Exception as e:
            logger.error(f"Error loading feature registry config from {config_path}: {e}")
            return 0
    
    def save_to_config(self, config_path: str) -> bool:
        """
        Save the registry to a JSON configuration file.
        
        Args:
            config_path: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                features = []
                for name, info in self._registry.items():
                    feature = {
                        'name': name,
                        'description': info['description'],
                        'group': info['group'],
                        'aliases': info['aliases'],
                        'metadata': info['metadata']
                    }
                    features.append(feature)
            
            with open(config_path, 'w') as f:
                json.dump({'features': features}, f, indent=2)
                
            logger.info(f"Saved {len(features)} features to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving feature registry config to {config_path}: {e}")
            return False
    
    def initialize_default_features(self) -> None:
        """Initialize the registry with default core features."""
        # Price features
        self.register_feature(
            name="current_price",
            description="Current market price of the token",
            group="price",
            aliases=["price", "latest_price", "market_price"]
        )
        
        self.register_feature(
            name="price_change_pct_5m",
            description="Percentage price change over the last 5 minutes",
            group="price",
            aliases=["price_change_5m", "pct_change_5m"]
        )
        
        self.register_feature(
            name="price_change_pct_15m",
            description="Percentage price change over the last 15 minutes",
            group="price",
            aliases=["price_change_15m", "pct_change_15m"]
        )
        
        # Volume features
        self.register_feature(
            name="volume_5m",
            description="Trading volume over the last 5 minutes",
            group="volume",
            aliases=["recent_volume", "short_term_volume"]
        )
        
        self.register_feature(
            name="volume_15m",
            description="Trading volume over the last 15 minutes",
            group="volume",
            aliases=["medium_term_volume"]
        )
        
        # Moving average features
        self.register_feature(
            name="ma_5m",
            description="Moving average price over 5 minutes",
            group="indicator",
            aliases=["moving_average_5m", "short_ma"]
        )
        
        self.register_feature(
            name="ma_15m",
            description="Moving average price over 15 minutes",
            group="indicator",
            aliases=["moving_average_15m", "medium_ma"]
        )
        
        # Technical indicators
        self.register_feature(
            name="rsi_14",
            description="Relative Strength Index with period 14",
            group="indicator",
            aliases=["rsi", "relative_strength_index"]
        )
        
        self.register_feature(
            name="macd_histogram",
            description="MACD histogram value",
            group="indicator",
            aliases=["macd_hist", "macd_bar"]
        )
        
        self.register_feature(
            name="volatility_5m",
            description="Price volatility over 5 minutes",
            group="indicator",
            aliases=["short_term_volatility", "price_volatility"]
        )
        
        # Signal features
        self.register_feature(
            name="price_momentum_signal",
            description="Trading signal based on price momentum",
            group="signal",
            aliases=["momentum_signal", "price_signal"]
        )
        
        self.register_feature(
            name="volume_spike_signal",
            description="Trading signal based on volume spikes",
            group="signal",
            aliases=["volume_signal", "spike_signal"]
        )
        
        # Model prediction features
        self.register_feature(
            name="model_prediction",
            description="Machine learning model prediction",
            group="ml",
            aliases=["prediction", "ml_prediction"]
        )
        
        self.register_feature(
            name="prediction_confidence",
            description="Confidence score of model prediction",
            group="ml",
            aliases=["model_confidence", "confidence_score"]
        )
        
        logger.info("Initialized registry with default features")
    
    def get_all_features_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information for all registered features.
        
        Returns:
            Dictionary of feature information, keyed by standard name
        """
        with self._lock:
            result = {}
            for name in self._registry:
                result[name] = self.get_feature_info(name)
            return result
    
    def validate_feature_dict(self, features: Dict[str, Any]) -> List[str]:
        """
        Validate a dictionary of features against the registry.
        
        Args:
            features: Dictionary with feature names as keys
            
        Returns:
            List of unregistered feature names
        """
        with self._lock:
            unregistered = []
            for name in features:
                try:
                    self.get_standard_name(name)
                except KeyError:
                    unregistered.append(name)
            return unregistered
    
    def validate_feature_list(self, features: List[str]) -> List[str]:
        """
        Validate a list of features against the registry.
        
        Args:
            features: List of feature names
            
        Returns:
            List of unregistered feature names
        """
        with self._lock:
            unregistered = []
            for name in features:
                try:
                    self.get_standard_name(name)
                except KeyError:
                    unregistered.append(name)
            return unregistered 