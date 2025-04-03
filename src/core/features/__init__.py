"""
Features Module

This module provides functionality for computing and managing features
used by the trading engine for decision-making.

Core components:
- FeatureSystem: Central system for managing and computing features
- Feature: Base interface for individual features
- FeatureProvider: Base interface for feature providers
- FeatureCache: Caching mechanism for feature computations
- DefaultFeatureManager: Default implementation of the feature manager
- FeatureManager: Interface for feature management
- FeatureRegistry: Central registry for feature names and metadata
"""

from .interfaces import Feature, FeatureProvider, FeatureManager
from .feature_system import FeatureSystem
from .cache import FeatureCache, InMemoryFeatureCache
from .providers import BaseFeatureProvider, PriceFeatureProvider
from .signal_feature import PriceMomentumSignalFeature, VolumeSpikeTradingSignalFeature
from .manager import DefaultFeatureManager
from .registry import FeatureRegistry

__all__ = [
    'Feature',
    'FeatureProvider',
    'FeatureManager',
    'FeatureSystem',
    'FeatureCache',
    'InMemoryFeatureCache',
    'BaseFeatureProvider',
    'PriceFeatureProvider',
    'PriceMomentumSignalFeature',
    'VolumeSpikeTradingSignalFeature',
    'DefaultFeatureManager',
    'FeatureRegistry',
] 