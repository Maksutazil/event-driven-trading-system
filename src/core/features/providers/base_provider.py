#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Feature Provider Module

This module provides the BaseFeatureProvider class, which serves as a foundation
for implementing feature providers.
"""

import logging
from typing import Dict, List, Any, Optional, Set

from ..interfaces import FeatureProvider

logger = logging.getLogger(__name__)


class BaseFeatureProvider(FeatureProvider):
    """
    Base class for feature providers.
    
    Feature providers are responsible for computing features from raw data
    and making them available to the feature system.
    """
    
    def __init__(self, name: str):
        """
        Initialize the feature provider.
        
        Args:
            name: Name of the feature provider
        """
        self._name = name
        self._provides: Set[str] = set()
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @property
    def name(self) -> str:
        """
        Get the name of the feature provider.
        
        Returns:
            Name of the provider
        """
        return self._name
    
    @property
    def provides(self) -> Set[str]:
        """
        Get the set of feature names this provider can compute.
        
        Returns:
            Set of feature names
        """
        return self._provides
    
    def get_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute and return features for the given context.
        
        This method should be overridden by subclasses to provide
        specific feature computation logic.
        
        Args:
            context: Context data for computation
            
        Returns:
            Dictionary of computed feature values
        """
        # This base implementation doesn't compute any features
        return {} 