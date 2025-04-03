#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Feature Provider Module

This module provides the BaseFeatureProvider class that serves as the foundation
for all feature providers in the system.
"""

import logging
from abc import abstractmethod
from typing import Dict, List, Any, Optional

from src.core.features.interfaces import FeatureProvider

logger = logging.getLogger(__name__)


class BaseFeatureProvider(FeatureProvider):
    """
    Base class for all feature providers.
    
    This class implements common functionality for feature providers and serves
    as a foundation for specialized feature providers.
    """
    
    def __init__(self, feature_names: List[str], dependencies: Dict[str, List[str]] = None):
        """
        Initialize the feature provider.
        
        Args:
            feature_names: List of features provided by this provider
            dependencies: Dictionary mapping features to their dependencies
        """
        self.feature_names = feature_names
        self.dependencies = dependencies or {}
        logger.debug(f"Initialized {self.__class__.__name__} with {len(feature_names)} features")
    
    def get_provided_features(self) -> List[str]:
        """
        Get the list of features provided by this provider.
        
        Returns:
            List of feature names
        """
        return self.feature_names
    
    def get_dependencies(self, feature_name: str) -> List[str]:
        """
        Get the dependencies for the specified feature.
        
        Args:
            feature_name: The name of the feature
            
        Returns:
            List of dependency feature names
        """
        if feature_name not in self.feature_names:
            logger.warning(f"Feature '{feature_name}' is not provided by {self.__class__.__name__}")
            return []
            
        return self.dependencies.get(feature_name, [])
    
    def get_required_data_types(self, feature_name: str) -> List[str]:
        """
        Get the required data types for computing the specified feature.
        
        Args:
            feature_name: The name of the feature
            
        Returns:
            List of required data type names
        """
        # Default implementation - override in specialized providers if needed
        return ["trade_data"]
    
    @abstractmethod
    def compute_feature(self, feature_name: str, token_id: str, data: Dict[str, Any]) -> Any:
        """
        Compute the specified feature for the given token using the provided data.
        
        Args:
            feature_name: The name of the feature to compute
            token_id: The ID of the token
            data: Dictionary containing required data for computation
            
        Returns:
            The computed feature value
        """
        pass 