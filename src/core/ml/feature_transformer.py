#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Transformer Module

This module provides functionality for transforming raw features into
formats suitable for machine learning models, leveraging the feature
registry for standardized naming across the system.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
import pandas as pd
from datetime import datetime

from ..features.registry import FeatureRegistry
from ..features.interfaces import FeatureManager

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """
    Transforms raw features into formats suitable for machine learning models.
    
    This class handles feature scaling, normalization, encoding, and selection
    to prepare data for model training and prediction. It uses the feature
    registry to ensure consistent feature naming across components.
    """
    
    def __init__(self, 
                 feature_manager: FeatureManager,
                 registry: Optional[FeatureRegistry] = None,
                 required_features: Optional[List[str]] = None):
        """
        Initialize the feature transformer.
        
        Args:
            feature_manager: Feature manager instance
            registry: Optional feature registry for name standardization
            required_features: List of features required by models
        """
        self._feature_manager = feature_manager
        self._registry = registry or feature_manager.registry
        
        # Features required by the model
        self._required_features = required_features or []
        
        # Feature transformation parameters
        self._scalers: Dict[str, Dict[str, float]] = {}  # feature_name -> {min, max, mean, std}
        self._encoders: Dict[str, Dict[str, int]] = {}  # feature_name -> {category -> index}
        
        # Feature group configuration
        self._feature_groups: Dict[str, List[str]] = {}
        
        logger.info(f"FeatureTransformer initialized with {len(self._required_features)} required features")
    
    @property
    def required_features(self) -> List[str]:
        """Get the list of required features."""
        return self._required_features
    
    def set_required_features(self, features: List[str]) -> None:
        """
        Set the list of required features.
        
        Args:
            features: List of feature names required by models
        """
        # Validate feature names with the registry
        self._required_features = []
        for feature in features:
            try:
                std_name = self._registry.get_standard_name(feature)
                self._required_features.append(std_name)
            except KeyError:
                logger.warning(f"Feature '{feature}' not found in registry, using as-is")
                self._required_features.append(feature)
        
        logger.info(f"Set {len(self._required_features)} required features")
    
    def add_feature_group(self, group_name: str, features: List[str]) -> None:
        """
        Add a feature group configuration.
        
        Args:
            group_name: Name of the feature group
            features: List of features in the group
        """
        # Standardize feature names
        std_features = []
        for feature in features:
            try:
                std_name = self._registry.get_standard_name(feature)
                std_features.append(std_name)
            except KeyError:
                logger.warning(f"Feature '{feature}' not found in registry, using as-is")
                std_features.append(feature)
        
        self._feature_groups[group_name] = std_features
        logger.info(f"Added feature group '{group_name}' with {len(std_features)} features")
    
    def get_feature_group(self, group_name: str) -> List[str]:
        """
        Get the features in a feature group.
        
        Args:
            group_name: Name of the feature group
            
        Returns:
            List of feature names in the group
        """
        return self._feature_groups.get(group_name, [])
    
    def configure_scaler(self, feature_name: str, min_val: float, max_val: float, 
                        mean: float, std: float) -> None:
        """
        Configure scaling parameters for a numeric feature.
        
        Args:
            feature_name: Feature name
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
            mean: Mean value for standardization
            std: Standard deviation for standardization
        """
        try:
            std_name = self._registry.get_standard_name(feature_name)
        except KeyError:
            logger.warning(f"Feature '{feature_name}' not found in registry, using as-is")
            std_name = feature_name
            
        self._scalers[std_name] = {
            'min': min_val,
            'max': max_val,
            'mean': mean,
            'std': std
        }
        logger.debug(f"Configured scaler for '{std_name}'")
    
    def configure_encoder(self, feature_name: str, categories: Dict[str, int]) -> None:
        """
        Configure encoding parameters for a categorical feature.
        
        Args:
            feature_name: Feature name
            categories: Dictionary mapping categories to indices
        """
        try:
            std_name = self._registry.get_standard_name(feature_name)
        except KeyError:
            logger.warning(f"Feature '{feature_name}' not found in registry, using as-is")
            std_name = feature_name
            
        self._encoders[std_name] = categories
        logger.debug(f"Configured encoder for '{std_name}' with {len(categories)} categories")
    
    def get_features_for_token(self, token_id: str) -> Dict[str, Any]:
        """
        Get required features for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dictionary of feature values
        """
        # Get features from the feature manager
        features = self._feature_manager.get_features_for_token(token_id, self._required_features)
        
        # Transform feature names to standardized names
        std_features = {}
        for name, value in features.items():
            try:
                std_name = self._registry.get_standard_name(name)
                std_features[std_name] = value
            except KeyError:
                # Feature not in registry, use original name
                std_features[name] = value
        
        # Check for missing required features
        missing = [f for f in self._required_features if f not in std_features]
        if missing:
            logger.warning(f"Missing required features for token {token_id}: {missing}")
        
        return std_features
    
    def transform_features(self, features: Dict[str, Any], normalize: bool = True) -> Dict[str, Any]:
        """
        Transform features for model input.
        
        Args:
            features: Dictionary of raw feature values
            normalize: Whether to normalize numeric features
            
        Returns:
            Dictionary of transformed feature values
        """
        result = {}
        
        # Transform each feature
        for name, value in features.items():
            try:
                # Get standardized name if possible
                try:
                    std_name = self._registry.get_standard_name(name)
                except KeyError:
                    std_name = name
                
                # Transform based on feature type
                if std_name in self._scalers and normalize:
                    # Numeric feature with scaling configuration
                    result[std_name] = self._normalize_feature(std_name, value)
                elif std_name in self._encoders and isinstance(value, str):
                    # Categorical feature with encoding configuration
                    result[std_name] = self._encode_feature(std_name, value)
                else:
                    # No transformation needed or available
                    result[std_name] = value
            except Exception as e:
                logger.error(f"Error transforming feature '{name}': {e}", exc_info=True)
                # Keep original value on error
                result[name] = value
        
        return result
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """
        Normalize a numeric feature.
        
        Args:
            feature_name: Feature name
            value: Raw feature value
            
        Returns:
            Normalized feature value
        """
        if feature_name not in self._scalers:
            return value
            
        scaler = self._scalers[feature_name]
        min_val = scaler['min']
        max_val = scaler['max']
        
        # Avoid division by zero
        if max_val == min_val:
            return 0.0
            
        # Min-max normalization to [0, 1]
        return (value - min_val) / (max_val - min_val)
    
    def _standardize_feature(self, feature_name: str, value: float) -> float:
        """
        Standardize a numeric feature (z-score).
        
        Args:
            feature_name: Feature name
            value: Raw feature value
            
        Returns:
            Standardized feature value
        """
        if feature_name not in self._scalers:
            return value
            
        scaler = self._scalers[feature_name]
        mean = scaler['mean']
        std = scaler['std']
        
        # Avoid division by zero
        if std == 0:
            return 0.0
            
        # Z-score standardization
        return (value - mean) / std
    
    def _encode_feature(self, feature_name: str, value: str) -> int:
        """
        Encode a categorical feature.
        
        Args:
            feature_name: Feature name
            value: Category value
            
        Returns:
            Encoded feature value
        """
        if feature_name not in self._encoders:
            return 0
            
        encoder = self._encoders[feature_name]
        return encoder.get(value, 0)
    
    def prepare_model_input(self, token_id: str, normalize: bool = True) -> np.ndarray:
        """
        Prepare model input for a token.
        
        Args:
            token_id: ID of the token
            normalize: Whether to normalize numeric features
            
        Returns:
            NumPy array of feature values, ready for model input
        """
        # Get and transform features
        raw_features = self.get_features_for_token(token_id)
        transformed = self.transform_features(raw_features, normalize)
        
        # Create input array with required features in correct order
        features = []
        for feature_name in self._required_features:
            if feature_name in transformed:
                features.append(transformed[feature_name])
            else:
                # Use default value for missing features
                features.append(0.0)
                logger.warning(f"Missing required feature '{feature_name}' for token {token_id}")
        
        return np.array(features, dtype=np.float32)
    
    def prepare_batch_input(self, token_ids: List[str], normalize: bool = True) -> np.ndarray:
        """
        Prepare model input for multiple tokens.
        
        Args:
            token_ids: List of token IDs
            normalize: Whether to normalize numeric features
            
        Returns:
            2D NumPy array of feature values, ready for model input
        """
        batch_features = []
        
        for token_id in token_ids:
            features = self.prepare_model_input(token_id, normalize)
            batch_features.append(features)
        
        return np.array(batch_features, dtype=np.float32)
    
    def create_feature_dataframe(self, token_ids: List[str], normalize: bool = False) -> pd.DataFrame:
        """
        Create a pandas DataFrame with features for multiple tokens.
        
        Args:
            token_ids: List of token IDs
            normalize: Whether to normalize numeric features
            
        Returns:
            DataFrame with token features
        """
        data = []
        
        for token_id in token_ids:
            # Get and transform features
            raw_features = self.get_features_for_token(token_id)
            transformed = self.transform_features(raw_features, normalize)
            
            # Add token ID
            transformed['token_id'] = token_id
            data.append(transformed)
        
        if not data:
            # Create empty DataFrame with correct columns
            columns = ['token_id'] + self._required_features
            return pd.DataFrame(columns=columns)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure all required features are included
        for feature in self._required_features:
            if feature not in df.columns:
                df[feature] = np.nan
        
        return df
    
    def save_configuration(self) -> Dict[str, Any]:
        """
        Save transformer configuration for later restoration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'required_features': self._required_features,
            'scalers': self._scalers,
            'encoders': self._encoders,
            'feature_groups': self._feature_groups
        }
    
    def load_configuration(self, config: Dict[str, Any]) -> None:
        """
        Load transformer configuration.
        
        Args:
            config: Configuration dictionary
        """
        if 'required_features' in config:
            self._required_features = config['required_features']
        
        if 'scalers' in config:
            self._scalers = config['scalers']
        
        if 'encoders' in config:
            self._encoders = config['encoders']
        
        if 'feature_groups' in config:
            self._feature_groups = config['feature_groups']
        
        logger.info(f"Loaded transformer configuration with {len(self._required_features)} required features") 