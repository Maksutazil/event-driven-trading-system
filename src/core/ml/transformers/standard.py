#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standard Feature Transformer Module

This module provides a standard feature transformer for preparing
features for machine learning models.
"""

import logging
import numpy as np
import pickle
import os
from typing import Dict, List, Any, Optional, Union, Set, Tuple

from ..interfaces import FeatureTransformer

logger = logging.getLogger(__name__)


class StandardFeatureTransformer(FeatureTransformer):
    """
    Standard feature transformer for preparing model inputs.
    
    This transformer:
    1. Selects specific features from the input
    2. Standardizes numerical features
    3. Handles missing values
    4. Can one-hot encode categorical features
    """
    
    def __init__(self, feature_list: List[str], 
                 categorical_features: Optional[List[str]] = None,
                 standardize: bool = True,
                 handle_missing: bool = True):
        """
        Initialize the standard feature transformer.
        
        Args:
            feature_list: List of feature names to include
            categorical_features: List of categorical feature names
            standardize: Whether to standardize numerical features
            handle_missing: Whether to handle missing values
        """
        self._feature_list = feature_list
        self._categorical_features = set(categorical_features or [])
        self._standardize = standardize
        self._handle_missing = handle_missing
        
        # Statistics for standardization
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}
        
        # Categorical feature values
        self._categorical_values: Dict[str, List[Any]] = {}
        
        # Flag to track if the transformer has been fitted
        self._is_fitted = False
        
        logger.info(f"Initialized StandardFeatureTransformer with {len(feature_list)} features")
    
    def transform(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw features to model inputs.
        
        Args:
            features: Raw features
            
        Returns:
            Transformed features suitable for model input
        """
        try:
            result = {"features": []}
            
            # Copy token_id if present
            if "token_id" in features:
                result["token_id"] = features["token_id"]
            
            # Check if transformer is fitted
            if not self._is_fitted:
                logger.warning("Transformer is not fitted. Using raw features.")
                for feature_name in self._feature_list:
                    if feature_name in features:
                        try:
                            result["features"].append(float(features[feature_name]))
                        except (ValueError, TypeError):
                            result["features"].append(0.0)
                    else:
                        result["features"].append(0.0)
                return result
            
            # Process each feature
            for feature_name in self._feature_list:
                if feature_name not in features:
                    # Handle missing feature
                    if feature_name in self._categorical_features:
                        # Use one-hot encoding with all zeros
                        values = self._categorical_values.get(feature_name, [])
                        result["features"].extend([0.0] * len(values))
                    else:
                        # Use mean for numerical features if handling missing values
                        if self._handle_missing:
                            value = self._means.get(feature_name, 0.0)
                            if self._standardize:
                                # Mean becomes 0 after standardization
                                value = 0.0
                        else:
                            value = 0.0
                        result["features"].append(value)
                else:
                    # Process existing feature
                    if feature_name in self._categorical_features:
                        # One-hot encode categorical feature
                        feat_value = features[feature_name]
                        values = self._categorical_values.get(feature_name, [])
                        if not values:
                            # No known values, treat as numerical
                            try:
                                result["features"].append(float(feat_value))
                            except (ValueError, TypeError):
                                result["features"].append(0.0)
                        else:
                            # One-hot encode
                            one_hot = [1.0 if val == feat_value else 0.0 for val in values]
                            result["features"].extend(one_hot)
                    else:
                        # Standardize numerical feature
                        try:
                            value = float(features[feature_name])
                            if self._standardize and feature_name in self._means:
                                std = self._stds.get(feature_name, 1.0)
                                if std > 1e-10:  # Avoid division by zero
                                    value = (value - self._means.get(feature_name, 0.0)) / std
                            result["features"].append(value)
                        except (ValueError, TypeError):
                            # Non-numeric value for numerical feature
                            if self._handle_missing:
                                value = self._means.get(feature_name, 0.0)
                                if self._standardize:
                                    value = 0.0
                            else:
                                value = 0.0
                            result["features"].append(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transforming features: {e}", exc_info=True)
            # Return empty features as fallback
            return {"features": [0.0] * len(self._feature_list), "token_id": features.get("token_id")}
    
    def fit(self, features_list: List[Dict[str, Any]]) -> bool:
        """
        Fit the transformer to a batch of features.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Whether the fit was successful
        """
        try:
            if not features_list:
                logger.warning("Cannot fit transformer with empty feature list")
                return False
            
            # Collect all values for each feature
            all_values: Dict[str, List[Any]] = {}
            for feature_name in self._feature_list:
                all_values[feature_name] = []
            
            # Extract values from feature dictionaries
            for features in features_list:
                for feature_name in self._feature_list:
                    if feature_name in features:
                        value = features[feature_name]
                        if feature_name in self._categorical_features:
                            all_values[feature_name].append(value)
                        else:
                            try:
                                all_values[feature_name].append(float(value))
                            except (ValueError, TypeError):
                                # Skip non-numeric values for numerical features
                                pass
            
            # Calculate statistics for numerical features
            for feature_name in self._feature_list:
                if feature_name not in self._categorical_features:
                    values = all_values[feature_name]
                    if values:
                        self._means[feature_name] = float(np.mean(values))
                        self._stds[feature_name] = float(np.std(values))
                        if self._stds[feature_name] < 1e-10:  # Avoid division by zero
                            self._stds[feature_name] = 1.0
                    else:
                        self._means[feature_name] = 0.0
                        self._stds[feature_name] = 1.0
                else:
                    # Get unique values for categorical features
                    self._categorical_values[feature_name] = list(set(all_values[feature_name]))
            
            self._is_fitted = True
            logger.info(f"Fitted transformer with {len(features_list)} examples")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting transformer: {e}", exc_info=True)
            return False
    
    def get_required_features(self) -> List[str]:
        """
        Get the list of features required by this transformer.
        
        Returns:
            List of feature names
        """
        return self._feature_list.copy()
    
    def save(self, file_path: str) -> bool:
        """
        Save the transformer to a file.
        
        Args:
            file_path: Path to save the transformer
            
        Returns:
            Whether the save was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare state for serialization
            state = {
                "feature_list": self._feature_list,
                "categorical_features": list(self._categorical_features),
                "standardize": self._standardize,
                "handle_missing": self._handle_missing,
                "means": self._means,
                "stds": self._stds,
                "categorical_values": self._categorical_values,
                "is_fitted": self._is_fitted
            }
            
            # Save the state
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Saved transformer to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving transformer: {e}", exc_info=True)
            return False
    
    @classmethod
    def load(cls, file_path: str) -> Optional['StandardFeatureTransformer']:
        """
        Load a transformer from a file.
        
        Args:
            file_path: Path to the transformer file
            
        Returns:
            Loaded transformer or None if loading fails
        """
        try:
            # Load the state
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
            
            # Create a new instance
            transformer = cls(
                feature_list=state["feature_list"],
                categorical_features=state["categorical_features"],
                standardize=state["standardize"],
                handle_missing=state["handle_missing"]
            )
            
            # Restore the state
            transformer._means = state["means"]
            transformer._stds = state["stds"]
            transformer._categorical_values = state["categorical_values"]
            transformer._is_fitted = state["is_fitted"]
            
            logger.info(f"Loaded transformer from {file_path}")
            return transformer
        except Exception as e:
            logger.error(f"Error loading transformer: {e}", exc_info=True)
            return None 