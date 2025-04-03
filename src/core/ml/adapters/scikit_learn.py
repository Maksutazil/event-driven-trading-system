#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scikit-Learn Model Adapter Module

This module provides adapters for scikit-learn machine learning models.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
import joblib

from ..interfaces import Model

logger = logging.getLogger(__name__)


class ScikitLearnModel(Model):
    """
    Adapter for scikit-learn models.
    
    This class wraps a scikit-learn model, implementing the Model interface.
    """
    
    def __init__(self, model_id: str, model_object, model_type: str = "classification"):
        """
        Initialize a scikit-learn model adapter.
        
        Args:
            model_id: Identifier for the model
            model_object: The scikit-learn model object
            model_type: Type of the model (default: "classification")
        """
        self._model_id = model_id
        self._model_object = model_object
        self._model_type = model_type
        self._created_time = time.time()
        self._metadata = {
            "framework": "scikit-learn",
            "type": model_type,
            "class": model_object.__class__.__name__,
            "created": self._created_time,
            "updated": self._created_time
        }
        
        logger.info(f"Initialized scikit-learn model adapter for {model_id}")
    
    @property
    def model_id(self) -> str:
        """Get the model identifier."""
        return self._model_id
    
    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self._model_type
    
    def predict(self, features: Dict[str, Any]) -> Any:
        """
        Make a prediction using the model.
        
        Args:
            features: Features to use for prediction
            
        Returns:
            Model prediction
        """
        try:
            # Convert features to numpy array
            X = self._dict_to_array(features)
            
            # Make prediction
            if self._model_type == "classification" and hasattr(self._model_object, "predict_proba"):
                # For classifiers, get probability estimates
                proba = self._model_object.predict_proba(X)
                class_idx = int(self._model_object.predict(X)[0])
                
                # Get class labels if available
                if hasattr(self._model_object, "classes_"):
                    classes = self._model_object.classes_.tolist()
                    prediction = {
                        "class_id": class_idx,
                        "class_label": classes[class_idx] if class_idx < len(classes) else str(class_idx),
                        "probabilities": {str(cls): float(p) for cls, p in zip(classes, proba[0])}
                    }
                else:
                    prediction = {
                        "class_id": class_idx,
                        "probabilities": {str(i): float(p) for i, p in enumerate(proba[0])}
                    }
            else:
                # For regressors, get point estimates
                pred_value = self._model_object.predict(X)[0]
                if isinstance(pred_value, np.ndarray):
                    prediction = float(pred_value[0])
                else:
                    prediction = float(pred_value)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction with model {self.model_id}: {e}", exc_info=True)
            return None
    
    def update(self, features: Dict[str, Any], target: Any) -> bool:
        """
        Update the model with new data.
        
        Args:
            features: Features for the new data point
            target: Target value for the new data point
            
        Returns:
            Whether the update was successful
        """
        try:
            # Convert features to numpy array
            X = self._dict_to_array(features)
            
            # Convert target to appropriate format
            if isinstance(target, list):
                y = np.array(target)
            else:
                y = np.array([target])
            
            # Check if model has partial_fit method (for online learning)
            if hasattr(self._model_object, "partial_fit"):
                self._model_object.partial_fit(X, y)
                
                # Update metadata
                self._metadata["updated"] = time.time()
                
                logger.info(f"Updated model {self.model_id} with new data")
                return True
            else:
                logger.warning(f"Model {self.model_id} does not support online learning")
                return False
                
        except Exception as e:
            logger.error(f"Error updating model {self.model_id}: {e}", exc_info=True)
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the model.
        
        Returns:
            Dictionary of model metadata
        """
        return self._metadata.copy()
    
    def save(self, file_path: str) -> bool:
        """
        Save the model to a file.
        
        Args:
            file_path: Path to save the model
            
        Returns:
            Whether the save was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the model
            joblib.dump(self._model_object, file_path)
            
            logger.info(f"Saved model {self.model_id} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {self.model_id}: {e}", exc_info=True)
            return False
    
    def _dict_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert a feature dictionary to a numpy array.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Numpy array ready for scikit-learn
        """
        # Extract features as an array
        if "features" in features and isinstance(features["features"], list):
            # Already in array format
            X = np.array(features["features"]).reshape(1, -1)
        else:
            # Need to extract features
            # This assumes features are already ordered correctly
            feature_values = [v for k, v in features.items() 
                              if k != "token_id" and k != "timestamp" and isinstance(v, (int, float))]
            X = np.array(feature_values).reshape(1, -1)
        
        return X


class ScikitLearnModelAdapter:
    """
    Adapter for loading scikit-learn models.
    """
    
    def load(self, model_id: str, model_path: str, model_type: str = None) -> Optional[Model]:
        """
        Load a scikit-learn model from a file.
        
        Args:
            model_id: Identifier for the model
            model_path: Path to the model file
            model_type: Optional type of the model (will be detected if not provided)
            
        Returns:
            ScikitLearnModel instance or None if loading fails
        """
        try:
            # Load the model using joblib
            model_object = joblib.load(model_path)
            
            # Determine model type if not provided
            if model_type is None:
                model_type = self._determine_model_type(model_object)
            
            # Create and return the adapter
            return ScikitLearnModel(model_id, model_object, model_type)
            
        except Exception as e:
            logger.error(f"Error loading scikit-learn model {model_id}: {e}", exc_info=True)
            return None
    
    def _determine_model_type(self, model_object) -> str:
        """
        Determine the type of the scikit-learn model.
        
        Args:
            model_object: The scikit-learn model object
            
        Returns:
            Model type as a string
        """
        # Check if model is a classifier or regressor
        if hasattr(model_object, "predict_proba"):
            return "classification"
        else:
            return "regression" 