#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pump and Dump Detection Model

This module provides a model implementation for detecting pump and dump events
in token trading using both standard and early detection features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta
import xgboost as xgb
import os

from src.core.ml.interfaces import Model

logger = logging.getLogger(__name__)


class PumpDumpDetectionModel(Model):
    """
    Model for detecting pump and dump events in token trading.
    
    This model combines features from both PumpDetectionFeatureProvider and
    EarlyPumpDetectionProvider to predict pump and dump events at various stages.
    It uses XGBoost for training on historical data.
    """
    
    def __init__(self):
        """
        Initialize the pump dump detection model.
        """
        self._model_id = "pump_dump_detection_model"
        self._model_type = "classification"
        self._model_version = "1.0.0"
        self._clf = None  # XGBoost classifier
        
        # Default required features
        self._required_features = [
            'price_velocity',
            'volume_surge_ratio',
            'buy_sell_ratio',
            'price_oscillation',
            'abnormal_activity_score',
            'immediate_price_change',
            'trade_frequency',
            'buyer_dominance',
            'volume_intensity',
            'early_pump_score'
        ]
        
        # Default parameters for model
        self._parameters = {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1,
            'class_weight': 'balanced'
        }
        
        logger.info(f"Initialized PumpDumpDetectionModel")
    
    @property
    def model_id(self) -> str:
        """
        Get the model identifier.
        
        Returns:
            Model ID
        """
        return self._model_id
    
    @property
    def model_type(self) -> str:
        """
        Get the type of this model.
        
        Returns:
            Model type identifier
        """
        return self._model_type
    
    @property
    def model_version(self) -> str:
        """
        Get the version of this model.
        
        Returns:
            Model version identifier
        """
        return self._model_version
    
    def get_required_features(self) -> List[str]:
        """
        Get the features required for prediction.
        
        Returns:
            List of required feature names
        """
        return self._required_features
    
    def set_required_features(self, features: List[str]) -> None:
        """
        Set the required features for this model.
        
        Args:
            features: List of feature names
        """
        self._required_features = features
        logger.info(f"Updated required features: {features}")
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the parameters for training the model.
        
        Args:
            parameters: Dictionary of model parameters
        """
        self._parameters = parameters
        logger.info(f"Updated model parameters: {parameters}")
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction using the trained model.
        
        Args:
            features: Features to use for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if not self._clf:
            logger.warning("Model not trained yet, using fallback prediction")
            return self._fallback_prediction(features)
        
        try:
            # Extract features needed for prediction
            feature_values = []
            for feature in self._required_features:
                if feature in features:
                    feature_values.append(features[feature])
                else:
                    logger.warning(f"Feature {feature} missing, using 0")
                    feature_values.append(0.0)
            
            # Convert to numpy array for prediction
            X = np.array([feature_values])
            feature_names = self._required_features
            dtest = xgb.DMatrix(X, feature_names=feature_names)
            
            # Get raw predictions
            raw_preds = self._clf.predict(dtest)
            probas = self._clf.predict_proba(dtest)[0]
            
            # Map predictions to class labels
            class_id = int(raw_preds[0])
            class_labels = {
                0: "NORMAL",
                1: "PUMP",
                2: "DUMP",
            }
            class_label = class_labels.get(class_id, "UNKNOWN")
            
            # Calculate confidence based on probability
            confidence = probas[class_id]
            
            # Return prediction results
            return {
                "class_id": class_id,
                "class_label": class_label,
                "probabilities": {
                    "NORMAL": float(probas[0]),
                    "PUMP": float(probas[1] if len(probas) > 1 else 0),
                    "DUMP": float(probas[2] if len(probas) > 2 else 0),
                },
                "confidence": float(confidence),
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return self._fallback_prediction(features)
    
    def _fallback_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback prediction when the model is not trained.
        Uses rule-based logic similar to the PumpPredictorModel.
        
        Args:
            features: Features for prediction
        
        Returns:
            Dictionary with prediction results
        """
        # Extract relevant features with fallbacks
        abnormal_score = features.get('abnormal_activity_score', 0.0)
        early_pump_score = features.get('early_pump_score', 0.0)
        price_velocity = features.get('price_velocity', 0.0)
        
        # Simple rule-based classification
        if early_pump_score > 0.7 or price_velocity > 5.0:
            class_id = 1  # PUMP
            confidence = max(early_pump_score, min(1.0, price_velocity / 10.0))
        elif price_velocity < -5.0:
            class_id = 2  # DUMP
            confidence = min(1.0, abs(price_velocity) / 10.0)
        else:
            class_id = 0  # NORMAL
            confidence = 1.0 - abnormal_score
        
        # Map class IDs to labels
        class_labels = {
            0: "NORMAL",
            1: "PUMP",
            2: "DUMP",
        }
        
        # Return prediction results
        return {
            "class_id": class_id,
            "class_label": class_labels[class_id],
            "probabilities": {
                "NORMAL": 0.8 if class_id == 0 else 0.1,
                "PUMP": 0.8 if class_id == 1 else 0.1,
                "DUMP": 0.8 if class_id == 2 else 0.1,
            },
            "confidence": float(confidence),
            "using_fallback": True
        }
    
    def train(self, training_data: pd.DataFrame) -> None:
        """
        Train the model with historical data.
        
        Args:
            training_data: DataFrame with feature data and labels
        """
        if training_data.empty:
            logger.warning("No training data provided, cannot train model")
            return
        
        try:
            logger.info(f"Training model with {len(training_data)} samples")
            
            # Prepare features and target
            X = training_data.drop(['class_id', 'timestamp', 'token_id'], axis=1, errors='ignore')
            y = training_data['class_id']
            
            # Keep track of actual feature names used
            self._required_features = list(X.columns)
            
            # Create and train XGBoost classifier
            params = {
                'max_depth': self._parameters.get('max_depth', 10),
                'eta': self._parameters.get('learning_rate', 0.1),
                'objective': 'multi:softprob',
                'num_class': 3,  # NORMAL, PUMP, DUMP
                'eval_metric': 'mlogloss',
                'scale_pos_weight': self._parameters.get('scale_pos_weight', 1.0)
            }
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
            
            # Train model
            num_rounds = self._parameters.get('n_estimators', 100)
            self._clf = xgb.train(params, dtrain, num_rounds)
            
            logger.info(f"Model training completed successfully with {num_rounds} rounds")
            logger.info(f"Features used: {self._required_features}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
    
    def evaluate(self, eval_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on evaluation data.
        
        Args:
            eval_data: DataFrame with feature data and true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if eval_data.empty:
            logger.warning("No evaluation data provided")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        if not self._clf:
            logger.warning("Model not trained, cannot evaluate")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            # Prepare features and target
            X = eval_data.drop(['class_id', 'timestamp', 'token_id'], axis=1, errors='ignore')
            y_true = eval_data['class_id'].values
            
            # Create DMatrix for prediction
            dtest = xgb.DMatrix(X, feature_names=list(X.columns))
            
            # Make predictions
            y_pred_proba = self._clf.predict(dtest)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            correct = np.sum(y_pred == y_true)
            accuracy = correct / len(y_true)
            
            # Class-specific metrics
            classes = [0, 1, 2]  # NORMAL, PUMP, DUMP
            precision = {}
            recall = {}
            f1 = {}
            
            for cls in classes:
                # True positives, false positives, false negatives
                tp = np.sum((y_pred == cls) & (y_true == cls))
                fp = np.sum((y_pred == cls) & (y_true != cls))
                fn = np.sum((y_pred != cls) & (y_true == cls))
                
                # Calculate metrics (avoid division by zero)
                if tp + fp > 0:
                    precision[cls] = tp / (tp + fp)
                else:
                    precision[cls] = 0.0
                    
                if tp + fn > 0:
                    recall[cls] = tp / (tp + fn)
                else:
                    recall[cls] = 0.0
                    
                if precision[cls] + recall[cls] > 0:
                    f1[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls])
                else:
                    f1[cls] = 0.0
            
            # Macro-average of metrics
            macro_precision = sum(precision.values()) / len(precision)
            macro_recall = sum(recall.values()) / len(recall)
            macro_f1 = sum(f1.values()) / len(f1)
            
            return {
                "accuracy": float(accuracy),
                "precision": float(macro_precision),
                "recall": float(macro_recall),
                "f1": float(macro_f1),
                "precision_normal": float(precision[0]),
                "recall_normal": float(recall[0]),
                "f1_normal": float(f1[0]),
                "precision_pump": float(precision[1]),
                "recall_pump": float(recall[1]),
                "f1_pump": float(f1[1]),
                "precision_dump": float(precision[2]),
                "recall_dump": float(recall[2]),
                "f1_dump": float(f1[2]),
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}", exc_info=True)
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(e)}

    def save_model(self, model_path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path where to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if not self._clf:
            logger.warning("No trained model to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save XGBoost model
            self._clf.save_model(model_path)
            logger.info(f"Saved XGBoost model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}", exc_info=True)
            return False 