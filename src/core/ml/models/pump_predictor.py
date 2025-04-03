#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pump Predictor Model

This module provides a model implementation for predicting pump and dump events
in token trading activity.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from src.core.ml.interfaces import Model
from src.core.features.manager import FeatureManager

logger = logging.getLogger(__name__)


class PumpPredictorModel(Model):
    """
    Model for predicting pump and dump events.
    
    This model uses features from the PumpDetectionFeatureProvider to predict
    whether a token is currently in a pump or dump phase, or about to enter one.
    """
    
    def __init__(self, model_id: str, feature_manager: FeatureManager):
        """
        Initialize the pump predictor model.
        
        Args:
            model_id: Identifier for this model
            feature_manager: Feature manager for retrieving features
        """
        self._model_id = model_id
        self._feature_manager = feature_manager
        self._model_type = "classification"
        self._model_version = "1.0.0"
        
        # Define required features
        self._required_features = [
            'price_velocity',
            'price_acceleration',
            'volume_surge_ratio',
            'volume_volatility',
            'buy_sell_volume_ratio',
            'price_deviation',
            'price_volatility_ratio',
            'pump_pattern_score',
            'dump_pattern_score',
            'abnormal_activity_score',
        ]
        
        # Thresholds for prediction
        self._pump_threshold = 0.7     # Score above which we predict a pump
        self._dump_threshold = 0.7     # Score above which we predict a dump
        self._neutral_threshold = 0.3  # Score below which we predict normal activity
        
        logger.info(f"Initialized PumpPredictorModel with ID: {model_id}")
    
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
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a pump/dump prediction based on the provided features.
        
        Args:
            features: Features to use for prediction
            
        Returns:
            Dictionary with prediction results:
            - class_id: 0 (normal), 1 (pump), 2 (dump), 3 (peak/distribution)
            - class_label: String label for the class
            - probabilities: Class probabilities
            - phase: Numeric pump phase (0-4)
            - confidence: Confidence score for the prediction
        """
        try:
            # Extract relevant features
            pump_score = features.get('pump_pattern_score', 0.0)
            dump_score = features.get('dump_pattern_score', 0.0)
            abnormal_score = features.get('abnormal_activity_score', 0.0)
            
            # Get pump phase if available, or calculate from scores
            phase = features.get('pump_phase_detection')
            if phase is None:
                if pump_score < self._neutral_threshold and dump_score < self._neutral_threshold:
                    phase = 0  # No pump/dump
                elif pump_score > self._pump_threshold and dump_score < self._neutral_threshold:
                    phase = 2  # Pump in progress
                elif dump_score > self._dump_threshold:
                    phase = 4  # Dump in progress
                elif pump_score > 0.3 and dump_score < 0.3:
                    phase = 1  # Early accumulation
                else:
                    phase = 3  # Peak/distribution
            
            # Determine prediction class
            # 0 = Normal activity
            # 1 = Pump in progress
            # 2 = Dump in progress
            # 3 = Peak/distribution phase
            
            class_id = 0  # Default to normal activity
            if phase == 0:
                class_id = 0  # Normal
            elif phase in [1, 2]:
                class_id = 1  # Pump
            elif phase == 4:
                class_id = 2  # Dump
            else:  # phase == 3
                class_id = 3  # Peak/distribution
            
            # Calculate prediction confidence
            confidence = 0.0
            if class_id == 0:
                # Confidence is inverse of abnormal score
                confidence = 1.0 - abnormal_score
            elif class_id == 1:
                # Confidence is pump score
                confidence = pump_score
            elif class_id == 2:
                # Confidence is dump score
                confidence = dump_score
            else:  # class_id == 3
                # Peak confidence is based on both pump and dump being moderate
                confidence = 1.0 - abs(pump_score - dump_score)
            
            # Map class IDs to labels
            class_labels = {
                0: "NORMAL",
                1: "PUMP",
                2: "DUMP",
                3: "PEAK",
            }
            
            # Calculate probabilities for each class
            if class_id == 0:
                # Normal state
                proba_normal = confidence
                proba_pump = pump_score * (1 - proba_normal)
                proba_dump = dump_score * (1 - proba_normal)
                proba_peak = max(0, 1 - proba_normal - proba_pump - proba_dump)
            elif class_id == 1:
                # Pump state
                proba_pump = confidence
                proba_normal = (1 - abnormal_score) * (1 - proba_pump)
                proba_dump = 0.1 * (1 - proba_pump - proba_normal)
                proba_peak = max(0, 1 - proba_normal - proba_pump - proba_dump)
            elif class_id == 2:
                # Dump state
                proba_dump = confidence
                proba_normal = (1 - abnormal_score) * (1 - proba_dump)
                proba_pump = 0.1 * (1 - proba_dump - proba_normal)
                proba_peak = max(0, 1 - proba_normal - proba_pump - proba_dump)
            else:  # class_id == 3
                # Peak/distribution state
                proba_peak = confidence
                proba_pump = pump_score * (1 - proba_peak)
                proba_dump = dump_score * (1 - proba_peak)
                proba_normal = max(0, 1 - proba_peak - proba_pump - proba_dump)
            
            # Normalize probabilities to ensure they sum to 1
            total_proba = proba_normal + proba_pump + proba_dump + proba_peak
            if total_proba > 0:
                proba_normal /= total_proba
                proba_pump /= total_proba
                proba_dump /= total_proba
                proba_peak /= total_proba
            
            # Return prediction results
            return {
                "class_id": class_id,
                "class_label": class_labels[class_id],
                "probabilities": {
                    "NORMAL": float(proba_normal),
                    "PUMP": float(proba_pump),
                    "DUMP": float(proba_dump),
                    "PEAK": float(proba_peak),
                },
                "phase": int(phase),
                "confidence": float(confidence),
            }
            
        except Exception as e:
            logger.error(f"Error making pump prediction: {e}")
            return {
                "class_id": 0,
                "class_label": "NORMAL",
                "probabilities": {"NORMAL": 1.0, "PUMP": 0.0, "DUMP": 0.0, "PEAK": 0.0},
                "phase": 0,
                "confidence": 0.5,
                "error": str(e),
            }
    
    def train(self, training_data: pd.DataFrame) -> None:
        """
        Train the model with historical data.
        
        This initial version uses rule-based thresholds, but future versions
        could train an actual machine learning model with historical data.
        
        Args:
            training_data: DataFrame with feature data and labels
        """
        if training_data.empty:
            logger.warning("No training data provided, using default thresholds")
            return
        
        try:
            # Extract pump events (class 1)
            pump_events = training_data[training_data['class_id'] == 1]
            
            # Extract dump events (class 2)
            dump_events = training_data[training_data['class_id'] == 2]
            
            # Extract normal events (class 0)
            normal_events = training_data[training_data['class_id'] == 0]
            
            # If we have enough samples, optimize thresholds
            if len(pump_events) >= 10 and len(normal_events) >= 10:
                # Find pump threshold that maximizes F1 score
                best_f1 = 0
                best_threshold = 0.7
                
                for threshold in np.arange(0.3, 0.9, 0.05):
                    # Calculate predictions using this threshold
                    predicted_pumps = training_data['pump_pattern_score'] >= threshold
                    actual_pumps = training_data['class_id'] == 1
                    
                    # Calculate metrics
                    true_positives = sum(predicted_pumps & actual_pumps)
                    false_positives = sum(predicted_pumps & ~actual_pumps)
                    false_negatives = sum(~predicted_pumps & actual_pumps)
                    
                    # Calculate precision and recall
                    precision = true_positives / max(1, true_positives + false_positives)
                    recall = true_positives / max(1, true_positives + false_negatives)
                    
                    # Calculate F1 score
                    f1 = 2 * precision * recall / max(0.001, precision + recall)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                # Update pump threshold
                self._pump_threshold = best_threshold
                logger.info(f"Optimized pump threshold: {self._pump_threshold}")
            
            # Similarly for dump threshold
            if len(dump_events) >= 10 and len(normal_events) >= 10:
                # Find dump threshold that maximizes F1 score
                best_f1 = 0
                best_threshold = 0.7
                
                for threshold in np.arange(0.3, 0.9, 0.05):
                    # Calculate predictions using this threshold
                    predicted_dumps = training_data['dump_pattern_score'] >= threshold
                    actual_dumps = training_data['class_id'] == 2
                    
                    # Calculate metrics
                    true_positives = sum(predicted_dumps & actual_dumps)
                    false_positives = sum(predicted_dumps & ~actual_dumps)
                    false_negatives = sum(~predicted_dumps & actual_dumps)
                    
                    # Calculate precision and recall
                    precision = true_positives / max(1, true_positives + false_positives)
                    recall = true_positives / max(1, true_positives + false_negatives)
                    
                    # Calculate F1 score
                    f1 = 2 * precision * recall / max(0.001, precision + recall)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                # Update dump threshold
                self._dump_threshold = best_threshold
                logger.info(f"Optimized dump threshold: {self._dump_threshold}")
            
            # Update neutral threshold (typically lower than pump/dump thresholds)
            self._neutral_threshold = min(self._pump_threshold, self._dump_threshold) * 0.6
            logger.info(f"Updated neutral threshold: {self._neutral_threshold}")
            
        except Exception as e:
            logger.error(f"Error training pump predictor model: {e}")
    
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
        
        try:
            # Make predictions on evaluation data
            predictions = []
            for _, row in eval_data.iterrows():
                # Convert row to features dictionary
                features = {col: row[col] for col in row.index if col in self._required_features}
                
                # Make prediction
                pred = self.predict(features)
                predictions.append(pred['class_id'])
            
            # Get true labels
            true_labels = eval_data['class_id'].values
            
            # Calculate metrics
            correct = sum(p == t for p, t in zip(predictions, true_labels))
            accuracy = correct / len(true_labels)
            
            # Class-specific metrics
            classes = [0, 1, 2, 3]  # Normal, Pump, Dump, Peak
            precision = {}
            recall = {}
            f1 = {}
            
            for cls in classes:
                # True positives, false positives, false negatives
                tp = sum(1 for p, t in zip(predictions, true_labels) if p == cls and t == cls)
                fp = sum(1 for p, t in zip(predictions, true_labels) if p == cls and t != cls)
                fn = sum(1 for p, t in zip(predictions, true_labels) if p != cls and t == cls)
                
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
            
            # Return evaluation metrics
            return {
                "accuracy": accuracy,
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "precision_normal": precision[0],
                "recall_normal": recall[0],
                "f1_normal": f1[0],
                "precision_pump": precision[1],
                "recall_pump": recall[1],
                "f1_pump": f1[1],
                "precision_dump": precision[2],
                "recall_dump": recall[2],
                "f1_dump": f1[2],
                "precision_peak": precision[3],
                "recall_peak": recall[3],
                "f1_peak": f1[3],
            }
            
        except Exception as e:
            logger.error(f"Error evaluating pump predictor model: {e}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(e)} 