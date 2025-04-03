#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Learning Interfaces

This module defines the interfaces for machine learning components,
including model trainers, evaluators, and data collectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from datetime import datetime


class ModelTrainer(ABC):
    """
    Interface for model trainers.
    
    Model trainers are responsible for training models using
    prepared datasets and hyperparameters.
    """
    
    @abstractmethod
    def train(self, train_data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model.
        
        Args:
            train_data: Training data
            hyperparameters: Hyperparameters for model training
            
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Get the features required for training.
        
        Returns:
            List of feature names
        """
        pass


class ModelEvaluator(ABC):
    """
    Interface for model evaluators.
    
    Model evaluators are responsible for evaluating trained models
    using holdout datasets and computing performance metrics.
    """
    
    @abstractmethod
    def evaluate(self, model: Any, eval_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate a model.
        
        Args:
            model: Trained model to evaluate
            eval_data: Evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[str]:
        """
        Get the metrics computed by this evaluator.
        
        Returns:
            List of metric names
        """
        pass


class DataCollector(ABC):
    """
    Interface for data collectors.
    
    Data collectors are responsible for gathering historical data
    for model training and evaluation.
    """
    
    @abstractmethod
    def collect_data(self, 
                    token_ids: List[str],
                    feature_list: List[str],
                    start_time: datetime,
                    end_time: datetime) -> pd.DataFrame:
        """
        Collect data for model training.
        
        Args:
            token_ids: List of tokens to collect data for
            feature_list: List of features to collect
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            DataFrame containing collected data
        """
        pass
    
    @abstractmethod
    def get_available_features(self) -> List[str]:
        """
        Get the available features from this collector.
        
        Returns:
            List of available feature names
        """
        pass


class Model(ABC):
    """
    Interface for ML models.
    
    Models provide a standard interface for making predictions
    using feature data.
    """
    
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Any:
        """
        Make a prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Prediction value or values
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Get the features required for prediction.
        
        Returns:
            List of required feature names
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Get the type of this model.
        
        Returns:
            Model type identifier
        """
        pass
    
    @property
    @abstractmethod
    def model_version(self) -> str:
        """
        Get the version of this model.
        
        Returns:
            Model version identifier
        """
        pass


class FeatureTransformer(ABC):
    """Interface for transforming raw features to model inputs."""
    
    @abstractmethod
    def transform(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw features to model inputs.
        
        Args:
            features: Raw features
            
        Returns:
            Transformed features suitable for model input
        """
        pass
        
    @abstractmethod
    def fit(self, features_list: List[Dict[str, Any]]) -> bool:
        """
        Fit the transformer to a batch of features.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Whether the fit was successful
        """
        pass
        
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Get the list of features required by this transformer.
        
        Returns:
            List of feature names
        """
        pass


class ModelManager(ABC):
    """Interface for the model manager component."""
    
    @abstractmethod
    def load_model(self, model_id: str, model_path: str, model_type: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a model from a file.
        
        Args:
            model_id: Identifier for the model
            model_path: Path to the model file
            model_type: Type of the model (e.g., 'classification', 'regression')
            metadata: Optional metadata for the model
            
        Returns:
            Whether the load was successful
        """
        pass
        
    @abstractmethod
    def get_prediction(self, model_id: str, features: Dict[str, Any]) -> Any:
        """
        Get a prediction from a model.
        
        Args:
            model_id: Identifier for the model
            features: Features to use for prediction
            
        Returns:
            Model prediction
        """
        pass
        
    @abstractmethod
    def update_model(self, model_id: str, features: Dict[str, Any], target: Any) -> bool:
        """
        Update a model with new data.
        
        Args:
            model_id: Identifier for the model
            features: Features for the new data point
            target: Target value for the new data point
            
        Returns:
            Whether the update was successful
        """
        pass
        
    @abstractmethod
    def register_transformer(self, model_id: str, transformer: FeatureTransformer) -> bool:
        """
        Register a feature transformer for a model.
        
        Args:
            model_id: Identifier for the model
            transformer: FeatureTransformer instance
            
        Returns:
            Whether the registration was successful
        """
        pass
        
    @abstractmethod
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dictionary of performance metrics
        """
        pass
        
    @abstractmethod
    def list_models(self) -> List[str]:
        """
        Get a list of all available models.
        
        Returns:
            List of model identifiers
        """
        pass
        
    @abstractmethod
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dictionary of model metadata
        """
        pass
        
    @abstractmethod
    def evaluate_model(self, model_id: str, eval_features: List[Dict[str, Any]], 
                     eval_targets: List[Any]) -> Dict[str, Any]:
        """
        Evaluate a model on validation data.
        
        Args:
            model_id: Identifier for the model
            eval_features: List of feature dictionaries
            eval_targets: List of target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass 