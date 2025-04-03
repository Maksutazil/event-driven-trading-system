#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Manager Module

This module provides the DefaultModelManager implementation that manages
machine learning models, their transformers, and performance tracking.
"""

import os
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from sklearn import metrics as skmetrics

from src.core.events import EventBus, Event, EventType
from src.core.features import FeatureManager

from .interfaces import ModelManager, Model, FeatureTransformer
from .adapters.scikit_learn import ScikitLearnModelAdapter
from .transformers.standard import StandardFeatureTransformer
from .exceptions import (
    TransformerError, ModelNotFoundError, ModelLoadError, ModelSaveError, 
    ModelUpdateError, ModelPredictionError, TransformerNotFoundError, 
    TransformerFitError, TransformerTransformError, InvalidFeatureError, 
    MissingFeatureError, InvalidModelTypeError
)
from src.core.ml.error_handler import MLErrorHandler

logger = logging.getLogger(__name__)


class DefaultModelManager(ModelManager):
    """
    Enhanced implementation of the ModelManager interface.
    
    This class manages machine learning models, their transformers, and performance tracking.
    It loads models, makes predictions, and handles model updates.
    
    Key features:
    - Integration with FeatureManager for automatic feature retrieval
    - Event handling for trade and token creation events
    - Batch operations for predictions and updates
    - Model evaluation and selection utilities
    """
    
    def __init__(self, event_bus: EventBus, feature_manager: FeatureManager):
        """
        Initialize the model manager.
        
        Args:
            event_bus: EventBus for publishing model-related events
            feature_manager: FeatureManager for accessing token features
        """
        self.event_bus = event_bus
        self.feature_manager = feature_manager
        
        # Model storage
        self._models: Dict[str, Model] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Feature transformers for each model
        self._transformers: Dict[str, FeatureTransformer] = {}
        
        # Performance tracking
        self._performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Prediction history for each model
        self._prediction_history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history_size = 1000  # Maximum number of predictions to keep in history
        
        # Locks for thread safety
        self._model_lock = threading.RLock()
        self._performance_lock = threading.RLock()
        self._prediction_lock = threading.RLock()
        
        # Create error handler
        self._error_handler = MLErrorHandler(event_bus=event_bus)
        
        # Register for events if event_bus is provided
        if event_bus is not None:
            self._register_event_handlers()
        
        logger.info("ModelManager initialized")
    
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
            
        Raises:
            ModelLoadError: If the model cannot be loaded
            InvalidModelTypeError: If the model type is not valid
        """
        if not os.path.exists(model_path):
            error = ModelLoadError(model_path, "Model file not found")
            self._error_handler.handle_error(
                error,
                context={
                    'component': 'DefaultModelManager',
                    'operation': 'load_model',
                    'model_id': model_id,
                    'model_path': model_path
                }
            )
            raise error
        
        try:
            if model_type not in ['classification', 'regression']:
                error = InvalidModelTypeError(model_type)
                self._error_handler.handle_error(
                    error,
                    context={
                        'component': 'DefaultModelManager',
                        'operation': 'load_model',
                        'model_id': model_id,
                        'model_type': model_type
                    }
                )
                raise error
                
            with self._model_lock:
                # Choose the appropriate adapter based on file extension and model_type
                model_adapter = self._get_model_adapter(model_path, model_type)
                if not model_adapter:
                    raise ModelLoadError(model_path, f"No suitable adapter found for model type: {model_type}")
                
                # Load the model
                logger.info(f"Loading model {model_id} from {model_path}")
                model = model_adapter.load(model_id, model_path, model_type)
                if model is None:
                    raise ModelLoadError(model_path, f"Adapter failed to load model")
                
                # Store the model and metadata
                self._models[model_id] = model
                self._model_metadata[model_id] = metadata or {}
                
                # Add basic metadata if not provided
                if "name" not in self._model_metadata[model_id]:
                    self._model_metadata[model_id]["name"] = model_id
                if "type" not in self._model_metadata[model_id]:
                    self._model_metadata[model_id]["type"] = model_type
                if "created_at" not in self._model_metadata[model_id]:
                    self._model_metadata[model_id]["created_at"] = time.time()
                if "loaded_at" not in self._model_metadata[model_id]:
                    self._model_metadata[model_id]["loaded_at"] = time.time()
                
                # Initialize performance metrics
                with self._performance_lock:
                    self._performance_metrics[model_id] = {
                        "predictions_count": 0,
                        "last_updated": time.time(),
                        "last_prediction": None,
                        "accuracy": None,
                        "error_rate": 0.0,
                        "mse": None,
                        "mae": None
                    }
                    
                # Initialize prediction history
                with self._prediction_lock:
                    self._prediction_history[model_id] = []
                
                # Publish model loaded event
                self._publish_model_event(EventType.MODEL_LOADED, model_id)
                
                logger.info(f"Successfully loaded model {model_id}")
                return True
                
        except (ModelLoadError, InvalidModelTypeError):
            # These are already handled above, just re-raise
            raise
        except Exception as e:
            error = ModelLoadError(model_path, str(e))
            self._error_handler.handle_error(
                error,
                context={
                    'component': 'DefaultModelManager',
                    'operation': 'load_model',
                    'model_id': model_id,
                    'model_path': model_path,
                    'model_type': model_type
                }
            )
            raise error
    
    @MLErrorHandler.retry_static(max_attempts=3, delay=0.2, backoff=2.0, 
                       exceptions=[TransformerError, ModelPredictionError])
    def get_prediction(self, model_id: str, features: Dict[str, Any]) -> Any:
        """
        Get a prediction from a model.
        
        Args:
            model_id: Identifier for the model
            features: Features to use for prediction
            
        Returns:
            Model prediction
            
        Raises:
            ModelNotFoundError: If the model is not found
            ModelPredictionError: If there is an error making the prediction
            TransformerNotFoundError: If a transformer is required but not found
            MissingFeatureError: If required features are missing
        """
        try:
            with self._model_lock:
                if model_id not in self._models:
                    error = ModelNotFoundError(model_id)
                    self._error_handler.handle_error(
                        error,
                        context={
                            'component': 'DefaultModelManager',
                            'operation': 'get_prediction',
                            'model_id': model_id,
                            'token_id': features.get('token_id', 'unknown')
                        }
                    )
                    raise error
                
                model = self._models[model_id]
                
                # Transform features if a transformer is registered
                try:
                    if model_id in self._transformers:
                        transformer = self._transformers[model_id]
                        transformed_features = transformer.transform(features)
                        logger.debug(f"Transformed features for model {model_id}")
                    else:
                        transformed_features = features
                        logger.debug(f"No transformer registered for model {model_id}, using raw features")
                except TransformerError as e:
                    # Try to recover using raw features if possible
                    recovery_context = {
                        'component': 'DefaultModelManager',
                        'operation': 'get_prediction.transform',
                        'model_id': model_id,
                        'token_id': features.get('token_id', 'unknown'),
                        'raw_features': features
                    }
                    
                    recovery_result = self._error_handler.try_recover(e, recovery_context)
                    if recovery_result is not None:
                        transformed_features = recovery_result
                        logger.info(f"Recovered from transformer error using raw features for model {model_id}")
                    else:
                        self._error_handler.handle_error(
                            e,
                            context=recovery_context
                        )
                        raise
                
                # Get prediction
                try:
                    prediction = model.predict(transformed_features)
                except Exception as e:
                    error = ModelPredictionError(model_id, str(e))
                    self._error_handler.handle_error(
                        error,
                        context={
                            'component': 'DefaultModelManager',
                            'operation': 'get_prediction.predict',
                            'model_id': model_id,
                            'token_id': features.get('token_id', 'unknown')
                        }
                    )
                    raise error
                
                # Update performance metrics
                with self._performance_lock:
                    metrics = self._performance_metrics[model_id]
                    metrics["predictions_count"] += 1
                    metrics["last_prediction"] = prediction
                    metrics["last_updated"] = time.time()
                
                # Record prediction in history
                token_id = features.get("token_id", "unknown")
                self._record_prediction(model_id, token_id, features, prediction)
                
                # Publish prediction event
                self._publish_prediction_event(model_id, prediction, features)
                
                return prediction
                
        except (ModelNotFoundError, ModelPredictionError, TransformerNotFoundError, MissingFeatureError) as e:
            # These errors are already logged and handled, so just re-raise
            raise
        except Exception as e:
            logger.error(f"Error getting prediction from model {model_id}: {e}", exc_info=True)
            with self._performance_lock:
                if model_id in self._performance_metrics:
                    self._performance_metrics[model_id]["error_rate"] += 1
                    
            error = ModelPredictionError(model_id, str(e))
            self._error_handler.handle_error(
                error,
                context={
                    'component': 'DefaultModelManager',
                    'operation': 'get_prediction',
                    'model_id': model_id,
                    'token_id': features.get('token_id', 'unknown')
                }
            )
            raise error
    
    def get_prediction_for_token(self, model_id: str, token_id: str) -> Any:
        """
        Get a prediction for a specific token.
        
        Args:
            model_id: Identifier for the model
            token_id: Identifier for the token
            
        Returns:
            Model prediction
            
        Raises:
            ModelNotFoundError: If the model is not found
            ModelPredictionError: If there is an error making the prediction
            MissingFeatureError: If required features are missing
        """
        try:
            with self._model_lock:
                if model_id not in self._models:
                    logger.error(f"Model {model_id} not found")
                    raise ModelNotFoundError(model_id)
                
                # Get the list of required features
                required_features = []
                if model_id in self._transformers:
                    transformer = self._transformers[model_id]
                    required_features = transformer.get_required_features()
                
                # Fetch features for the token from feature manager
                try:
                    features = self.feature_manager.get_features_for_token(token_id, required_features)
                    
                    # Check if we have missing features
                    if required_features and (not features or not all(f in features for f in required_features)):
                        missing_features = [f for f in required_features if f not in features]
                        error = MissingFeatureError(missing_features[0] if missing_features else "required features")
                        
                        # Try to recover by providing default values
                        recovery_context = {
                            'component': 'DefaultModelManager',
                            'operation': 'get_prediction_for_token',
                            'model_id': model_id,
                            'token_id': token_id,
                            'required_features': required_features,
                            'available_features': features or {},
                            'default_feature_value': 0.0
                        }
                        
                        recovery_result = self._error_handler.try_recover(error, recovery_context)
                        if recovery_result is not None:
                            features = recovery_result
                            logger.info(f"Recovered from missing features by using defaults for model {model_id}")
                        else:
                            self._error_handler.handle_error(
                                error,
                                context=recovery_context
                            )
                            raise error
                    
                    # Add token_id to features if not already present
                    if "token_id" not in features:
                        features["token_id"] = token_id
                    
                    # Make the prediction
                    prediction = self.get_prediction(model_id, features)
                    logger.info(f"Prediction for token {token_id} from model {model_id}: {prediction}")
                    return prediction
                    
                except MissingFeatureError:
                    # Already handled above, just re-raise
                    raise
                except Exception as e:
                    error = ModelPredictionError(model_id, f"Failed to get features: {str(e)}")
                    self._error_handler.handle_error(
                        error,
                        context={
                            'component': 'DefaultModelManager',
                            'operation': 'get_prediction_for_token.get_features',
                            'model_id': model_id,
                            'token_id': token_id
                        }
                    )
                    raise error
                
        except (ModelNotFoundError, ModelPredictionError, MissingFeatureError):
            # These are already handled above, just re-raise
            raise
        except Exception as e:
            error = ModelPredictionError(model_id, str(e))
            self._error_handler.handle_error(
                error,
                context={
                    'component': 'DefaultModelManager',
                    'operation': 'get_prediction_for_token',
                    'model_id': model_id,
                    'token_id': token_id
                }
            )
            raise error
    
    def get_batch_predictions(self, model_id: str, token_ids: List[str]) -> Dict[str, Any]:
        """
        Get predictions for multiple tokens.
        
        Args:
            model_id: Identifier for the model
            token_ids: List of token identifiers
            
        Returns:
            Dictionary mapping token IDs to predictions
        """
        results = {}
        for token_id in token_ids:
            prediction = self.get_prediction_for_token(model_id, token_id)
            if prediction is not None:
                results[token_id] = prediction
        
        return results
    
    def update_model(self, model_id: str, features: Dict[str, Any], target: Any) -> bool:
        """
        Update a model with new data.
        
        Args:
            model_id: Identifier for the model
            features: Features for the new data point
            target: Target value for the new data point
            
        Returns:
            Whether the update was successful
            
        Raises:
            ModelNotFoundError: If the model is not found
            ModelUpdateError: If there is an error updating the model
            TransformerNotFoundError: If a transformer is required but not found
            MissingFeatureError: If required features are missing
        """
        try:
            with self._model_lock:
                if model_id not in self._models:
                    raise ModelNotFoundError(model_id)
                
                model = self._models[model_id]
                
                # Transform features if a transformer is registered
                if model_id in self._transformers:
                    transformer = self._transformers[model_id]
                    transformed_features = transformer.transform(features)
                else:
                    transformed_features = features
                
                # Update the model
                updated = model.update(transformed_features, target)
                
                if updated:
                    # Update metadata
                    with self._model_lock:
                        self._model_metadata[model_id]["last_updated"] = time.time()
                    
                    # Publish model updated event
                    self._publish_model_event(EventType.MODEL_UPDATED, model_id)
                    
                    logger.info(f"Successfully updated model {model_id}")
                
                return updated
                
        except (ModelNotFoundError, ModelUpdateError, TransformerNotFoundError, MissingFeatureError):
            # Re-raise these exceptions
            raise
        except Exception as e:
            raise ModelUpdateError(model_id, str(e))
    
    def update_model_for_token(self, model_id: str, token_id: str, target: Any) -> bool:
        """
        Update a model with data from a specific token.
        
        Args:
            model_id: Identifier for the model
            token_id: Identifier for the token
            target: Target value for the update
            
        Returns:
            Whether the update was successful
        """
        try:
            with self._model_lock:
                if model_id not in self._models:
                    logger.error(f"Model {model_id} not found")
                    return False
                
                # Get the list of required features
                required_features = []
                if model_id in self._transformers:
                    transformer = self._transformers[model_id]
                    required_features = transformer.get_required_features()
                
                # Get features from feature manager
                if not required_features:
                    features = self.feature_manager.get_features_for_token(token_id)
                else:
                    features = self.feature_manager.get_features_for_token(token_id, required_features)
                
                # Add token_id to features
                features["token_id"] = token_id
                
                # Update the model using the existing method
                return self.update_model(model_id, features, target)
        
        except Exception as e:
            logger.error(f"Error updating model {model_id} with token {token_id}: {e}", exc_info=True)
            return False
    
    def register_transformer(self, model_id: str, transformer: FeatureTransformer) -> bool:
        """
        Register a feature transformer for a model.
        
        Args:
            model_id: Identifier for the model
            transformer: FeatureTransformer instance
            
        Returns:
            Whether the registration was successful
            
        Raises:
            ModelNotFoundError: If the model is not found
        """
        try:
            with self._model_lock:
                if model_id not in self._models:
                    raise ModelNotFoundError(model_id)
                
                self._transformers[model_id] = transformer
                logger.info(f"Registered transformer for model {model_id}")
                return True
                
        except ModelNotFoundError:
            # Re-raise this exception
            raise
        except Exception as e:
            logger.error(f"Error registering transformer for model {model_id}: {e}", exc_info=True)
            return False
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dictionary of performance metrics
            
        Raises:
            ModelNotFoundError: If the model is not found
        """
        with self._performance_lock:
            if model_id not in self._performance_metrics:
                raise ModelNotFoundError(model_id)
            
            return self._performance_metrics[model_id].copy()
    
    def list_models(self) -> List[str]:
        """
        Get a list of all available models.
        
        Returns:
            List of model identifiers
        """
        with self._model_lock:
            return list(self._models.keys())
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dictionary of model metadata
            
        Raises:
            ModelNotFoundError: If the model is not found
        """
        with self._model_lock:
            if model_id not in self._model_metadata:
                raise ModelNotFoundError(model_id)
            
            # Combine model metadata with model's own metadata
            combined_metadata = self._model_metadata[model_id].copy()
            if model_id in self._models:
                model_metadata = self._models[model_id].get_metadata()
                combined_metadata.update(model_metadata)
                
            return combined_metadata
    
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
            
        Raises:
            ModelNotFoundError: If the model is not found
            ModelPredictionError: If there is an error making predictions
            TransformerTransformError: If there is an error transforming features
        """
        try:
            with self._model_lock:
                if model_id not in self._models:
                    raise ModelNotFoundError(model_id)
                
                model = self._models[model_id]
                
                # Get model type
                model_type = model.model_type
                
                # Transform features if a transformer is registered
                transformed_features = []
                try:
                    if model_id in self._transformers:
                        transformer = self._transformers[model_id]
                        for features in eval_features:
                            transformed = transformer.transform(features)
                            transformed_features.append(transformed)
                    else:
                        transformed_features = eval_features
                except Exception as e:
                    raise TransformerTransformError(str(e))
                
                # Make predictions
                predictions = []
                try:
                    for features in transformed_features:
                        pred = model.predict(features)
                        
                        # Extract prediction value from classification result if needed
                        if isinstance(pred, dict) and "class_id" in pred:
                            pred = pred["class_id"]
                        
                        predictions.append(pred)
                except Exception as e:
                    raise ModelPredictionError(model_id, str(e))
                
                # Calculate metrics based on model type
                metrics = {}
                if model_type == "classification":
                    # Classification metrics
                    if len(set(eval_targets)) > 2:
                        # Multi-class classification
                        metrics["accuracy"] = float(skmetrics.accuracy_score(eval_targets, predictions))
                        metrics["f1_weighted"] = float(skmetrics.f1_score(eval_targets, predictions, average="weighted"))
                        try:
                            metrics["confusion_matrix"] = skmetrics.confusion_matrix(eval_targets, predictions).tolist()
                        except:
                            pass
                    else:
                        # Binary classification
                        metrics["accuracy"] = float(skmetrics.accuracy_score(eval_targets, predictions))
                        metrics["precision"] = float(skmetrics.precision_score(eval_targets, predictions, zero_division=0))
                        metrics["recall"] = float(skmetrics.recall_score(eval_targets, predictions, zero_division=0))
                        metrics["f1"] = float(skmetrics.f1_score(eval_targets, predictions, zero_division=0))
                else:
                    # Regression metrics
                    metrics["mse"] = float(skmetrics.mean_squared_error(eval_targets, predictions))
                    metrics["mae"] = float(skmetrics.mean_absolute_error(eval_targets, predictions))
                    metrics["r2"] = float(skmetrics.r2_score(eval_targets, predictions))
                    
                # Update performance metrics
                with self._performance_lock:
                    perf_metrics = self._performance_metrics[model_id]
                    for k, v in metrics.items():
                        perf_metrics[k] = v
                    perf_metrics["last_evaluated"] = time.time()
                    perf_metrics["eval_samples"] = len(eval_targets)
                
                # Add evaluation timestamp
                metrics["timestamp"] = time.time()
                metrics["samples"] = len(eval_targets)
                
                logger.info(f"Evaluated model {model_id} on {len(eval_targets)} samples")
                return metrics
                
        except (ModelNotFoundError, ModelPredictionError, TransformerTransformError):
            # Re-raise these exceptions
            raise
        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}", exc_info=True)
            return {"error": str(e)}
    
    def evaluate_model_on_tokens(self, model_id: str, token_ids: List[str], 
                               target_feature: str, prediction_window: int = 1) -> Dict[str, Any]:
        """
        Evaluate a model on historical data for a set of tokens.
        
        This method gets historical feature data and evaluates the model by comparing
        predictions to actual values of a target feature after a prediction window.
        
        Args:
            model_id: Identifier for the model
            token_ids: List of token identifiers to evaluate on
            target_feature: Feature name to use as the target/ground truth
            prediction_window: Number of events/periods to look ahead for target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if model_id not in self._models:
                logger.error(f"Model {model_id} not found")
                return {}
            
            model = self._models[model_id]
            
            # Get event history manager if available
            event_history_manager = None
            if hasattr(self.event_bus, "get_event_history_manager"):
                event_history_manager = self.event_bus.get_event_history_manager()
            
            if not event_history_manager:
                logger.error("Event history manager not available for evaluation")
                return {"error": "Event history manager not available"}
            
            # Get transformer for the model
            transformer = self._transformers.get(model_id)
            
            # Collect evaluation data
            eval_features = []
            eval_targets = []
            
            for token_id in token_ids:
                # Get token event history
                token_events = event_history_manager.get_events_for_token(token_id)
                if not token_events:
                    continue
                    
                # Sort events by timestamp
                token_events.sort(key=lambda e: e.timestamp)
                
                # Get feature snapshots at each event
                for i in range(len(token_events) - prediction_window):
                    # Get features at time of event i
                    event_time = token_events[i].timestamp
                    features = self.feature_manager.get_features_for_token(token_id)
                    features["token_id"] = token_id
                    features["timestamp"] = event_time
                    
                    # Get target value at time of event i + prediction_window
                    future_event = token_events[i + prediction_window]
                    future_features = self.feature_manager.get_features_for_token(token_id)
                    
                    if target_feature in future_features:
                        target_value = future_features[target_feature]
                        
                        # Add to evaluation data
                        eval_features.append(features)
                        eval_targets.append(target_value)
            
            # Evaluate on collected data
            if eval_features and eval_targets:
                return self.evaluate_model(model_id, eval_features, eval_targets)
            else:
                logger.warning(f"No evaluation data collected for model {model_id}")
                return {"error": "No evaluation data collected", "samples": 0}
                
        except Exception as e:
            logger.error(f"Error evaluating model {model_id} on tokens: {e}", exc_info=True)
            return {"error": str(e)}
            
    def get_best_model_for_token(self, token_id: str, model_type: str = None) -> Optional[str]:
        """
        Get the best performing model for a specific token.
        
        Args:
            token_id: Identifier for the token
            model_type: Optional filter for model type
            
        Returns:
            Model ID of the best model, or None if no suitable model found
        """
        try:
            # Get list of all models
            models = self.list_models()
            if not models:
                return None
            
            best_model = None
            best_score = -float('inf')
            
            for model_id in models:
                # Check model type if specified
                if model_type is not None:
                    metadata = self.get_model_metadata(model_id)
                    if metadata.get('type') != model_type:
                        continue
                
                # Get model performance metrics
                metrics = self.get_model_performance(model_id)
                
                # Define a score based on the model type
                score = None
                if model_type == 'classification':
                    score = metrics.get('accuracy') or metrics.get('f1') or 0.0
                else:  # regression
                    # For regression, lower error is better
                    mse = metrics.get('mse')
                    if mse is not None:
                        score = -mse  # Negative because we want to maximize
                
                # Update best model if this one is better
                if score is not None and (best_model is None or score > best_score):
                    best_model = model_id
                    best_score = score
            
            return best_model
                
        except Exception as e:
            logger.error(f"Error finding best model for token {token_id}: {e}", exc_info=True)
            return None
    
    def save_model(self, model_id: str, file_path: str) -> bool:
        """
        Save a model to a file.
        
        Args:
            model_id: Identifier for the model
            file_path: Path to save the model
            
        Returns:
            Whether the save was successful
            
        Raises:
            ModelNotFoundError: If the model is not found
            ModelSaveError: If there is an error saving the model
        """
        try:
            with self._model_lock:
                if model_id not in self._models:
                    raise ModelNotFoundError(model_id)
                
                model = self._models[model_id]
                
                # Check if model has save method
                if not hasattr(model, "save"):
                    raise ModelSaveError(model_id, file_path, "Model does not support saving")
                
                # Save the model
                try:
                    saved = model.save(file_path)
                except Exception as e:
                    raise ModelSaveError(model_id, file_path, str(e))
                
                if saved:
                    # Save metadata alongside model
                    metadata_path = f"{file_path}.meta"
                    with open(metadata_path, 'w') as f:
                        json.dump(self.get_model_metadata(model_id), f, indent=4)
                    
                    logger.info(f"Saved model {model_id} to {file_path}")
                    
                return saved
                
        except (ModelNotFoundError, ModelSaveError):
            # Re-raise these exceptions
            raise
        except Exception as e:
            raise ModelSaveError(model_id, file_path, str(e))
    
    def save_transformer(self, model_id: str, file_path: str) -> bool:
        """
        Save a model's transformer to a file.
        
        Args:
            model_id: Identifier for the model
            file_path: Path to save the transformer
            
        Returns:
            Whether the save was successful
            
        Raises:
            TransformerNotFoundError: If no transformer is registered for the model
            ModelSaveError: If there is an error saving the transformer
        """
        try:
            with self._model_lock:
                if model_id not in self._transformers:
                    raise TransformerNotFoundError(model_id)
                
                transformer = self._transformers[model_id]
                
                # Check if transformer has save method
                if not hasattr(transformer, "save"):
                    raise ModelSaveError(model_id, file_path, "Transformer does not support saving")
                
                # Save the transformer
                try:
                    saved = transformer.save(file_path)
                except Exception as e:
                    raise ModelSaveError(model_id, file_path, f"Transformer save error: {str(e)}")
                
                if saved:
                    logger.info(f"Saved transformer for model {model_id} to {file_path}")
                    
                return saved
                
        except (TransformerNotFoundError, ModelSaveError):
            # Re-raise these exceptions
            raise
        except Exception as e:
            raise ModelSaveError(model_id, file_path, f"Transformer save error: {str(e)}")
    
    def _get_model_adapter(self, model_path: str, model_type: str) -> Any:
        """
        Get an appropriate model adapter for the file.
        
        Args:
            model_path: Path to the model file
            model_type: Type of the model
            
        Returns:
            ModelAdapter instance
        """
        # Determine the file extension
        _, file_ext = os.path.splitext(model_path)
        file_ext = file_ext.lower()
        
        # Choose adapter based on extension and type
        if file_ext in ['.pkl', '.joblib']:
            # Likely a scikit-learn model
            return ScikitLearnModelAdapter()
        elif file_ext in ['.h5', '.keras']:
            # Likely a Keras/TensorFlow model
            logger.warning("TensorFlow models not yet supported")
            return None
        else:
            logger.warning(f"Unknown model file extension: {file_ext}")
            return None
    
    def _publish_model_event(self, event_type: EventType, model_id: str, **kwargs) -> None:
        """
        Publish a model-related event.
        
        Args:
            event_type: Type of the event
            model_id: Identifier for the model
            **kwargs: Additional event data
        """
        if not self.event_bus:
            return
        
        data = {
            "model_id": model_id,
            "timestamp": time.time(),
            **kwargs
        }
        
        try:
            self.event_bus.publish(Event(event_type=event_type, data=data))
        except Exception as e:
            logger.error(f"Error publishing model event: {e}", exc_info=True)
    
    def _publish_prediction_event(self, model_id: str, prediction: Any, features: Dict[str, Any]) -> None:
        """
        Publish a prediction event.
        
        Args:
            model_id: Identifier for the model
            prediction: Model prediction
            features: Features used for the prediction
        """
        if not self.event_bus:
            return
        
        # Extract token_id from features if available
        token_id = features.get("token_id", "unknown")
        
        # Include important metadata for decision making
        metadata = self._model_metadata.get(model_id, {})
        model_type = metadata.get("type", "unknown")
        
        # Create a filtered set of features to include in the event
        # This avoids including very large feature sets but keeps important ones
        filtered_features = {}
        important_features = ["price", "volume", "price_change_pct", "volume_change_pct", 
                              "volatility", "momentum", "rsi", "timestamp"]
        
        for key in important_features:
            if key in features:
                filtered_features[key] = features[key]
        
        data = {
            "model_id": model_id,
            "token_id": token_id,
            "prediction": prediction,
            "prediction_type": model_type,
            "features": filtered_features,
            "timestamp": time.time(),
            "confidence": metadata.get("confidence", 1.0)  # Include confidence if available
        }
        
        try:
            logger.info(f"Publishing MODEL_PREDICTION event for token {token_id} from model {model_id}: {prediction}")
            self.event_bus.publish(Event(event_type=EventType.MODEL_PREDICTION, data=data))
        except Exception as e:
            logger.error(f"Error publishing prediction event: {e}", exc_info=True)
    
    def _record_prediction(self, model_id: str, token_id: str, features: Dict[str, Any], 
                        prediction: Any) -> None:
        """
        Record a prediction in the history.
        
        Args:
            model_id: Identifier for the model
            token_id: Identifier for the token
            features: Features used for the prediction
            prediction: Model prediction
        """
        try:
            with self._prediction_lock:
                # Add to history
                history = self._prediction_history.get(model_id, [])
                
                # Create record
                record = {
                    "token_id": token_id,
                    "timestamp": time.time(),
                    "prediction": prediction
                }
                
                # Add key features if available
                for key in ["price", "current_price", "volume", "market_cap"]:
                    if key in features:
                        record[key] = features[key]
                
                # Add to history
                history.append(record)
                
                # Limit history size
                if len(history) > self._max_history_size:
                    history = history[-self._max_history_size:]
                
                self._prediction_history[model_id] = history
                
        except Exception as e:
            logger.error(f"Error recording prediction for model {model_id}: {e}", exc_info=True)
    
    def _register_event_handlers(self) -> None:
        """Register handlers for events."""
        try:
            if self.event_bus is None:
                return
            
            # Import the standardized EventHandlerWrapper
            from src.core.events.base import EventHandlerWrapper
            
            # Register for token trade events
            self.event_bus.subscribe(EventType.TOKEN_TRADE, 
                                   EventHandlerWrapper(self._handle_token_trade_event))
            
            # Register for token created events
            self.event_bus.subscribe(EventType.TOKEN_CREATED, 
                                   EventHandlerWrapper(self._handle_token_created_event))
            
            logger.info("Registered event handlers for ModelManager")
            
        except Exception as e:
            logger.error(f"Error registering event handlers: {e}", exc_info=True)
    
    def _handle_token_trade_event(self, event: Event) -> None:
        """
        Handle token trade events.
        
        Makes predictions for all models when a token trade occurs.
        
        Args:
            event: Token trade event
        """
        try:
            token_id = event.data.get('token_id') or event.data.get('mint')
            if not token_id:
                logger.warning("Token trade event has no token_id or mint field")
                return
            
            logger.debug(f"Handling token trade event for {token_id}")
            
            # Make predictions for all models
            for model_id in self.list_models():
                try:
                    prediction = self.get_prediction(model_id, token_id)
                    logger.debug(f"Model {model_id} predicted {prediction} for token {token_id}")
                except Exception as e:
                    logger.error(f"Error getting prediction for token {token_id} from model {model_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error handling token trade event: {e}", exc_info=True)
    
    def _handle_token_created_event(self, event: Event) -> None:
        """
        Handle token created events.
        
        Initializes predictions for newly created tokens.
        
        Args:
            event: Token created event
        """
        try:
            token_id = event.data.get('token_id') or event.data.get('mint')
            if not token_id:
                logger.warning("Token created event has no token_id or mint field")
                return
            
            logger.debug(f"Handling token created event for {token_id}")
            
            # Make initial predictions for all models
            # This helps establish baseline predictions early
            for model_id in self.list_models():
                try:
                    prediction = self.get_prediction(model_id, token_id)
                    logger.debug(f"Initial prediction for new token {token_id} using model {model_id}: {prediction}")
                except Exception as e:
                    logger.error(f"Error getting initial prediction for token {token_id} from model {model_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error handling token created event: {e}", exc_info=True)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors handled by the model manager.
        
        Returns:
            Dictionary with error statistics
        """
        return self._error_handler.get_error_statistics()
    
    def clear_error_history(self) -> None:
        """Clear the error history and counts."""
        self._error_handler.clear_error_history() 