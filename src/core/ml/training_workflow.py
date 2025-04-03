#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Workflow Module

This module provides functionality for scheduling and executing machine
learning model training jobs. It supports automatic retraining on schedules,
data-driven triggers, and manual invocation.
"""

import logging
import time
import os
import json
import yaml
import pickle
import joblib
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import pandas as pd
from datetime import datetime, timedelta
import threading
import schedule
from pathlib import Path

from ..events import EventBus, Event, EventType
from ..features.interfaces import FeatureManager
from ..features.registry import FeatureRegistry
from .feature_transformer import FeatureTransformer
from .interfaces import ModelTrainer, ModelEvaluator, DataCollector

logger = logging.getLogger(__name__)


class ModelTrainingWorkflow:
    """
    Coordinates the entire model training workflow.
    
    This class handles scheduling and executing model training jobs,
    including data collection, preprocessing, training, evaluation,
    and deployment of new models.
    """
    
    def __init__(self, 
                 feature_manager: FeatureManager,
                 event_bus: Optional[EventBus] = None,
                 base_model_path: str = 'models',
                 config_path: str = 'config/ml_training.yaml',
                 enable_scheduling: bool = True):
        """
        Initialize the model training workflow.
        
        Args:
            feature_manager: Feature manager for accessing historical data
            event_bus: Optional event bus for publishing training events
            base_model_path: Base path to store trained models
            config_path: Path to training configuration YAML file
            enable_scheduling: Whether to enable scheduled training jobs
        """
        self._feature_manager = feature_manager
        self._event_bus = event_bus
        self._base_model_path = base_model_path
        self._config_path = config_path
        
        # Ensure model directory exists
        os.makedirs(base_model_path, exist_ok=True)
        
        # Components
        self._trainers: Dict[str, ModelTrainer] = {}
        self._evaluators: Dict[str, ModelEvaluator] = {}
        self._data_collectors: Dict[str, DataCollector] = {}
        
        # Scheduler
        self._scheduler_thread = None
        self._scheduler_stop_event = threading.Event()
        self._enable_scheduling = enable_scheduling
        
        # Training configurations
        self._configs: Dict[str, Dict[str, Any]] = {}
        
        # Load configurations if file exists
        if os.path.exists(config_path):
            self._load_configs()
        
        logger.info(f"ModelTrainingWorkflow initialized with scheduling {'enabled' if enable_scheduling else 'disabled'}")
    
    def register_trainer(self, model_type: str, trainer: ModelTrainer) -> None:
        """
        Register a model trainer.
        
        Args:
            model_type: Type of model (e.g., 'price_prediction')
            trainer: Trainer implementation for this model type
        """
        self._trainers[model_type] = trainer
        logger.info(f"Registered trainer for model type: {model_type}")
    
    def register_evaluator(self, model_type: str, evaluator: ModelEvaluator) -> None:
        """
        Register a model evaluator.
        
        Args:
            model_type: Type of model (e.g., 'price_prediction')
            evaluator: Evaluator implementation for this model type
        """
        self._evaluators[model_type] = evaluator
        logger.info(f"Registered evaluator for model type: {model_type}")
    
    def register_data_collector(self, model_type: str, collector: DataCollector) -> None:
        """
        Register a data collector.
        
        Args:
            model_type: Type of model (e.g., 'price_prediction')
            collector: Data collector implementation for this model type
        """
        self._data_collectors[model_type] = collector
        logger.info(f"Registered data collector for model type: {model_type}")
    
    def add_training_job(self, 
                        model_id: str,
                        model_type: str,
                        schedule_interval: Optional[str] = None,
                        feature_list: Optional[List[str]] = None,
                        hyperparameters: Optional[Dict[str, Any]] = None,
                        data_window_days: int = 30,
                        token_ids: Optional[List[str]] = None,
                        min_training_samples: int = 1000,
                        eval_ratio: float = 0.2) -> bool:
        """
        Add a new model training job.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'price_prediction')
            schedule_interval: Cron-like schedule (e.g., 'daily', '1h', '30m')
            feature_list: List of features to use for training
            hyperparameters: Model hyperparameters
            data_window_days: Days of historical data to use
            token_ids: List of tokens to include in training
            min_training_samples: Minimum required training samples
            eval_ratio: Ratio of data to use for evaluation
            
        Returns:
            True if job was added successfully, False otherwise
        """
        if model_type not in self._trainers:
            logger.error(f"No trainer registered for model type: {model_type}")
            return False
        
        # Create job configuration
        config = {
            'model_id': model_id,
            'model_type': model_type,
            'schedule_interval': schedule_interval,
            'feature_list': feature_list or [],
            'hyperparameters': hyperparameters or {},
            'data_window_days': data_window_days,
            'token_ids': token_ids or [],
            'min_training_samples': min_training_samples,
            'eval_ratio': eval_ratio,
            'last_trained': None,
            'active': True,
            'version': 1
        }
        
        # Add to configurations
        self._configs[model_id] = config
        
        # Schedule the job if requested
        if schedule_interval and self._enable_scheduling:
            self._schedule_job(model_id, schedule_interval)
        
        # Save the updated configurations
        self._save_configs()
        
        logger.info(f"Added training job for model '{model_id}' of type '{model_type}'")
        return True
    
    def update_training_job(self, model_id: str, **kwargs) -> bool:
        """
        Update an existing training job configuration.
        
        Args:
            model_id: ID of the model to update
            **kwargs: Configuration parameters to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        if model_id not in self._configs:
            logger.error(f"No training job found for model ID: {model_id}")
            return False
        
        config = self._configs[model_id]
        
        # Update configuration
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
        
        # Handle schedule changes
        if 'schedule_interval' in kwargs and self._enable_scheduling:
            schedule_interval = kwargs['schedule_interval']
            if schedule_interval:
                self._schedule_job(model_id, schedule_interval)
            else:
                # Remove from schedule
                pass
        
        # Save the updated configurations
        self._save_configs()
        
        logger.info(f"Updated training job for model '{model_id}'")
        return True
    
    def remove_training_job(self, model_id: str) -> bool:
        """
        Remove a training job.
        
        Args:
            model_id: ID of the model to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if model_id not in self._configs:
            logger.error(f"No training job found for model ID: {model_id}")
            return False
        
        # Remove from configurations
        del self._configs[model_id]
        
        # Save the updated configurations
        self._save_configs()
        
        logger.info(f"Removed training job for model '{model_id}'")
        return True
    
    def start_scheduler(self) -> None:
        """Start the training job scheduler."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Scheduler is already running")
            return
        
        if not self._enable_scheduling:
            logger.warning("Scheduling is disabled")
            return
        
        self._scheduler_stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        
        logger.info("Started training job scheduler")
    
    def stop_scheduler(self) -> None:
        """Stop the training job scheduler."""
        if not self._scheduler_thread or not self._scheduler_thread.is_alive():
            logger.warning("Scheduler is not running")
            return
        
        self._scheduler_stop_event.set()
        self._scheduler_thread.join(timeout=5.0)
        
        logger.info("Stopped training job scheduler")
    
    def train_model(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Execute a model training job.
        
        Args:
            model_id: ID of the model to train
            force: Whether to force training even if conditions aren't met
            
        Returns:
            Dictionary with training results
        """
        if model_id not in self._configs:
            raise ValueError(f"No training job found for model ID: {model_id}")
        
        config = self._configs[model_id]
        model_type = config['model_type']
        
        if model_type not in self._trainers:
            raise ValueError(f"No trainer registered for model type: {model_type}")
        
        # Get components
        trainer = self._trainers[model_type]
        evaluator = self._evaluators.get(model_type)
        collector = self._data_collectors.get(model_type)
        
        # Ensure we have a data collector
        if not collector:
            raise ValueError(f"No data collector registered for model type: {model_type}")
        
        logger.info(f"Starting training job for model '{model_id}' of type '{model_type}'")
        
        try:
            # 1. Collect training data
            start_time = datetime.now() - timedelta(days=config['data_window_days'])
            
            train_data, eval_data = self._collect_data(
                collector=collector,
                token_ids=config['token_ids'],
                feature_list=config['feature_list'],
                start_time=start_time,
                end_time=datetime.now(),
                eval_ratio=config['eval_ratio']
            )
            
            # Check if we have enough data
            if len(train_data) < config['min_training_samples'] and not force:
                logger.warning(f"Insufficient training data for model '{model_id}': {len(train_data)} < {config['min_training_samples']}")
                return {
                    'success': False,
                    'error': 'insufficient_data',
                    'samples': len(train_data)
                }
            
            # 2. Train the model
            model, train_metrics = trainer.train(
                train_data=train_data,
                hyperparameters=config['hyperparameters']
            )
            
            # 3. Evaluate the model
            eval_metrics = {}
            if evaluator and len(eval_data) > 0:
                eval_metrics = evaluator.evaluate(model, eval_data)
            
            # 4. Save the model
            version = config['version']
            save_path = self._save_model(model_id, model, version, config, train_metrics, eval_metrics)
            
            # 5. Update job configuration
            config['last_trained'] = datetime.now().isoformat()
            config['version'] += 1
            self._save_configs()
            
            # 6. Publish training completion event
            if self._event_bus:
                self._publish_training_event(model_id, model_type, True, save_path, train_metrics, eval_metrics)
            
            result = {
                'success': True,
                'model_id': model_id,
                'version': version,
                'save_path': save_path,
                'train_samples': len(train_data),
                'eval_samples': len(eval_data),
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics
            }
            
            logger.info(f"Successfully trained model '{model_id}' version {version}")
            return result
            
        except Exception as e:
            logger.error(f"Error training model '{model_id}': {str(e)}", exc_info=True)
            
            # Publish training error event
            if self._event_bus:
                self._publish_training_event(model_id, model_type, False, None, {}, {}, str(e))
            
            return {
                'success': False,
                'error': str(e),
                'model_id': model_id
            }
    
    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """
        List all training jobs.
        
        Returns:
            List of job configurations
        """
        return list(self._configs.values())
    
    def get_training_job(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get training job configuration.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Job configuration or None if not found
        """
        return self._configs.get(model_id)
    
    def _load_configs(self) -> None:
        """Load training job configurations from file."""
        try:
            with open(self._config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data or not isinstance(config_data, dict):
                logger.warning(f"Invalid or empty configuration in {self._config_path}")
                return
            
            self._configs = config_data.get('jobs', {})
            
            # Schedule jobs
            if self._enable_scheduling:
                for model_id, config in self._configs.items():
                    if config.get('active', True) and config.get('schedule_interval'):
                        self._schedule_job(model_id, config['schedule_interval'])
            
            logger.info(f"Loaded {len(self._configs)} training job configurations")
        except Exception as e:
            logger.error(f"Error loading training configurations: {str(e)}", exc_info=True)
    
    def _save_configs(self) -> None:
        """Save training job configurations to file."""
        try:
            config_data = {'jobs': self._configs}
            
            with open(self._config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info(f"Saved {len(self._configs)} training job configurations")
        except Exception as e:
            logger.error(f"Error saving training configurations: {str(e)}", exc_info=True)
    
    def _schedule_job(self, model_id: str, interval: str) -> None:
        """
        Schedule a training job.
        
        Args:
            model_id: ID of the model
            interval: Schedule interval (e.g., 'daily', '1h')
        """
        if not self._enable_scheduling:
            return
        
        # Parse the interval
        if interval == 'daily':
            # Schedule job to run daily at midnight
            schedule.every().day.at("00:00").do(self._run_scheduled_job, model_id=model_id)
            logger.info(f"Scheduled model '{model_id}' for daily training at midnight")
        elif interval == 'weekly':
            # Schedule job to run weekly on Monday at midnight
            schedule.every().monday.at("00:00").do(self._run_scheduled_job, model_id=model_id)
            logger.info(f"Scheduled model '{model_id}' for weekly training on Monday at midnight")
        elif interval.endswith('h'):
            # Schedule job to run every X hours
            try:
                hours = int(interval[:-1])
                schedule.every(hours).hours.do(self._run_scheduled_job, model_id=model_id)
                logger.info(f"Scheduled model '{model_id}' for training every {hours} hours")
            except ValueError:
                logger.error(f"Invalid hour interval: {interval}")
        elif interval.endswith('m'):
            # Schedule job to run every X minutes
            try:
                minutes = int(interval[:-1])
                schedule.every(minutes).minutes.do(self._run_scheduled_job, model_id=model_id)
                logger.info(f"Scheduled model '{model_id}' for training every {minutes} minutes")
            except ValueError:
                logger.error(f"Invalid minute interval: {interval}")
        else:
            logger.error(f"Unsupported schedule interval: {interval}")
    
    def _run_scheduled_job(self, model_id: str) -> None:
        """
        Run a scheduled training job.
        
        Args:
            model_id: ID of the model to train
        """
        try:
            logger.info(f"Running scheduled training job for model '{model_id}'")
            self.train_model(model_id)
        except Exception as e:
            logger.error(f"Error in scheduled training job for model '{model_id}': {str(e)}", exc_info=True)
    
    def _run_scheduler(self) -> None:
        """Run the scheduler loop."""
        logger.info("Starting scheduler loop")
        
        while not self._scheduler_stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)
        
        logger.info("Scheduler loop stopped")
    
    def _collect_data(self,
                     collector: DataCollector,
                     token_ids: List[str],
                     feature_list: List[str],
                     start_time: datetime,
                     end_time: datetime,
                     eval_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect and split training data.
        
        Args:
            collector: Data collector to use
            token_ids: List of tokens to include
            feature_list: List of features to collect
            start_time: Start time for data collection
            end_time: End time for data collection
            eval_ratio: Ratio of data to use for evaluation
            
        Returns:
            Tuple of (training_data, evaluation_data)
        """
        # Collect the data
        data = collector.collect_data(
            token_ids=token_ids,
            feature_list=feature_list,
            start_time=start_time,
            end_time=end_time
        )
        
        if len(data) == 0:
            logger.warning("No data collected")
            return pd.DataFrame(), pd.DataFrame()
        
        # Split into training and evaluation sets
        if eval_ratio <= 0 or eval_ratio >= 1.0:
            return data, pd.DataFrame()
        
        # Shuffle and split (default is time-series split)
        split_idx = int(len(data) * (1.0 - eval_ratio))
        
        # Time-based split
        train_data = data.iloc[:split_idx]
        eval_data = data.iloc[split_idx:]
        
        return train_data, eval_data
    
    def _save_model(self,
                   model_id: str,
                   model: Any,
                   version: int,
                   config: Dict[str, Any],
                   train_metrics: Dict[str, Any],
                   eval_metrics: Dict[str, Any]) -> str:
        """
        Save a trained model.
        
        Args:
            model_id: ID of the model
            model: Trained model object
            version: Model version
            config: Training configuration
            train_metrics: Training metrics
            eval_metrics: Evaluation metrics
            
        Returns:
            Path to the saved model
        """
        # Create model directory
        model_dir = os.path.join(self._base_model_path, model_id, f"v{version}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model object
        model_path = os.path.join(model_dir, f"{model_id}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save configuration and metrics
        metadata = {
            'model_id': model_id,
            'version': version,
            'model_type': config['model_type'],
            'feature_list': config['feature_list'],
            'hyperparameters': config['hyperparameters'],
            'training_date': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature transformer config if available
        if 'transformer_config' in config:
            transformer_path = os.path.join(model_dir, 'transformer.json')
            with open(transformer_path, 'w') as f:
                json.dump(config['transformer_config'], f, indent=2)
        
        logger.info(f"Saved model '{model_id}' version {version} to {model_path}")
        
        return model_path
    
    def _publish_training_event(self,
                              model_id: str,
                              model_type: str,
                              success: bool,
                              model_path: Optional[str],
                              train_metrics: Dict[str, Any],
                              eval_metrics: Dict[str, Any],
                              error: Optional[str] = None) -> None:
        """
        Publish a training event.
        
        Args:
            model_id: ID of the model
            model_type: Type of model
            success: Whether training was successful
            model_path: Path to the saved model
            train_metrics: Training metrics
            eval_metrics: Evaluation metrics
            error: Optional error message
        """
        if not self._event_bus:
            return
            
        event_data = {
            'model_id': model_id,
            'model_type': model_type,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }
        
        if model_path:
            event_data['model_path'] = model_path
            
        if error:
            event_data['error'] = error
        
        try:
            self._event_bus.publish(Event(
                event_type=EventType.MODEL_TRAINED,
                data=event_data,
                source="ModelTrainingWorkflow"
            ))
        except Exception as e:
            logger.error(f"Error publishing training event: {str(e)}", exc_info=True) 