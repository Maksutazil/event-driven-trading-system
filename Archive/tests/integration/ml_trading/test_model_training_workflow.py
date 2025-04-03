#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test for Model Training Workflow

This module tests the integration of the model training workflow with the
feature management and event systems, ensuring that models can be properly
trained, evaluated, and made available to the trading system.
"""

import unittest
import os
import shutil
import tempfile
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

from src.core.ml.interfaces import ModelTrainer, ModelEvaluator, DataCollector, Model
from src.core.ml.training_workflow import ModelTrainingWorkflow
from src.core.events import EventType, Event
from tests.integration.base_integration_test import BaseIntegrationTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleModel:
    """A simple model implementation for testing."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_type = "regression"
        self.model_version = "1.0.0"
        self.coefficients = None
    
    def train(self, X, y):
        """Simple linear regression."""
        # Basic implementation of ordinary least squares
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        return self
    
    def predict(self, X):
        """Make predictions with the model."""
        if self.coefficients is None:
            return np.zeros(X.shape[0])
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_intercept @ self.coefficients


class MockModelTrainer(ModelTrainer):
    """Mock model trainer for testing."""
    
    def __init__(self):
        self.training_calls = 0
        
    def train(self, train_data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Train a model on the provided data."""
        self.training_calls += 1
        
        # Extract features and target
        X = train_data.drop(columns=['target']).values
        y = train_data['target'].values
        
        # Create and train a simple model
        model = SimpleModel(hyperparameters.get('model_id', 'test_model'))
        model.train(X, y)
        
        # Return the model and some metrics
        metrics = {
            'training_samples': len(train_data),
            'training_time': 0.1,
            'training_iterations': 1
        }
        
        return model, metrics
    
    def get_required_features(self) -> List[str]:
        """Get the features required by this trainer."""
        return [
            "market.binance.price",
            "market.binance.volume",
            "technical.momentum.rsi",
            "technical.trend.sma"
        ]


class MockModelEvaluator(ModelEvaluator):
    """Mock model evaluator for testing."""
    
    def __init__(self):
        self.evaluation_calls = 0
        
    def evaluate(self, model: Any, eval_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a model on the provided data."""
        self.evaluation_calls += 1
        
        # Extract features and target
        X = eval_data.drop(columns=['target']).values
        y = eval_data['target'].values
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate some metrics
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        
        return {
            'mse': mse,
            'mae': mae,
            'samples': len(eval_data),
            'evaluation_time': 0.05
        }
    
    def get_metrics(self) -> List[str]:
        """Get the metrics provided by this evaluator."""
        return ['mse', 'mae']


class MockDataCollector(DataCollector):
    """Mock data collector for testing."""
    
    def __init__(self):
        self.collection_calls = 0
        self.data = self._generate_test_data()
        
    def collect_data(self, 
                    token_ids: List[str],
                    feature_list: List[str],
                    start_time: datetime,
                    end_time: datetime) -> pd.DataFrame:
        """Collect data for the specified tokens and features."""
        self.collection_calls += 1
        
        # Filter data by time range and tokens
        filtered_data = self.data[
            (self.data['timestamp'] >= start_time) & 
            (self.data['timestamp'] <= end_time) &
            (self.data['token_id'].isin(token_ids))
        ].copy()
        
        # Select only the requested features plus token_id and timestamp
        features_to_include = ['token_id', 'timestamp', 'target']
        for feature in feature_list:
            if feature in self.data.columns:
                features_to_include.append(feature)
        
        return filtered_data[features_to_include]
    
    def get_available_features(self) -> List[str]:
        """Get the features available from this collector."""
        # Return all columns except token_id, timestamp, and target
        return [col for col in self.data.columns if col not in ['token_id', 'timestamp', 'target']]
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic test data."""
        # Create a date range for the past 60 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Create token IDs
        token_ids = ['TOKEN_A', 'TOKEN_B', 'TOKEN_C']
        
        # Generate a DataFrame with all combinations of dates and tokens
        data = []
        for token_id in token_ids:
            for date in dates:
                # Generate random feature values
                price = np.random.uniform(10, 100)
                volume = np.random.uniform(1000, 10000)
                rsi = np.random.uniform(0, 100)
                sma = price * (1 + np.random.uniform(-0.1, 0.1))
                
                # Calculate target (next hour's price change %)
                target = np.random.uniform(-0.05, 0.05)
                
                data.append({
                    'token_id': token_id,
                    'timestamp': date,
                    'market.binance.price': price,
                    'market.binance.volume': volume,
                    'technical.momentum.rsi': rsi,
                    'technical.trend.sma': sma,
                    'target': target
                })
        
        return pd.DataFrame(data)


class TestModelTrainingWorkflow(BaseIntegrationTest):
    """
    Integration tests for the model training workflow.
    """
    
    def setUp(self) -> None:
        """Set up the test environment."""
        super().setUp()
        
        # Create a temporary directory for models
        self.model_dir = tempfile.mkdtemp()
        
        # Create mock components
        self.feature_manager = self.create_feature_manager()
        self.trainer = MockModelTrainer()
        self.evaluator = MockModelEvaluator()
        self.data_collector = MockDataCollector()
        
        # Create the training workflow
        self.training_workflow = ModelTrainingWorkflow(
            feature_manager=self.feature_manager,
            event_bus=self.event_bus,
            base_model_path=self.model_dir,
            enable_scheduling=False  # Disable scheduling for tests
        )
        
        # Register components
        self.training_workflow.register_trainer("regression", self.trainer)
        self.training_workflow.register_evaluator("regression", self.evaluator)
        self.training_workflow.register_data_collector("regression", self.data_collector)
        
        # Create event capture for training events
        self.event_bus.subscribe(EventType.MODEL_UPDATED, self.on_model_updated)
        self.model_update_events = []
    
    def tearDown(self) -> None:
        """Clean up after the test."""
        super().tearDown()
        
        # Remove the temporary model directory
        shutil.rmtree(self.model_dir)
    
    def on_model_updated(self, event: Event) -> None:
        """Callback for MODEL_UPDATED events."""
        self.model_update_events.append(event)
    
    def test_training_job_lifecycle(self):
        """
        Test the full lifecycle of a training job.
        
        This test validates:
        1. Adding a training job
        2. Training a model
        3. Updating a training job
        4. Removing a training job
        """
        # Define test tokens
        test_tokens = ["TOKEN_A", "TOKEN_B"]
        
        # Add a training job
        job_added = self.training_workflow.add_training_job(
            model_id="test_price_model",
            model_type="regression",
            feature_list=self.trainer.get_required_features(),
            hyperparameters={"learning_rate": 0.01},
            data_window_days=30,
            token_ids=test_tokens,
            min_training_samples=10
        )
        
        self.assertTrue(job_added, "Failed to add training job")
        
        # Get the job and verify its configuration
        job = self.training_workflow.get_training_job("test_price_model")
        self.assertIsNotNone(job, "Training job not found")
        self.assertEqual(job["model_type"], "regression")
        self.assertEqual(job["token_ids"], test_tokens)
        
        # Train the model
        result = self.training_workflow.train_model("test_price_model")
        
        # Verify the training was successful
        self.assertTrue(result["success"], "Model training failed")
        self.assertIn("model_path", result, "Model path not in training result")
        self.assertIn("eval_metrics", result, "Evaluation metrics not in training result")
        
        # Verify that the model file exists
        self.assertTrue(os.path.exists(result["model_path"]), "Model file does not exist")
        
        # Verify that the component methods were called
        self.assertEqual(self.trainer.training_calls, 1, "Trainer was not called")
        self.assertEqual(self.evaluator.evaluation_calls, 1, "Evaluator was not called")
        self.assertGreater(self.data_collector.collection_calls, 0, "Data collector was not called")
        
        # Verify that a MODEL_UPDATED event was published
        self.assertEqual(len(self.model_update_events), 1, "MODEL_UPDATED event was not published")
        event = self.model_update_events[0]
        self.assertEqual(event.data["model_id"], "test_price_model")
        self.assertTrue(event.data["success"])
        
        # Update the training job
        updated = self.training_workflow.update_training_job(
            model_id="test_price_model",
            hyperparameters={"learning_rate": 0.02},
            data_window_days=15
        )
        
        self.assertTrue(updated, "Failed to update training job")
        
        # Verify the updates were applied
        job = self.training_workflow.get_training_job("test_price_model")
        self.assertEqual(job["hyperparameters"]["learning_rate"], 0.02)
        self.assertEqual(job["data_window_days"], 15)
        
        # Remove the training job
        removed = self.training_workflow.remove_training_job("test_price_model")
        self.assertTrue(removed, "Failed to remove training job")
        
        # Verify the job was removed
        job = self.training_workflow.get_training_job("test_price_model")
        self.assertIsNone(job, "Training job was not removed")
    
    def test_multiple_model_training(self):
        """
        Test training multiple models.
        
        This test validates that multiple models can be trained
        and that their artifacts are correctly managed.
        """
        # Define test tokens
        test_tokens = ["TOKEN_A", "TOKEN_C"]
        
        # Add two training jobs
        self.training_workflow.add_training_job(
            model_id="model_a",
            model_type="regression",
            feature_list=self.trainer.get_required_features()[:2],  # Subset of features
            token_ids=test_tokens,
            data_window_days=20
        )
        
        self.training_workflow.add_training_job(
            model_id="model_b",
            model_type="regression",
            feature_list=self.trainer.get_required_features(),  # All features
            token_ids=test_tokens[:1],  # Just one token
            data_window_days=10
        )
        
        # Train both models
        result_a = self.training_workflow.train_model("model_a")
        result_b = self.training_workflow.train_model("model_b")
        
        # Verify both trainings were successful
        self.assertTrue(result_a["success"], "Training model_a failed")
        self.assertTrue(result_b["success"], "Training model_b failed")
        
        # Verify different model paths
        self.assertNotEqual(result_a["model_path"], result_b["model_path"], 
                           "Models have the same path")
        
        # Verify that both model files exist
        self.assertTrue(os.path.exists(result_a["model_path"]), "model_a file does not exist")
        self.assertTrue(os.path.exists(result_b["model_path"]), "model_b file does not exist")
        
        # Verify correct number of training and evaluation calls
        self.assertEqual(self.trainer.training_calls, 2, "Trainer was not called twice")
        self.assertEqual(self.evaluator.evaluation_calls, 2, "Evaluator was not called twice")
        
        # Verify MODEL_UPDATED events for both models
        self.assertEqual(len(self.model_update_events), 2, "Expected two MODEL_UPDATED events")
        model_ids = [event.data["model_id"] for event in self.model_update_events]
        self.assertIn("model_a", model_ids, "No event for model_a")
        self.assertIn("model_b", model_ids, "No event for model_b")
    
    def test_training_with_insufficient_data(self):
        """
        Test training with insufficient data.
        
        This test validates the behavior when there's not enough
        data to train a model based on min_training_samples.
        """
        # Add a training job with a very high min_training_samples
        self.training_workflow.add_training_job(
            model_id="insufficient_data_model",
            model_type="regression",
            min_training_samples=1000000,  # Unreasonably high
            data_window_days=1  # Very short window
        )
        
        # Try to train the model
        result = self.training_workflow.train_model("insufficient_data_model")
        
        # Verify training failed due to insufficient data
        self.assertFalse(result["success"], "Training should have failed")
        self.assertIn("error", result, "No error message in result")
        self.assertIn("insufficient data", result["error"].lower(), 
                     "Error message does not mention insufficient data")
        
        # Verify no model file was created
        self.assertNotIn("model_path", result, "Model path should not be in result")
        
        # Verify the correct components were called
        self.assertEqual(self.trainer.training_calls, 0, "Trainer should not have been called")
        self.assertEqual(self.evaluator.evaluation_calls, 0, "Evaluator should not have been called")
        self.assertGreater(self.data_collector.collection_calls, 0, "Data collector was not called")


if __name__ == "__main__":
    unittest.main() 