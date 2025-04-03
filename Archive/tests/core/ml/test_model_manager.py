#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the DefaultModelManager implementation.
"""

import os
import unittest
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import pickle
import joblib

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification

from src.core.ml.model_manager import DefaultModelManager
from src.core.ml.adapters.scikit_learn import ScikitLearnModel, ScikitLearnModelAdapter
from src.core.ml.transformers.standard import StandardFeatureTransformer
from src.core.events.event_bus import EventBus
from src.core.events.event import Event, EventType


class TestDefaultModelManager(unittest.TestCase):
    """Test suite for the DefaultModelManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = os.path.join(self.temp_dir.name, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.event_bus = Mock(spec=EventBus)
        self.event_bus.subscribe = Mock()
        self.event_bus.publish = Mock()
        
        self.model_manager = DefaultModelManager(
            model_dir=self.model_dir,
            event_bus=self.event_bus
        )
        
        # Create test models
        self._create_test_models()
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    def _create_test_models(self):
        """Create test models for testing."""
        # Create regression model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        reg_model = LinearRegression()
        reg_model.fit(X, y)
        
        # Create classification model
        X_cls, y_cls = make_classification(n_samples=100, n_features=5, random_state=42)
        cls_model = LogisticRegression()
        cls_model.fit(X_cls, y_cls)
        
        # Save models
        self.reg_model_path = os.path.join(self.model_dir, "regression_model.joblib")
        self.cls_model_path = os.path.join(self.model_dir, "classification_model.joblib")
        
        joblib.dump(reg_model, self.reg_model_path)
        joblib.dump(cls_model, self.cls_model_path)
        
        # Create features
        self.feature_names = [f"feature_{i}" for i in range(5)]
        self.reg_features = {name: X[0][i] for i, name in enumerate(self.feature_names)}
        self.cls_features = {name: X_cls[0][i] for i, name in enumerate(self.feature_names)}
    
    def test_load_model(self):
        """Test loading a model from a file."""
        model_id = "regression_model"
        model = self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Check model is loaded correctly
        self.assertIsInstance(model, ScikitLearnModel)
        self.assertEqual(model.get_id(), model_id)
        self.assertEqual(model.get_type(), "regression")
        
        # Check model is stored in manager
        self.assertIn(model_id, self.model_manager.list_models())
        
        # Check event was published
        self.event_bus.publish.assert_called_with(
            Event(
                event_type=EventType.MODEL_LOADED,
                data={"model_id": model_id, "model_type": "regression"}
            )
        )
    
    def test_load_model_file_not_found(self):
        """Test loading a model from a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.model_manager.load_model("missing_model", "nonexistent_file.joblib")
    
    def test_get_prediction_without_transformer(self):
        """Test getting a prediction from a model without a transformer."""
        model_id = "regression_model"
        self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Get prediction
        prediction = self.model_manager.get_prediction(model_id, self.reg_features)
        
        # Check prediction
        self.assertIsInstance(prediction, np.ndarray)
        
        # Check event was published
        self.event_bus.publish.assert_any_call(
            Event(
                event_type=EventType.MODEL_PREDICTION,
                data={"model_id": model_id, "features": self.reg_features}
            )
        )
        
        # Check prediction is recorded in history
        self.assertIn(model_id, self.model_manager._prediction_history)
        self.assertEqual(len(self.model_manager._prediction_history[model_id]), 1)
    
    def test_get_prediction_with_transformer(self):
        """Test getting a prediction from a model with a transformer."""
        model_id = "regression_model"
        self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Create and register transformer
        transformer = StandardFeatureTransformer(self.feature_names)
        transformer.fit({name: np.array([i]) for i, name in enumerate(self.feature_names)})
        self.model_manager.register_transformer(model_id, transformer)
        
        # Get prediction
        prediction = self.model_manager.get_prediction(model_id, self.reg_features)
        
        # Check prediction
        self.assertIsInstance(prediction, np.ndarray)
    
    def test_update_model(self):
        """Test updating a model with new data."""
        model_id = "regression_model"
        self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Mock the update method since LinearRegression doesn't support online learning
        with patch.object(ScikitLearnModel, 'update', return_value=True):
            # Update model
            result = self.model_manager.update_model(
                model_id, self.reg_features, np.array([1.0])
            )
            
            # Check result
            self.assertTrue(result)
            
            # Check event was published
            self.event_bus.publish.assert_any_call(
                Event(
                    event_type=EventType.MODEL_UPDATED,
                    data={"model_id": model_id, "features": self.reg_features}
                )
            )
    
    def test_register_transformer(self):
        """Test registering a transformer for a model."""
        model_id = "regression_model"
        self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Create transformer
        transformer = StandardFeatureTransformer(self.feature_names)
        
        # Register transformer
        self.model_manager.register_transformer(model_id, transformer)
        
        # Check transformer is registered
        self.assertIn(model_id, self.model_manager._transformers)
        self.assertEqual(self.model_manager._transformers[model_id], transformer)
    
    def test_get_model_performance(self):
        """Test getting performance metrics for a model."""
        model_id = "regression_model"
        self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Set some performance metrics
        self.model_manager._performance_metrics[model_id] = {
            "mae": 0.5,
            "mse": 0.25,
            "r2": 0.8
        }
        
        # Get performance metrics
        metrics = self.model_manager.get_model_performance(model_id)
        
        # Check metrics
        self.assertEqual(metrics["mae"], 0.5)
        self.assertEqual(metrics["mse"], 0.25)
        self.assertEqual(metrics["r2"], 0.8)
    
    def test_list_models(self):
        """Test listing all available models."""
        # Load multiple models
        self.model_manager.load_model("regression_model", self.reg_model_path)
        self.model_manager.load_model("classification_model", self.cls_model_path)
        
        # List models
        models = self.model_manager.list_models()
        
        # Check models
        self.assertIn("regression_model", models)
        self.assertIn("classification_model", models)
        self.assertEqual(len(models), 2)
    
    def test_get_model_metadata(self):
        """Test getting metadata for a model."""
        model_id = "regression_model"
        model = self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Set some metadata
        model._metadata = {
            "created_at": "2023-01-01",
            "version": "1.0",
            "description": "Test model"
        }
        
        # Get metadata
        metadata = self.model_manager.get_model_metadata(model_id)
        
        # Check metadata
        self.assertEqual(metadata["created_at"], "2023-01-01")
        self.assertEqual(metadata["version"], "1.0")
        self.assertEqual(metadata["description"], "Test model")
    
    def test_evaluate_regression_model(self):
        """Test evaluating a regression model."""
        model_id = "regression_model"
        self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Create evaluation data
        X, y = make_regression(n_samples=20, n_features=5, random_state=43)
        eval_data = [
            {self.feature_names[i]: X[j][i] for i in range(5)}
            for j in range(20)
        ]
        
        # Evaluate model
        metrics = self.model_manager.evaluate_model(model_id, eval_data, y)
        
        # Check metrics
        self.assertIn("mae", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("r2", metrics)
        
        # Check metrics are stored
        self.assertEqual(
            self.model_manager._performance_metrics[model_id],
            metrics
        )
    
    def test_evaluate_classification_model(self):
        """Test evaluating a classification model."""
        model_id = "classification_model"
        self.model_manager.load_model(model_id, self.cls_model_path)
        
        # Create evaluation data
        X, y = make_classification(n_samples=20, n_features=5, random_state=43)
        eval_data = [
            {self.feature_names[i]: X[j][i] for i in range(5)}
            for j in range(20)
        ]
        
        # Evaluate model
        metrics = self.model_manager.evaluate_model(model_id, eval_data, y)
        
        # Check metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Check metrics are stored
        self.assertEqual(
            self.model_manager._performance_metrics[model_id],
            metrics
        )
    
    def test_save_model(self):
        """Test saving a model to a file."""
        model_id = "regression_model"
        model = self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Save model to a new path
        new_path = os.path.join(self.model_dir, "new_regression_model.joblib")
        result = self.model_manager.save_model(model_id, new_path)
        
        # Check result
        self.assertTrue(result)
        
        # Check file exists
        self.assertTrue(os.path.exists(new_path))
    
    def test_save_transformer(self):
        """Test saving a transformer to a file."""
        model_id = "regression_model"
        self.model_manager.load_model(model_id, self.reg_model_path)
        
        # Create and register transformer
        transformer = StandardFeatureTransformer(self.feature_names)
        transformer.fit({name: np.array([i]) for i, name in enumerate(self.feature_names)})
        self.model_manager.register_transformer(model_id, transformer)
        
        # Save transformer to a new path
        new_path = os.path.join(self.model_dir, "transformer.pkl")
        result = self.model_manager.save_transformer(model_id, new_path)
        
        # Check result
        self.assertTrue(result)
        
        # Check file exists
        self.assertTrue(os.path.exists(new_path))


if __name__ == '__main__':
    unittest.main() 