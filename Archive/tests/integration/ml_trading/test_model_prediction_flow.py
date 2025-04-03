#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test for Model Prediction Flow

This module tests the complete flow from model predictions to trading signals,
validating that MODEL_PREDICTION events are properly consumed by the SignalGenerator
and influence the generation of trading signals.
"""

import unittest
import logging
import time
from typing import Dict, Any
from datetime import datetime

from src.core.events import EventType, Event
from src.core.trading.interfaces import TradingSignal
from src.core.ml.exceptions import ModelNotFoundError
from tests.integration.base_integration_test import BaseIntegrationTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelPredictionMock:
    """Mock model for testing predictions."""
    
    def __init__(self, prediction_value: float = 0.8):
        self.prediction_value = prediction_value
        self.call_count = 0
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Make a mock prediction.
        
        Args:
            features: Input features
            
        Returns:
            Mock prediction value
        """
        self.call_count += 1
        return self.prediction_value


class TestModelPredictionFlow(BaseIntegrationTest):
    """
    Integration tests for MODEL_PREDICTION event flow from ModelManager to SignalGenerator.
    """
    
    def setUp(self) -> None:
        """Set up the test environment."""
        super().setUp()
        
        # Create and initialize components
        self.model_manager = self.create_ml_components()
        self.trading_engine = self.create_trading_components()
        
        # Mock model for testing
        self.mock_model = ModelPredictionMock(prediction_value=0.8)
        
        # Mock signal generator to track signal generation
        self.original_evaluate_model_prediction = self.trading_engine.signal_generator.evaluate_model_prediction
        self.signal_generator_calls = 0
        
        def mock_evaluate_model_prediction(token_id, prediction, features, timestamp):
            self.signal_generator_calls += 1
            return self.original_evaluate_model_prediction(token_id, prediction, features, timestamp)
        
        self.trading_engine.signal_generator.evaluate_model_prediction = mock_evaluate_model_prediction
        
        # Create a test token
        self.test_token_id = "TEST_TOKEN_123"
        self.mock_features = self.generate_mock_features(self.test_token_id)
    
    def tearDown(self) -> None:
        """Clean up after the test."""
        # Restore original methods
        if hasattr(self, 'original_evaluate_model_prediction'):
            self.trading_engine.signal_generator.evaluate_model_prediction = self.original_evaluate_model_prediction
        
        super().tearDown()
    
    def test_model_prediction_generates_trading_signal(self):
        """
        Test that a MODEL_PREDICTION event leads to a trading signal.
        
        This test validates that:
        1. A MODEL_PREDICTION event is properly published
        2. The SignalGenerator receives and processes the prediction
        3. The trading signal is influenced by the model prediction
        """
        # Publish mock model prediction with positive value (buy signal)
        prediction_value = 0.8  # Strong buy signal
        self.publish_mock_prediction(self.test_token_id, prediction_value)
        
        # Wait for the event to be processed
        self.assertTrue(
            self.event_capture.wait_for_event(EventType.MODEL_PREDICTION),
            "MODEL_PREDICTION event was not captured"
        )
        
        # Update features with the prediction
        self.mock_features["model_prediction"] = prediction_value
        
        # Generate signals using the features with prediction
        signals = self.trading_engine.signal_generator.generate_signals(
            self.test_token_id,
            self.mock_features,
            datetime.now()
        )
        
        # Verify signals were generated
        self.assertGreater(len(signals), 0, "No trading signals were generated")
        
        # Verify at least one signal is an entry signal with positive score
        entry_signals = [s for s in signals if s.signal_type == 'entry']
        self.assertGreater(len(entry_signals), 0, "No entry signals were generated")
        
        # Verify signal scores are influenced by the model prediction
        for signal in entry_signals:
            logger.info(f"Signal: type={signal.signal_type}, score={signal.score}")
            self.assertGreater(signal.score, 0.5, "Signal score not influenced by positive model prediction")
    
    def test_model_prediction_influences_signal_strength(self):
        """
        Test that different model prediction values result in different signal strengths.
        
        This test validates that:
        1. The signal score is proportional to the model prediction value
        2. The model weight parameter affects the signal strength
        """
        # Test with different prediction values
        test_values = [-0.8, -0.2, 0.0, 0.2, 0.8]
        signal_scores = []
        
        for prediction_value in test_values:
            # Update features with the prediction
            self.mock_features["model_prediction"] = prediction_value
            
            # Generate signals
            signals = self.trading_engine.signal_generator.generate_signals(
                self.test_token_id,
                self.mock_features,
                datetime.now()
            )
            
            # Record signal scores
            if signals:
                # Get the highest magnitude score
                max_score = max(signals, key=lambda s: abs(s.score)).score
                signal_scores.append(max_score)
            else:
                signal_scores.append(0.0)
        
        # Verify score progression (should generally follow the trend of prediction values)
        logger.info(f"Prediction values: {test_values}")
        logger.info(f"Signal scores: {signal_scores}")
        
        # Check that extreme predictions lead to more extreme signals
        self.assertLess(signal_scores[0], signal_scores[2], "Negative prediction should give lower score than neutral")
        self.assertGreater(signal_scores[4], signal_scores[2], "Positive prediction should give higher score than neutral")
    
    def test_model_weight_parameter_affects_signal(self):
        """
        Test that the model_weight parameter affects the influence of model predictions.
        
        This test validates that:
        1. Higher model_weight increases the model's influence on signal generation
        2. Lower model_weight decreases the model's influence
        """
        # Test with different model weights
        test_weights = [0.1, 0.5, 0.9]
        prediction_value = 0.8  # Strong buy signal
        signal_scores = []
        
        # Save original model weight
        original_weight = self.trading_engine.signal_generator._params['model_weight']
        
        try:
            for weight in test_weights:
                # Set model weight
                self.trading_engine.signal_generator._params['model_weight'] = weight
                
                # Update features with the prediction
                self.mock_features["model_prediction"] = prediction_value
                
                # Generate signals
                signals = self.trading_engine.signal_generator.generate_signals(
                    self.test_token_id,
                    self.mock_features,
                    datetime.now()
                )
                
                # Record signal scores
                if signals:
                    # Get the highest score (for entry signals)
                    entry_signals = [s for s in signals if s.signal_type == 'entry']
                    if entry_signals:
                        max_score = max(entry_signals, key=lambda s: s.score).score
                        signal_scores.append(max_score)
                    else:
                        signal_scores.append(0.0)
                else:
                    signal_scores.append(0.0)
        finally:
            # Restore original model weight
            self.trading_engine.signal_generator._params['model_weight'] = original_weight
        
        # Verify that higher weights lead to scores closer to the prediction value
        logger.info(f"Model weights: {test_weights}")
        logger.info(f"Signal scores: {signal_scores}")
        
        # Score differences should decrease as weight increases
        if len(signal_scores) >= 3:
            self.assertLessEqual(
                abs(prediction_value - signal_scores[2]),
                abs(prediction_value - signal_scores[0]),
                "Higher weight should make score closer to prediction value"
            )


if __name__ == "__main__":
    unittest.main() 